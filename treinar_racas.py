"""Treina cabeça própria EfficientNet em características locais congeladas (CPU/GPU)."""
# Treina uma nova camada de raças sobre características da EfficientNet pré-treinada e congelada. Cães e gatos usam manifestos e diretórios de saída separados.
import argparse
import json
import random
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import models

from animal_classifier import AnimalDataset, device
from evaluation_metrics import summarize_confusion

ROOT = Path(__file__).resolve().parent
CHECKPOINT = ROOT / 'outputs' / 'treino_proprio' / 'best.pt'
BASE = ROOT / 'modelos' / 'efficientnet_b0_rwightman-7f5810bc.pth'


# Resume os acertos e a matriz de confusão para acompanhar o classificador.
def metrics(head, features, labels, classes):
    head.eval()
    with torch.inference_mode():
        logits = head(features.to(device())).cpu()
        loss = nn.functional.cross_entropy(logits, labels).item()
    predictions = logits.argmax(-1)
    confusion = torch.zeros(len(classes), len(classes), dtype=torch.int64)
    for actual, predicted in zip(labels.tolist(), predictions.tolist()):
        confusion[actual, predicted] += 1
    summary = summarize_confusion(confusion.tolist(), classes)
    return {**summary, 'loss': loss, 'accuracy_tipo_raca': summary['accuracy'],
            'top5_accuracy': (logits.topk(min(5, len(classes)), -1).indices == labels[:, None]).any(-1).float().mean().item(),
            'confusion_matrix': confusion.tolist()}


# Extrai ou reutiliza características em cache, treina a camada final e escolhe o melhor checkpoint pela validação. Só depois avalia o teste.
def train(args):
    # Fixamos as sementes para tornar os sorteios mais reproduzíveis nas mesmas condições de execução.
    random.seed(42)
    torch.manual_seed(42)
    torch.set_num_threads(args.threads)
    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    classes, splits = manifest['classes'], manifest['splits']
    if len(classes) < 2 or any(not splits[name] for name in ['train', 'val', 'test']):
        raise ValueError('Manifesto de dados incompleto.')
    checkpoint = Path(args.output).resolve() / 'best.pt'
    output = checkpoint.parent
    output.mkdir(parents=True, exist_ok=True)
    (output / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    # Os pesos ImageNet fornecem as características visuais; substituímos a saída original pela quantidade de raças do manifesto.
    model = models.efficientnet_b0(weights=None)
    model.load_state_dict(torch.load(BASE, map_location='cpu', weights_only=True))
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(classes))
    model.to(device()).eval()
    for parameter in model.features.parameters():
        parameter.requires_grad = False
    # Cache invalidado quando o manifesto ou os pesos-base mudam.
    import hashlib
    digest = hashlib.sha256(manifest_path.read_bytes() + BASE.read_bytes())
    for samples in splits.values():
        for path, label in samples:
            digest.update(hashlib.sha256(Path(path).read_bytes()).digest())
    signature = digest.hexdigest()
    # Cada foto vira um vetor de características. O cache só é reutilizado se manifesto, pesos e conteúdo das imagens mantiverem a mesma assinatura.
    features = {}
    print(f'Treinamento próprio: {len(classes)} raças, dispositivo {device()}', flush=True)
    for split in ['train', 'val', 'test']:
        cache = output / f'features_{split}.pt'
        saved = torch.load(cache, weights_only=True) if cache.exists() else None
        if saved and saved.get('signature') == signature:
            features[split] = saved['features'], saved['labels']
            print(f'Cache reutilizado: {split}', flush=True)
            continue
        loader = DataLoader(AnimalDataset(splits[split]), batch_size=args.batch_size,
                            num_workers=0, shuffle=False)
        all_features, all_labels = [], []
        started = time.monotonic()
        with torch.inference_mode():
            for batch, (images, labels) in enumerate(loader):
                tensor = model.avgpool(model.features(images.to(device()))).flatten(1).cpu()
                all_features.append(tensor)
                all_labels.append(labels)
                if batch % 20 == 0 or batch + 1 == len(loader):
                    print(f'Características {split}: lote {batch+1}/{len(loader)} ({time.monotonic()-started:.0f}s)', flush=True)
        features[split] = torch.cat(all_features), torch.cat(all_labels)
        torch.save({'signature': signature, 'features': features[split][0], 'labels': features[split][1]}, cache)
    # Só a camada final aprende novos pesos. A rede que extrai características permanece congelada.
    optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=args.lr, weight_decay=.01)
    train_loader = DataLoader(TensorDataset(*features['train']), batch_size=128, shuffle=True)
    history, best, stale = [], float('inf'), 0
    for epoch in range(args.epochs):
        model.classifier.train()
        running = 0.
        for batch_features, labels in train_loader:
            # Limpamos os gradientes, calculamos o erro, propagamos esse erro e atualizamos os pesos da camada final.
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(model.classifier(batch_features.to(device())), labels.to(device()))
            loss.backward()
            optimizer.step()
            running += loss.item() * len(labels)
        # A validação acompanha o treino sem atualizar pesos. Ela decide qual época salvar e quando parar por falta de melhora.
        validation = metrics(model.classifier, *features['val'], classes)
        history.append({'epoch': epoch+1, 'train_loss': running/len(features['train'][1]),
                        'val_loss': validation['loss'], 'val_accuracy': validation['accuracy_tipo_raca']})
        (output / 'history.json').write_text(json.dumps(history, indent=2), encoding='utf-8')
        print(f"Época {epoch+1}: val_loss={validation['loss']:.4f}; val_acerto={validation['accuracy_tipo_raca']:.2%}", flush=True)
        if validation['loss'] < best:
            best, stale = validation['loss'], 0
            # Gravamos primeiro em arquivo temporário e só então substituímos o checkpoint, reduzindo o risco de deixar uma gravação incompleta.
            temp = output / 'best.tmp'
            torch.save({'state_dict': model.cpu().state_dict(), 'classes': classes,
                        'architecture': 'efficientnet_b0', 'epoch': epoch+1,
                        'training': 'frozen_imagenet_features_own_linear_classifier',
                        'source_manifest_sha256': signature}, temp)
            temp.replace(checkpoint)
            model.to(device())
        else:
            stale += 1
        if stale >= args.patience:
            break
    # Recuperamos a melhor época, e não necessariamente a última. O teste reservado produz o relatório final das cinco métricas.
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    model.load_state_dict(saved['state_dict'])
    report = metrics(model.classifier, *features['test'], classes)
    report['checkpoint_sha256'] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    report['source'] = manifest.get('source')
    report['note'] = 'Teste reservado da base publicada; backbone pré-treinado ImageNet. Não equivale a generalização em fotos independentes.'
    (output / 'test_metrics.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"Concluído: {checkpoint}; teste={report['accuracy_tipo_raca']:.2%}, F1={report['macro_f1']:.4f}", flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', default=str(ROOT / 'datasets' / 'stanford_dogs.json'))
    parser.add_argument('--output', default=str(CHECKPOINT.parent))
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--patience', type=int, default=8)
    parser.add_argument('--lr', type=float, default=.003)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    if min(args.epochs, args.patience, args.batch_size, args.threads) < 1 or args.lr <= 0:
        parser.error('Parâmetros de treinamento devem ser positivos.')
    train(args)
