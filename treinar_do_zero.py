"""Treina todas as camadas da EfficientNet, sem carregar pesos pré-treinados."""
import argparse
import hashlib
import json
import math
import random
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from animal_classifier import AnimalDataset, build_model, device
from evaluation_metrics import summarize_confusion

ROOT = Path(__file__).resolve().parent


def write_json(path, value):
    # Primeiro gravamos um temporário; assim um leitor não encontra JSON pela metade.
    temporary = path.with_suffix('.tmp.json')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def save_checkpoint(path, value):
    temporary = path.with_suffix('.tmp.pt')
    torch.save(value, temporary)
    temporary.replace(path)


@torch.inference_mode()
def evaluate(model, loader, classes):
    # Avaliação não altera os pesos nem as estatísticas das camadas BatchNorm.
    model.eval()
    confusion = torch.zeros(len(classes), len(classes), dtype=torch.int64)
    total_loss, count = 0., 0
    for images, labels in loader:
        images, labels = images.to(device()), labels.to(device())
        logits = model(images)
        total_loss += nn.functional.cross_entropy(logits, labels, reduction='sum').item()
        count += len(labels)
        pairs = labels * len(classes) + logits.argmax(1)
        confusion += torch.bincount(pairs.cpu(), minlength=len(classes)**2).reshape(confusion.shape)
    report = summarize_confusion(confusion.tolist(), classes)
    return {**report, 'loss': total_loss / count, 'confusion_matrix': confusion.tolist(),
            'accuracy_tipo_raca': report['accuracy']}


def train(args, animal):
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    manifest_dir = Path(getattr(args, 'manifest_dir', None) or ROOT / 'datasets')
    manifest_path = manifest_dir / ('oxford_cats.json' if animal == 'gato' else 'stanford_dogs.json')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    classes, splits = manifest['classes'], manifest['splits']
    if len(classes) < 2 or any(c['animal'] != animal for c in classes):
        raise ValueError('Classes incompatíveis com a espécie selecionada.')
    if any(not splits.get(name) for name in ('train', 'val', 'test')):
        raise ValueError('Treino, validação e teste devem conter imagens.')
    # A assinatura inclui os bytes das fotos: mudanças nos dados impedem uma retomada incompatível.
    digest = hashlib.sha256(manifest_path.read_bytes())
    seen = set()
    for samples in splits.values():
        for path, label in samples:
            if not isinstance(label, int) or not 0 <= label < len(classes):
                raise ValueError('Rótulo inválido no manifesto.')
            image_hash = hashlib.sha256(Path(path).read_bytes()).digest()
            if image_hash in seen:
                raise ValueError('Imagem duplicada entre os registros do manifesto.')
            seen.add(image_hash)
            digest.update(image_hash)
    signature = digest.hexdigest()
    output = Path(args.output).resolve() / animal
    output.mkdir(parents=True, exist_ok=True)
    last = output / 'last.pt'
    if last.exists() and not args.resume:
        raise ValueError(f'Já existe treino em {output}. Use --resume ou outra pasta --output.')
    write_json(output / 'manifest.json', manifest)
    settings = {k: getattr(args, k) for k in ('epochs', 'patience', 'batch_size', 'lr', 'seed')}
    # pretrained=False cria pesos aleatórios. Nenhum arquivo ImageNet ou cache de características é lido.
    model = build_model(len(classes), pretrained=False).to(device())
    if not all(p.requires_grad for p in model.parameters()):
        raise ValueError('Todas as camadas precisam estar liberadas para aprender.')
    parent_hash = None
    # Dados novos exigem um novo experimento. Podemos aproveitar exclusivamente os
    # nossos pesos aprendidos do zero; isso não é uma retomada do otimizador antigo.
    init_root = getattr(args, 'init_root', None)
    if init_root and not last.exists():
        initial_path = Path(init_root) / animal / 'best.pt'
        initial = torch.load(initial_path, map_location='cpu', weights_only=True)
        if (initial.get('pretrained') is not False or initial.get('training') not in
                {'random_initialization_all_layers', 'continued_own_scratch_all_layers'}):
            raise ValueError('A continuação aceita somente pesos próprios com origem no treinamento do zero.')
        if initial['classes'] != classes:
            raise ValueError('A continuação exige a mesma ordem e quantidade de classes.')
        model.load_state_dict(initial['state_dict'])
        parent_hash = hashlib.sha256(initial_path.read_bytes()).hexdigest()
        print(f'{animal}: continuando dos pesos próprios {initial_path}, época original {initial["epoch"]}', flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=.01)
    loaders = {name: DataLoader(AnimalDataset(samples, training=name == 'train'),
                               batch_size=args.batch_size, shuffle=name == 'train', num_workers=0)
               for name, samples in splits.items()}
    history, best, stale, start = [], float('inf'), 0, 0
    if last.exists():
        saved = torch.load(last, map_location='cpu', weights_only=True)
        if saved['source_manifest_sha256'] != signature or saved['settings'] != settings:
            raise ValueError('A retomada exige os mesmos dados e parâmetros do treinamento original.')
        model.load_state_dict(saved['state_dict'])
        optimizer.load_state_dict(saved['optimizer'])
        history, best, stale, start = saved['history'], saved['best_loss'], saved['stale'], saved['epoch']
        torch.set_rng_state(saved['rng'])
        random.setstate(saved['python_rng'])
        parent_hash = saved.get('initial_checkpoint_sha256')
        if torch.cuda.is_available() and saved.get('cuda_rng') is not None:
            torch.cuda.set_rng_state_all(saved['cuda_rng'])
    status_path = output / 'status.json'
    if parent_hash and not last.exists():
        # A rede de origem também é candidata a melhor modelo. A continuação só
        # substitui esses pesos se reduzir a perda no mesmo conjunto de validação.
        baseline = evaluate(model, loaders['val'], classes)
        best = baseline['loss']
        write_json(output / 'baseline_validation.json', baseline)
        save_checkpoint(output / 'best.pt', {
            'state_dict': model.state_dict(), 'classes': classes, 'architecture': 'efficientnet_b0',
            'training': 'continued_own_scratch_all_layers', 'pretrained': False, 'epoch': 0,
            'initial_checkpoint_sha256': parent_hash, 'source_manifest_sha256': signature, 'settings': settings})
        print(f'{animal}: referência antes da continuação: val_loss={best:.4f}, '
              f'acurácia={baseline["accuracy"]:.2%}', flush=True)
    print(f'{animal}: {len(classes)} classes; {device()}; todos os pesos treináveis; início época {start + 1}', flush=True)
    stopped = stale >= args.patience
    for epoch in range(start, args.epochs):
        if stopped:
            break
        # A taxa de aprendizado diminui suavemente. As fotos variam a cada época pelas transformações de treino.
        lr = args.lr * (.05 + .95 * (1 + math.cos(math.pi * epoch / args.epochs)) / 2)
        for group in optimizer.param_groups:
            group['lr'] = lr
        model.train()
        loss_sum, count = 0., 0
        started = time.monotonic()
        for batch, (images, labels) in enumerate(loaders['train'], 1):
            images, labels = images.to(device()), labels.to(device())
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(model(images), labels)
            if not torch.isfinite(loss):
                raise RuntimeError('Perda não finita; interrompido para não gravar pesos inválidos.')
            # O gradiente atravessa toda a rede, incluindo as primeiras convoluções.
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
            loss_sum += loss.item() * len(labels)
            count += len(labels)
            if batch == 1 or batch % 20 == 0 or batch == len(loaders['train']):
                status = {'estado': 'treinando', 'animal': animal, 'epoch': epoch + 1,
                          'epochs': args.epochs, 'batch': batch, 'batches': len(loaders['train']),
                          'train_loss': loss_sum / count, 'segundos_epoca': time.monotonic() - started}
                write_json(status_path, status)
                print(f'{animal} época {epoch + 1}/{args.epochs}, lote {batch}/{len(loaders["train"])}: loss={loss_sum/count:.4f}', flush=True)
        # A validação escolhe os pesos. O conjunto de teste não participa dessa escolha.
        validation = evaluate(model, loaders['val'], classes)
        history.append({'epoch': epoch + 1, 'train_loss': loss_sum/count,
                        'val_loss': validation['loss'], 'val_accuracy': validation['accuracy'], 'lr': lr,
                        'seconds': time.monotonic() - started})
        improved = validation['loss'] < best
        best, stale = (validation['loss'], 0) if improved else (best, stale + 1)
        saved = {'state_dict': model.state_dict(), 'classes': classes, 'architecture': 'efficientnet_b0',
                 'training': 'continued_own_scratch_all_layers' if parent_hash else 'random_initialization_all_layers',
                 'pretrained': False, 'initial_checkpoint_sha256': parent_hash, 'epoch': epoch + 1,
                 'source_manifest_sha256': signature, 'settings': settings}
        if improved:
            save_checkpoint(output / 'best.pt', saved)
        # last guarda também o otimizador e os sorteios para continuar da última época completa.
        save_checkpoint(last, {**saved, 'optimizer': optimizer.state_dict(), 'history': history,
                               'best_loss': best, 'stale': stale, 'rng': torch.get_rng_state(),
                               'python_rng': random.getstate(),
                               'cuda_rng': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None})
        write_json(output / 'history.json', history)
        stopped = stale >= args.patience
        print(f'{animal}: validação acurácia={validation["accuracy"]:.2%}, loss={validation["loss"]:.4f}', flush=True)
    # Uma retomada de um treino concluído não precisa consultar o teste novamente.
    if (output / 'test_metrics.json').exists():
        print(f'{animal}: avaliação já concluída.', flush=True)
        return
    write_json(status_path, {'estado': 'avaliando_teste', 'animal': animal, 'epoch': len(history)})
    model.load_state_dict(torch.load(output / 'best.pt', map_location='cpu', weights_only=True)['state_dict'])
    report = evaluate(model, loaders['test'], classes)
    report.update(checkpoint_sha256=hashlib.sha256((output / 'best.pt').read_bytes()).hexdigest(),
                  training='continued_own_scratch_all_layers' if parent_hash else 'random_initialization_all_layers',
                  pretrained=False, initial_checkpoint_sha256=parent_hash, source=manifest.get('source'),
                  note=('Continuação dos nossos pesos originados do zero; teste preservado, já consultado no experimento anterior. '
                        'Não equivale a teste independente final.' if parent_hash else
                        'Rede inteira treinada do zero; teste reservado da base, sem pré-treinamento ImageNet.'))
    write_json(output / 'test_metrics.json', report)
    write_json(status_path, {'estado': 'concluido', 'animal': animal, 'epoch': len(history),
                             'accuracy': report['accuracy'], 'macro_f1': report['macro_f1']})
    print(f'{animal}: concluído; acurácia={report["accuracy"]:.2%}, F1={report["macro_f1"]:.4f}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--animal', choices=['gato', 'cachorro', 'todos'], default='todos')
    parser.add_argument('--output', default=str(ROOT / 'outputs' / 'do_zero'))
    parser.add_argument('--manifest-dir', help='Pasta com os manifestos; o padrão mantém os dados originais.')
    parser.add_argument('--init-root', help='Pasta dos pesos próprios do zero; inicia uma continuação em nova saída.')
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=.001)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if min(args.epochs, args.patience, args.threads) < 1 or args.batch_size < 2 or not 0 < args.lr < float('inf'):
        parser.error('Parâmetros positivos obrigatórios; batch-size deve ser pelo menos 2.')
    torch.set_num_threads(args.threads)
    # Os gatos têm menos imagens: a fila começa por eles e depois segue para os cães.
    for animal in (['gato', 'cachorro'] if args.animal == 'todos' else [args.animal]):
        train(args, animal)


if __name__ == '__main__':
    main()
