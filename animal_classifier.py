"""Treinamento, avaliação e inferência de tipos e raças de animais."""
# Reúne os componentes básicos de imagens e EfficientNet. O treinamento genérico deste arquivo é uma alternativa; o treino do zero usa treinar_do_zero.py; treinar_racas.py preserva o experimento de transferência.
import argparse
import hashlib
import json
import random
from pathlib import Path

import torch
from PIL import Image, ImageOps
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
WEIGHTS = models.EfficientNet_B0_Weights.DEFAULT


# Prepara a imagem no formato esperado pela rede. No treino, as variações visuais ajudam a reduzir a memorização; na avaliação, a transformação é fixa.
def transform(training=False):
    if not training:
        return WEIGHTS.transforms()
    return transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


# Abre a foto, corrige sua orientação EXIF e padroniza os canais em RGB.
def open_image(path):
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert('RGB')


# Percorre as pastas de espécies e raças, verifica imagens e duplicatas e monta divisões de treino, validação e teste.
def scan_data(root, seed):
    """Divide cada classe em treino/validação/teste sem repetir arquivos."""
    root = Path(root)
    if not root.is_dir():
        raise ValueError(f'Pasta de imagens inexistente: {root}')
    classes, splits, seen = [], {'train': [], 'val': [], 'test': []}, {}
    rng = random.Random(seed)
    for animal in sorted(root.iterdir()):
        if not animal.is_dir():
            continue
        for breed in sorted(animal.iterdir()):
            if not breed.is_dir():
                continue
            files = sorted(p for p in breed.rglob('*') if p.suffix.lower() in EXTENSIONS)
            unique = []
            label = len(classes)
            for path in files:
                # Detecta cópias idênticas mesmo quando têm nomes diferentes.
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                if digest in seen:
                    if seen[digest] != (animal.name, breed.name):
                        raise ValueError(f'Imagem com rótulos conflitantes: {path}')
                    continue
                seen[digest] = (animal.name, breed.name)
                try:
                    open_image(path)
                except (OSError, ValueError) as exc:
                    raise ValueError(f'Imagem inválida: {path}') from exc
                unique.append(path)
            if len(unique) < 10:
                raise ValueError(f'{animal.name}/{breed.name}: mínimo de 10 imagens únicas; '
                                 f'encontradas {len(unique)}. Recomenda-se bem mais para treinar.')
            classes.append({'animal': animal.name, 'raca': breed.name})
            rng.shuffle(unique)
            count = max(1, int(len(unique) * 0.15))
            partitions = {'test': unique[:count], 'val': unique[count:2*count],
                          'train': unique[2*count:]}
            for name, paths in partitions.items():
                splits[name].extend((str(p.resolve()), label) for p in paths)
    if len(classes) < 2:
        raise ValueError('São necessárias pelo menos duas pastas de classes tipo/raça.')
    return classes, splits


# Adapta a lista de fotos ao Dataset do PyTorch, entregando uma imagem transformada e sua classe por acesso.
class AnimalDataset(Dataset):
    # Guarda os registros e a transformação que serão usados ao carregar cada amostra.
    def __init__(self, samples, training=False):
        self.samples = samples
        self.preprocess = transform(training)

    # Informa quantas amostras existem para que o carregador saiba o tamanho do conjunto.
    def __len__(self):
        return len(self.samples)

    # Carrega somente a foto solicitada e devolve o tensor junto de seu rótulo numérico.
    def __getitem__(self, index):
        path, label = self.samples[index]
        return self.preprocess(open_image(path)), label


# Cria a EfficientNet e ajusta a última camada para a quantidade de classes do projeto.
def build_model(num_classes, pretrained=False):
    model = models.efficientnet_b0(weights=WEIGHTS if pretrained else None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model


# Escolhe a GPU quando disponível; caso contrário, usa a CPU.
def device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Reconstrói a rede a partir do checkpoint e recupera a ordem das classes usada no treinamento.
def load_model(checkpoint):
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    model = build_model(len(saved['classes']))
    model.load_state_dict(saved['state_dict'])
    model.to(device()).eval()
    return model, saved['classes']


# Compara previsões com os rótulos reais e acumula os acertos e a matriz de confusão.
@torch.inference_mode()
def evaluate(model, loader, classes):
    model.eval()
    loss_sum, correct, animal_correct, total = 0., 0, 0, 0
    confusion = torch.zeros(len(classes), len(classes), dtype=torch.int64)
    for images, labels in loader:
        images, labels = images.to(device()), labels.to(device())
        logits = model(images)
        loss_sum += nn.functional.cross_entropy(logits, labels, reduction='sum').item()
        predicted = logits.argmax(1)
        correct += (predicted == labels).sum().item()
        for actual, guess in zip(labels.cpu().tolist(), predicted.cpu().tolist()):
            confusion[actual, guess] += 1
            animal_correct += classes[actual]['animal'] == classes[guess]['animal']
        total += len(labels)
    if not total:
        raise ValueError('Conjunto de avaliação vazio.')
    per_class = []
    for i, label in enumerate(classes):
        tp = confusion[i, i].item()
        precision = tp / max(1, confusion[:, i].sum().item())
        recall = tp / max(1, confusion[i].sum().item())
        per_class.append({**label, 'precision': precision, 'recall': recall,
                          'f1': 2 * precision * recall / max(1e-12, precision + recall),
                          'support': confusion[i].sum().item()})
    return {'loss': loss_sum / total, 'accuracy_tipo_raca': correct / total,
            'accuracy_tipo': animal_correct / total,
            'macro_f1': sum(c['f1'] for c in per_class) / len(classes),
            'classes': per_class, 'confusion_matrix': confusion.tolist(), 'samples': total}


# Executa o treinamento genérico, acompanha a validação e salva os pesos do melhor resultado.
def train(args):
    if args.epochs < 1 or args.batch_size < 1 or args.lr <= 0 or args.patience < 1:
        raise ValueError('Épocas, batch, taxa de aprendizado e paciência devem ser positivos.')
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    classes, splits = scan_data(args.data, args.seed)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    manifest = {'classes': classes, 'splits': splits, 'seed': args.seed}
    (output / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    loaders = {name: DataLoader(AnimalDataset(samples, name == 'train'),
                               batch_size=args.batch_size, shuffle=name == 'train',
                               num_workers=args.workers)
               for name, samples in splits.items()}
    model = build_model(len(classes), pretrained=not args.no_pretrained).to(device())
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    best, stale, history = float('inf'), 0, []
    print(f'Dispositivo: {device()}; classes: {len(classes)}')
    for epoch in range(args.epochs):
        model.train()
        total_loss, count = 0., 0
        for images, labels in loaders['train']:
            images, labels = images.to(device()), labels.to(device())
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(model(images), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(labels)
            count += len(labels)
        metrics = evaluate(model, loaders['val'], classes)
        history.append({'epoch': epoch + 1, 'train_loss': total_loss / count, 'val': metrics})
        print(f"Época {epoch+1}: loss={total_loss/count:.4f}, "
              f"val_loss={metrics['loss']:.4f}, acerto={metrics['accuracy_tipo_raca']:.1%}")
        if metrics['loss'] < best:
            best, stale = metrics['loss'], 0
            torch.save({'state_dict': model.state_dict(), 'classes': classes,
                        'architecture': 'efficientnet_b0', 'epoch': epoch+1}, output / 'best.pt')
        else:
            stale += 1
        if stale >= args.patience:
            break
    model, classes = load_model(output / 'best.pt')
    report = evaluate(model, loaders['test'], classes)
    (output / 'history.json').write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding='utf-8')
    (output / 'test_metrics.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"Modelo: {output / 'best.pt'}; acerto no teste: {report['accuracy_tipo_raca']:.1%}")


# Prepara uma foto, calcula os scores das classes e organiza as melhores opções para consulta.
@torch.inference_mode()
def predict(model, classes, image, threshold=0.6, top_k=3):
    if not 0 <= threshold <= 1 or top_k < 1:
        raise ValueError('Limiar deve estar entre 0 e 1 e top_k deve ser positivo.')
    image = ImageOps.exif_transpose(image).convert('RGB')
    model.eval()
    probabilities = model(transform()(image).unsqueeze(0).to(device())).softmax(1)[0].cpu()
    scores, indices = probabilities.topk(min(top_k, len(classes)))
    candidates = [{**classes[i], 'score': score} for score, i in zip(scores.tolist(), indices.tolist())]
    species = {}
    for label, score in zip(classes, probabilities.tolist()):
        species[label['animal']] = species.get(label['animal'], 0.) + score
    return {'aceito': candidates[0]['score'] >= threshold,
            'previsao': candidates[0], 'alternativas': candidates,
            'scores_por_tipo': species, 'limiar': threshold}


# Interpreta os argumentos do terminal e encaminha para treino, avaliação ou previsão.
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    training = commands.add_parser('train', help='Treinar e avaliar em teste reservado')
    training.add_argument('--data', default='data')
    training.add_argument('--output', default='outputs')
    training.add_argument('--epochs', type=int, default=20)
    training.add_argument('--batch-size', type=int, default=16)
    training.add_argument('--lr', type=float, default=0.0001)
    training.add_argument('--patience', type=int, default=5)
    training.add_argument('--seed', type=int, default=42)
    training.add_argument('--workers', type=int, default=0)
    training.add_argument('--no-pretrained', action='store_true', help='Sem download; treina do zero')
    inference = commands.add_parser('predict', help='Classificar uma foto')
    inference.add_argument('--model', default='outputs/best.pt')
    inference.add_argument('--image', required=True)
    inference.add_argument('--threshold', type=float, default=0.6)
    inference.add_argument('--top-k', type=int, default=3)
    evaluation = commands.add_parser('evaluate', help='Reavaliar o teste salvo no manifesto')
    evaluation.add_argument('--model', default='outputs/best.pt')
    evaluation.add_argument('--manifest', default='outputs/manifest.json')
    args = parser.parse_args()
    if args.command == 'train':
        train(args)
    elif args.command == 'predict':
        model, classes = load_model(args.model)
        print(json.dumps(predict(model, classes, open_image(args.image), args.threshold, args.top_k),
                         ensure_ascii=False, indent=2))
    else:
        model, classes = load_model(args.model)
        manifest = json.loads(Path(args.manifest).read_text(encoding='utf-8'))
        if classes != manifest['classes']:
            raise ValueError('As classes do manifesto não correspondem ao modelo.')
        loader = DataLoader(AnimalDataset(manifest['splits']['test']), batch_size=16)
        print(json.dumps(evaluate(model, loader, classes), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
