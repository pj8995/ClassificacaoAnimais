"""Treina identidade com as fotos locais e gatos negativos da base Oxford."""
# Treina a identidade com fotos da Paula e exemplos de outros gatos. Fotos repetidas ou parecidas são agrupadas para reduzir vazamento entre as divisões.
import hashlib
import json
import random
from pathlib import Path

import torch
from PIL import Image, ImageOps
from torch import nn
from animal_classifier import open_image, transform, device, load_model
# Experimento histórico, sem integração com a aplicação atual.
CAT_CHECKPOINT = Path(__file__).resolve().parent/'outputs'/'treino_gatos'/'best.pt'
from paula_identity import IDENTITY_PATH, identity_decision
from paula_phases import phase_label, train_phases

ROOT = Path(__file__).resolve().parent


# Produz uma assinatura visual comparando pixels vizinhos em uma miniatura em tons de cinza; assinaturas próximas sugerem fotos semelhantes.
def dhash(image):
    pixels = list(image.convert('L').resize((9, 8)).tobytes())
    return sum((pixels[y*9+x] > pixels[y*9+x+1]) << (y*8+x) for y in range(8) for x in range(8))


# Busca fotos também nas subpastas, remove duplicatas exatas e agrupa semelhanças visuais antes de separar treino, validação e teste.
def prepare_positives():
    files = sorted(p for p in (ROOT/'data'/'gato'/'paula').rglob('*') if p.is_file() and p.suffix.lower() in {'.jpg','.jpeg','.png','.webp'})
    unique, seen, duplicate, hashes = [], {}, [], []
    for path in files:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest in seen:
            if seen[digest] != phase_label(path):
                raise ValueError(f'A mesma foto foi rotulada como filhote e adulta: {path}')
            duplicate.append(str(path))
            continue
        seen[digest] = phase_label(path)
        hashes.append(dhash(open_image(path)))
        unique.append(str(path.resolve()))
    groups = []
    remaining = set(range(len(unique)))
    while remaining:
        group = {remaining.pop()}
        changed = True
        while changed:
            close = {j for j in remaining if any((hashes[j] ^ hashes[i]).bit_count() <= 6 for i in group)}
            changed = bool(close)
            group |= close
            remaining -= close
        groups.append(sorted(group))
    random.Random(42).shuffle(groups)
    if len(groups) < 3:
        raise ValueError('São necessários pelo menos três grupos de fotos visualmente distintas.')
    split = {'train': [], 'val': [], 'test': []}
    for group in groups:
        name = 'test' if len(split['test']) < len(unique)*.2 else 'val' if len(split['val']) < len(unique)*.2 else 'train'
        split[name].extend(group)
    if min(map(len, split.values())) < 2:
        raise ValueError('Divisão insuficiente; adicione fotos de momentos e fundos diferentes.')
    return unique, split, duplicate, groups


# Extrai características, treina a camada Paula/outros gatos, ajusta o limiar pela validação e salva pesos e relatórios, incluindo a fase quando há dados suficientes.
def main():
    torch.manual_seed(42)
    torch.set_num_threads(4)
    paths, positive_splits, duplicates, groups = prepare_positives()
    model, _ = load_model(CAT_CHECKPOINT)
    preprocess = transform()

    # Transforma cada foto e sua versão espelhada em vetores normalizados, mantendo a rede felina congelada.
    def encode(paths, mirror=True):
        embeddings = []
        with torch.inference_mode():
            for offset in range(0, len(paths), 8):
                photos = [open_image(p) for p in paths[offset:offset+8]]
                images = [v for image in photos for v in ([image, ImageOps.mirror(image)] if mirror else [image])]
                batch = torch.stack([preprocess(v) for v in images]).to(device())
                embeddings.append(model.avgpool(model.features(batch)).flatten(1).cpu())
        return torch.nn.functional.normalize(torch.cat(embeddings), dim=-1)

    print(f'Paula: {len(paths)} imagens únicas; divisão { {k:len(v) for k,v in positive_splits.items()} }', flush=True)
    positive = encode(paths).reshape(len(paths), 2, -1)
    manifest = json.loads((ROOT/'datasets'/'oxford_cats.json').read_text(encoding='utf-8'))
    negative = {}
    negative_paths = {}
    rng = random.Random(42)
    for split, count in [('train', 20), ('val', 5), ('test', 10)]:
        selected = []
        for label in range(len(manifest['classes'])):
            available = [p for p,y in manifest['splits'][split] if y == label]
            rng.shuffle(available)
            selected.extend(available[:count])
        negative_paths[split] = selected
        negative[split] = encode(selected).reshape(len(selected), 2, -1)
        print(f'Outros gatos {split}: {len(selected)}', flush=True)
    train_positive = positive[positive_splits['train']].flatten(0,1)
    train_negative = negative['train'].flatten(0,1)
    x = torch.cat([train_positive, train_negative])
    y = torch.cat([torch.ones(len(train_positive)), torch.zeros(len(train_negative))])
    head = nn.Linear(x.shape[1], 1)
    optimizer = torch.optim.AdamW(head.parameters(), lr=.03, weight_decay=.05)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(len(train_negative)/len(train_positive)))
    best_loss, stale, best = float('inf'), 0, None
    val_x = torch.cat([positive[positive_splits['val']], negative['val']]).flatten(0,1)
    val_y = torch.cat([torch.ones(len(positive_splits['val'])*2), torch.zeros(len(negative['val'])*2)])
    history = []
    for epoch in range(500):
        optimizer.zero_grad()
        loss = criterion(head(x).flatten(), y)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            val_loss = nn.functional.binary_cross_entropy_with_logits(head(val_x).flatten(), val_y).item()
        history.append({'epoch':epoch+1, 'train_loss':loss.item(), 'val_loss':val_loss})
        if val_loss < best_loss:
            best_loss, stale = val_loss, 0
            best = {k:v.detach().clone() for k,v in head.state_dict().items()}
        else:
            stale += 1
        if stale >= 40:
            break
    head.load_state_dict(best)
    with torch.no_grad():
        val_negative_scores = head(negative['val']).sigmoid().squeeze(-1)
    # Requer ambas as versões acima do limiar; este é escolhido somente na validação.
    threshold = min(.999, max(.8, val_negative_scores.min(-1).values.max().item()+.02))
    report = {}
    for split in ['val', 'test']:
        with torch.no_grad():
            p = head(positive[positive_splits[split]]).sigmoid().squeeze(-1).tolist()
            n = head(negative[split]).sigmoid().squeeze(-1).tolist()
        tp = sum(identity_decision(scores, threshold) for scores in p)
        fp = sum(identity_decision(scores, threshold) for scores in n)
        fn, tn = len(p)-tp, len(n)-fp
        precision = tp/(tp+fp) if tp+fp else 0.
        recall = tp/len(p)
        report[split] = {'tp':tp,'fp':fp,'fn':fn,'tn':tn,'precision':precision,'recall':recall,
                          'accuracy':(tp+tn)/(len(p)+len(n)), 'positives':len(p),'negatives':len(n),
                          'f1':2*precision*recall/(precision+recall) if precision+recall else 0.}
    output = IDENTITY_PATH.parent
    output.mkdir(parents=True, exist_ok=True)
    saved = {'weight':best['weight'].flatten(), 'bias':best['bias'].flatten(), 'threshold':threshold,
             'feature_checkpoint_sha256':hashlib.sha256(CAT_CHECKPOINT.read_bytes()).hexdigest()}
    saved['phase'], phase_report = train_phases(positive, paths, groups)
    (output/'fases_metrics.json').write_text(json.dumps(phase_report,ensure_ascii=False,indent=2),encoding='utf-8')
    torch.save(saved, IDENTITY_PATH)
    (output/'manifest.json').write_text(json.dumps({'photos':paths, 'splits':positive_splits,
        'near_duplicate_groups':groups,'excluded_exact_duplicates':duplicates,'negative_paths':negative_paths},ensure_ascii=False,indent=2),encoding='utf-8')
    (output/'history.json').write_text(json.dumps(history,indent=2),encoding='utf-8')
    report.update(threshold=threshold, unique_photos=len(paths), excluded_duplicates=len(duplicates),
                  fases=phase_report,
                  note='Teste pequeno e exploratório; agrupamento perceptual não garante separação por sessão. Novas fotos e gatos brancos parecidos são necessários para avaliar generalização.')
    (output/'test_metrics.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)


if __name__ == '__main__':
    main()
