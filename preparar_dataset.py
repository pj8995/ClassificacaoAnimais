"""Baixa Stanford Dogs e cria data/cachorro/<raca> e manifesto de treino."""
# Organiza as imagens caninas em pastas por raça e cria um manifesto com os caminhos e as divisões do conjunto. O treino consome esse manifesto.
import argparse
import csv
import hashlib
import io
import json
import random
import re
import tarfile
from pathlib import Path, PurePosixPath

import requests
from PIL import Image

ROOT = Path(__file__).resolve().parent
SOURCE = 'http://vision.stanford.edu/aditya86/ImageNetDogs/'


# Baixa o arquivo em blocos para um arquivo temporário e reaproveita o download final quando ele já existe.
def download(name, directory):
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / name
    if target.is_file():
        print(f'Reutilizando {target}', flush=True)
        return target
    partial = target.with_suffix(target.suffix + '.part')
    with requests.get(SOURCE + name, stream=True, timeout=(30, 120)) as response:
        response.raise_for_status()
        total = int(response.headers.get('content-length', 0))
        received, next_report = 0, 0
        with partial.open('wb') as stream:
            for chunk in response.iter_content(1024 * 1024):
                stream.write(chunk)
                received += len(chunk)
                if received >= next_report:
                    print(f'{name}: {received/1024**2:.0f} MB / {total/1024**2:.0f} MB', flush=True)
                    next_report = received + 50 * 1024**2
    partial.replace(target)
    return target


# Remove o identificador da origem e padroniza o nome da raça para uso seguro como pasta.
def breed_name(folder):
    name = re.sub(r'[-\s]+', '_', re.sub(r'^n\d{8}-', '', folder).lower())
    if not re.fullmatch(r'[a-z0-9_]+', name):
        raise ValueError(f'Nome de raça inválido: {folder}')
    return name


# Lê as listas oficiais e copia as imagens válidas do arquivo de origem sem extrair caminhos arbitrários.
def prepare(root=ROOT):
    from scipy.io import loadmat
    raw = root / 'downloads' / 'stanford_dogs'
    images = download('images.tar', raw)
    lists = download('lists.tar', raw)
    official = {}
    with tarfile.open(lists) as archive:
        for member in archive:
            if member.isfile() and PurePosixPath(member.name).name in {'train_list.mat', 'test_list.mat'}:
                data = loadmat(io.BytesIO(archive.extractfile(member).read()))
                split = 'train' if PurePosixPath(member.name).name == 'train_list.mat' else 'test'
                official[split] = [str(row[0][0]) for row in data['file_list']]
    if set(official) != {'train', 'test'}:
        raise ValueError('Listas oficiais incompletas.')
    wanted = set(official['train']) | set(official['test'])
    if len(wanted) != 20580 or set(official['train']) & set(official['test']):
        raise ValueError('Divisão oficial inesperada.')
    records = {}
    with tarfile.open(images) as archive:
        for member in archive:
            parts = PurePosixPath(member.name).parts
            if not member.isfile() or len(parts) != 3 or parts[0] != 'Images':
                continue
            relative = '/'.join(parts[1:])
            if relative not in wanted:
                continue
            if any(p in {'.', '..'} for p in parts):
                raise ValueError('Caminho inválido no arquivo de imagens.')
            content = archive.extractfile(member).read()
            with Image.open(io.BytesIO(content)) as image:
                image.verify()
            breed = breed_name(parts[1])
            destination = root / 'data' / 'cachorro' / breed / parts[2]
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not destination.exists():
                destination.write_bytes(content)
            elif hashlib.sha256(destination.read_bytes()).digest() != hashlib.sha256(content).digest():
                raise ValueError(f'Arquivo existente diferente: {destination}')
            records[relative] = {'path': str(destination.resolve()), 'breed': breed,
                                 'sha256': hashlib.sha256(content).hexdigest()}
            if len(records) % 2000 == 0:
                print(f'Organizadas e verificadas: {len(records)} fotos', flush=True)
    if set(records) != wanted:
        raise ValueError('Há imagens ausentes no pacote.')
    finalize(root, records, official, SOURCE)


# Agrupa imagens por hash para retirar duplicatas ou conflitos do manifesto e separa a validação, preservando o teste oficial.
def finalize(root, records, official, source, animal='cachorro', expected_classes=120,
             manifest_name='stanford_dogs', report_suffix=''):
    breeds = sorted({r['breed'] for r in records.values()})
    if len(breeds) != expected_classes:
        raise ValueError(f'Esperadas {expected_classes} classes.')
    classes = [{'animal': animal, 'raca': b} for b in breeds]
    labels = {b: i for i, b in enumerate(breeds)}
    groups = {}
    for relative, record in records.items():
        groups.setdefault(record['sha256'], []).append(relative)
    conflicts = {digest for digest, names in groups.items()
                 if len({records[name]['breed'] for name in names}) > 1}
    splits, seen, excluded = {'train': [], 'val': [], 'test': []}, set(), []
    # O teste tem prioridade: uma cópia idêntica nunca passa para treino/validação.
    selected = {'train': [], 'test': []}
    for split in ['test', 'train']:
        for relative in official[split]:
            record = records[relative]
            if record['sha256'] in conflicts or record['sha256'] in seen:
                excluded.append(relative)
                continue
            seen.add(record['sha256'])
            selected[split].append(record)
    rng = random.Random(42)
    for breed in breeds:
        training = sorted((r for r in selected['train'] if r['breed'] == breed), key=lambda r: r['path'])
        rng.shuffle(training)
        nval = max(1, round(len(training) * .2))
        for split, items in [('val', training[:nval]), ('train', training[nval:]),
                             ('test', [r for r in selected['test'] if r['breed'] == breed])]:
            splits[split].extend((r['path'], labels[breed]) for r in items)
    output = root / 'datasets'
    output.mkdir(exist_ok=True)
    manifest = {'classes': classes, 'splits': splits, 'seed': 42, 'source': source,
                'excluded_duplicates_or_conflicts': excluded,
                'protocol': 'Teste da fonte preservado; 20% do treino publicado para validação; deduplicação SHA256.'}
    (output / f'{manifest_name}.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    with (output / f'racas{report_suffix}.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['raca', 'fotos_na_pasta', 'treino', 'validacao', 'teste'])
        for label, breed in enumerate(breeds):
            writer.writerow([breed, sum(r['breed'] == breed for r in records.values()),
                             *[sum(y == label for _, y in splits[s]) for s in ['train', 'val', 'test']]])
    summary = {'source': source, 'breeds': len(breeds), 'images': len(records),
               'splits': {s: len(v) for s, v in splits.items()}, 'excluded': len(excluded)}
    (output / f'resumo{report_suffix}.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2), flush=True)


# Lê a versão fixada do dataset em Parquet e grava as imagens em pastas locais antes de gerar o manifesto.
def prepare_huggingface(root=ROOT):
    from huggingface_hub import snapshot_download
    import pyarrow.parquet as pq
    repository = 'Donghyun99/Stanford-Dogs'
    revision = '4246493e3d3e3a15a43c291ec5e394a003a2e10a'
    directory = root / 'downloads' / 'stanford_dogs_hf'
    snapshot_download(repo_id=repository, repo_type='dataset', revision=revision,
                      local_dir=str(directory), allow_patterns=['data/*.parquet', 'README.md'])
    records, official = {}, {'train': [], 'test': []}
    for split in ['train', 'test']:
        parquet = pq.ParquetFile(directory / 'data' / f'{split}-00000-of-00001.parquet')
        metadata = json.loads(parquet.schema_arrow.metadata[b'huggingface'])
        labels = metadata['info']['features']['label']['names']
        for batch in parquet.iter_batches(batch_size=64):
            for row in batch.to_pylist():
                breed = breed_name(labels[row['label']])
                content = row['image']['bytes']
                with Image.open(io.BytesIO(content)) as image:
                    image.verify()
                filename = f'{split}_{len(official[split]):05d}.jpg'
                relative = f'{breed}/{filename}'
                destination = root / 'data' / 'cachorro' / breed / filename
                destination.parent.mkdir(parents=True, exist_ok=True)
                if destination.exists() and destination.read_bytes() != content:
                    raise ValueError(f'Arquivo local conflitante: {destination}')
                if not destination.exists():
                    destination.write_bytes(content)
                records[relative] = {'path': str(destination.resolve()), 'breed': breed,
                                     'sha256': hashlib.sha256(content).hexdigest()}
                official[split].append(relative)
            if len(official[split]) % 1024 == 0:
                print(f'Fotos organizadas em {split}: {len(official[split])}', flush=True)
    if len(official['train']) != 12000 or len(official['test']) != 8580:
        raise ValueError('Tamanho inesperado da base espelhada.')
    finalize(root, records, official, f'https://huggingface.co/datasets/{repository}/tree/{revision}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', choices=['huggingface', 'stanford'], default='huggingface')
    args = parser.parse_args()
    (prepare_huggingface if args.source == 'huggingface' else prepare)()
