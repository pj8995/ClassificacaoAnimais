"""Acrescenta fotos públicas somente ao treino, mantendo validação e teste originais."""
import argparse
from collections import Counter
import csv
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import warnings
import zipfile

import numpy as np
from PIL import Image, ImageOps
from scipy.fft import dctn
import pyarrow.parquet as pq

from baixar_dados_extras import CAT_REVISION, COMMONS_REVISION, DOG_REVISION

ROOT = Path(__file__).resolve().parent
DOG_ALIASES = {
    'black_and_tan_coonhound': 'black', 'curly_coated_retriever': 'curly',
    'flat_coated_retriever': 'flat', 'german_short_haired_pointer': 'german_short',
    'soft_coated_wheaten_terrier': 'soft', 'wire_haired_fox_terrier': 'wire',
    'shih_tzu': 'shih', 'brabancon_griffo': 'brabancon_griffon',
}


class HashIndex:
    """Árvore BK: consulta assinaturas próximas sem comparar cada par de imagens."""
    def __init__(self):
        self.root = None

    def add(self, value):
        if self.root is None:
            self.root = (value, {})
            return
        node = self.root
        while True:
            distance = (node[0] ^ value).bit_count()
            if not distance:
                return
            if distance not in node[1]:
                node[1][distance] = (value, {})
                return
            node = node[1][distance]

    def contains(self, value, radius=6):
        pending = [self.root] if self.root else []
        while pending:
            current, children = pending.pop()
            distance = (current ^ value).bit_count()
            if distance <= radius:
                return True
            pending.extend(child for edge, child in children.items()
                           if distance - radius <= edge <= distance + radius)
        return False


def image_info(content, minimum_size=64):
    # pHash detecta muitas cópias redimensionadas/recomprimidas. A assinatura é
    # aproximada: não prova ausência de todas as variantes de uma foto ou sessão.
    with warnings.catch_warnings():
        warnings.simplefilter('error', Image.DecompressionBombWarning)
        with Image.open(io.BytesIO(content)) as original:
            if original.width * original.height > 20_000_000 or min(original.size) < minimum_size:
                raise ValueError('Dimensões fora do intervalo aceito.')
            extension = {'JPEG': '.jpg', 'PNG': '.png', 'WEBP': '.webp'}.get(original.format)
            if not extension:
                raise ValueError('Formato não aceito.')
            image = ImageOps.exif_transpose(original).convert('L').resize((32, 32), Image.Resampling.LANCZOS)
            values = dctn(np.asarray(image, dtype=np.float32), norm='ortho')[:8, :8].flatten()
            # O termo constante representa brilho, não estrutura visual.
            bits = values[1:] > np.median(values[1:])
            visual_hash = sum(int(bit) << index for index, bit in enumerate(bits))
    return hashlib.sha256(content).hexdigest(), visual_hash, extension


def cat_candidates():
    sources = [('gatos_atharva.zip', 'atharva', CAT_REVISION),
               ('gatos_commons.zip', 'commons', COMMONS_REVISION)]
    for filename, source, revision in sources:
        with zipfile.ZipFile(ROOT / 'downloads' / 'extras' / filename) as archive:
            metadata = {}
            if source == 'commons':
                name = next(n for n in archive.namelist() if n.endswith('/images_metadata.csv'))
                rows = csv.DictReader(io.StringIO(archive.read(name).decode('utf-8-sig')))
                metadata = {r['filename']: r for r in rows}
            for name in sorted(archive.namelist()):
                parts = PurePosixPath(name).parts
                if len(parts) < 3 or parts[1] != 'images' or name.endswith('/'):
                    continue
                if source == 'atharva':
                    if len(parts) != 4:
                        continue
                    breed = parts[2].removesuffix('_cat')
                    attribution = {'author': 'Atharva Taras (curador)', 'license': 'CC BY 4.0',
                                   'url': 'https://github.com/AtharvaTaras/Cat-Breeds-Dataset'}
                else:
                    row = metadata.get(parts[-1])
                    if not row:
                        continue
                    breed = {'main_coon': 'maine_coon'}.get(row['breed'], row['breed'])
                    attribution = row
                yield breed, archive.read(name), {'source': source, 'revision': revision,
                                                   'original_path': name, 'attribution': attribution}


def dog_candidates():
    files = sorted((ROOT / 'downloads' / 'extras' / 'tsinghua' / 'data').glob('train-*.parquet'))
    if len(files) != 6:
        raise ValueError('O download deve conter os seis arquivos de treino Tsinghua Dogs.')
    for path in files:
        parquet = pq.ParquetFile(path)
        labels = json.loads(parquet.schema_arrow.metadata[b'huggingface'])['info']['features']['label']['names']
        index = 0
        for batch in parquet.iter_batches(batch_size=32):
            for row in batch.to_pylist():
                label = re.sub(r'^\d+-n\d+-', '', labels[row['label']]).lower()
                breed = DOG_ALIASES.get(label, label)
                yield breed, row['image']['bytes'], {
                    'source': 'tsinghua_dogs', 'revision': DOG_REVISION,
                    'original_path': f'{path.name}:{index}',
                    'attribution': {'authors': 'Ding-Nan Zou, Song-Hai Zhang, Tai-Jiang Mu, Ming Zhang',
                                    'url': 'https://cg.cs.tsinghua.edu.cn/ThuDogs/',
                                    'mirror': 'https://huggingface.co/datasets/giacomov/tsinghua_dogs',
                                    'license': 'CC BY 4.0, conforme metadados do espelho'}}
                index += 1


def prepare(animal, maximum=300):
    name = 'oxford_cats' if animal == 'gato' else 'stanford_dogs'
    manifest = json.loads((ROOT / 'datasets' / f'{name}.json').read_text(encoding='utf-8'))
    original_splits = {k: list(v) for k, v in manifest['splits'].items()}
    labels = {label['raca']: i for i, label in enumerate(manifest['classes'])}
    exact, perceptual = set(), HashIndex()
    print(f'{animal}: indexando todas as imagens originais, inclusive validação e teste.', flush=True)
    for split in manifest['splits'].values():
        for path, label in split:
            content = Path(path).read_bytes()
            digest, visual, extension = image_info(content, minimum_size=0)
            exact.add(digest)
            perceptual.add(visual)
    additions, rejected, provenance = Counter(), Counter(), []
    exclusion_path = ROOT / 'datasets' / 'ampliado' / 'exclusoes_visuais.json'
    visual_exclusions = {row['sha256'] for row in json.loads(exclusion_path.read_text(encoding='utf-8'))} if exclusion_path.exists() else set()
    candidates = cat_candidates() if animal == 'gato' else dog_candidates()
    total = 0
    for breed, content, source in candidates:
        total += 1
        if total % 1000 == 0:
            print(f'{animal}: {total} candidatos, {sum(additions.values())} fotos novas aceitas.', flush=True)
        if breed not in labels:
            rejected['raca_fora_das_classes_atuais'] += 1
            continue
        if hashlib.sha256(content).hexdigest() in visual_exclusions:
            rejected['exclusao_revisao_visual'] += 1
            continue
        if additions[breed] >= maximum:
            rejected['limite_por_raca'] += 1
            continue
        try:
            digest, visual, extension = image_info(content)
        except (OSError, ValueError, Image.DecompressionBombWarning, Image.DecompressionBombError):
            rejected['imagem_invalida_ou_dimensoes'] += 1
            continue
        if digest in exact:
            rejected['duplicata_exata'] += 1
            continue
        if perceptual.contains(visual):
            rejected['semelhanca_visual_phash_6'] += 1
            continue
        destination = ROOT / 'data' / 'extras' / animal / breed / f'{source["source"]}_{digest[:24]}{extension}'
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and destination.read_bytes() != content:
            raise ValueError('Arquivo existente com conteúdo diferente.')
        destination.write_bytes(content)
        manifest['splits']['train'].append([str(destination.resolve()), labels[breed]])
        exact.add(digest)
        perceptual.add(visual)
        additions[breed] += 1
        provenance.append({**source, 'path': str(destination.resolve()), 'breed': breed, 'sha256': digest})
    if not provenance:
        raise ValueError(f'Nenhuma imagem nova para {animal}.')
    assert manifest['splits']['val'] == original_splits['val']
    assert manifest['splits']['test'] == original_splits['test']
    manifest['source'] = {'original': manifest.get('source'), 'additional_training_sources': sorted({p['source'] for p in provenance})}
    manifest['expansion'] = {'counts': dict(additions), 'rejected': dict(rejected), 'cap_per_breed': maximum,
                             'validation_test_unchanged': True,
                             'note': 'Rótulos da origem; sem auditoria humana integral. pHash reduz duplicatas, sem garantir independência por indivíduo/sessão.'}
    directory = ROOT / 'datasets' / 'ampliado'
    directory.mkdir(parents=True, exist_ok=True)
    for filename, value in [(f'{name}.json', manifest), (f'proveniencia_{animal}.json', provenance)]:
        (directory / filename).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'{animal}: treino {len(original_splits["train"])} -> {len(manifest["splits"]["train"])}; '
          f'novas fotos {len(provenance)}, {len(additions)} raças ampliadas; rejeitadas {dict(rejected)}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--animal', choices=['gato', 'cachorro', 'todos'], default='todos')
    parser.add_argument('--max-per-breed', type=int, default=300)
    args = parser.parse_args()
    if args.max_per_breed < 1:
        parser.error('O limite deve ser positivo.')
    for animal in (['gato', 'cachorro'] if args.animal == 'todos' else [args.animal]):
        prepare(animal, args.max_per_breed)
