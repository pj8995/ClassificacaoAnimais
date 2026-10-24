"""Baixa Oxford Pets e organiza somente as 12 raças de gatos."""
# Prepara somente as 12 raças felinas do Oxford-IIIT Pet, preservando os dados caninos. A versão fixada da origem permite reproduzir a preparação.
import hashlib
import io
import json
import re
from pathlib import Path

from PIL import Image
from huggingface_hub import snapshot_download
import pyarrow.parquet as pq
from preparar_dataset import finalize, breed_name
from pretrained_classifier import CAT_BREEDS

ROOT = Path(__file__).resolve().parent
REPO = 'timm/oxford-iiit-pet'
REVISION = '089695c834a7deb60505b7cc506672db1c31a6aa'


# Baixa os dados, filtra as raças de gatos, valida as fotos e usa a rotina comum para criar o manifesto felino.
def prepare(root=ROOT):
    directory = root / 'downloads' / 'oxford_pets'
    snapshot_download(repo_id=REPO, repo_type='dataset', revision=REVISION,
                      local_dir=str(directory), allow_patterns=['data/*.parquet', 'README.md'])
    records, official = {}, {'train': [], 'test': []}
    for split in ['train', 'test']:
        parquet = pq.ParquetFile(directory/'data'/f'{split}-00000-of-00001.parquet')
        metadata = json.loads(parquet.schema_arrow.metadata[b'huggingface'])
        labels = metadata['info']['features']['label']['names']
        for batch in parquet.iter_batches(batch_size=32):
            for row in batch.to_pylist():
                breed = breed_name(labels[row['label']])
                if breed not in CAT_BREEDS:
                    continue
                image_id = str(row['image_id'])
                if not re.fullmatch(r'[A-Za-z0-9_]+', image_id):
                    raise ValueError('Identificador de imagem inválido.')
                content = row['image']['bytes']
                with Image.open(io.BytesIO(content)) as image:
                    image.verify()
                filename = f'{split}_{image_id}.jpg'
                relative = f'{breed}/{filename}'
                destination = root/'data'/'gato'/breed/filename
                destination.parent.mkdir(parents=True, exist_ok=True)
                if destination.exists() and destination.read_bytes() != content:
                    raise ValueError(f'Imagem existente diferente: {destination}')
                if not destination.exists():
                    destination.write_bytes(content)
                records[relative] = {'path': str(destination.resolve()), 'breed': breed,
                                     'sha256': hashlib.sha256(content).hexdigest()}
                official[split].append(relative)
        print(f'Gatos organizados em {split}: {len(official[split])}', flush=True)
    if {r['breed'] for r in records.values()} != CAT_BREEDS:
        raise ValueError('Classes felinas ausentes ou inesperadas.')
    finalize(root, records, official, f'https://huggingface.co/datasets/{REPO}/tree/{REVISION}',
             animal='gato', expected_classes=12, manifest_name='oxford_cats', report_suffix='_gatos')


if __name__ == '__main__':
    prepare()
