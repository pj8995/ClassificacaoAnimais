"""Baixa versões fixas de bases públicas; não envia fotos locais para a internet."""
from pathlib import Path
import requests
from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parent
CAT_REVISION = '56a69059119626ed210023bbfc6fdcb8262e8703'
COMMONS_REVISION = 'd86c717b9940f17664d50d07e72f1d8457f12938'
DOG_REVISION = '2a466a14f36e7f911de32713734f52c1c35f9765'


def download_zip(repo, revision, name):
    destination = ROOT / 'downloads' / 'extras' / name
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return
    temporary = destination.with_suffix('.part')
    url = f'https://codeload.github.com/{repo}/zip/{revision}'
    print(f'Baixando {repo} na revisão {revision}', flush=True)
    with requests.get(url, stream=True, timeout=(30, 180)) as response:
        response.raise_for_status()
        received, last = 0, 0
        with temporary.open('wb') as stream:
            for chunk in response.iter_content(1024 * 1024):
                stream.write(chunk)
                received += len(chunk)
                if received - last >= 50 * 1024**2:
                    print(f'{name}: {received / 1024**2:.0f} MB', flush=True)
                    last = received
    import zipfile
    with zipfile.ZipFile(temporary) as archive:
        if archive.testzip() is not None:
            raise ValueError('Arquivo ZIP corrompido.')
    temporary.replace(destination)


if __name__ == '__main__':
    download_zip('AtharvaTaras/Cat-Breeds-Dataset', CAT_REVISION, 'gatos_atharva.zip')
    download_zip('CatInBeard/cat-breeds-dataset', COMMONS_REVISION, 'gatos_commons.zip')
    print('Baixando Tsinghua Dogs (somente a divisão train, sem executar código remoto).', flush=True)
    snapshot_download(repo_id='giacomov/tsinghua_dogs', repo_type='dataset', revision=DOG_REVISION,
                      local_dir=str(ROOT / 'downloads' / 'extras' / 'tsinghua'),
                      allow_patterns=['README.md', 'data/train-*.parquet'], max_workers=2)
    print('Downloads concluídos.', flush=True)
