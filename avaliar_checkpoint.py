"""Avalia o melhor checkpoint existente sem executar épocas de treinamento."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from animal_classifier import AnimalDataset, load_model
from treinar_do_zero import evaluate, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='outputs/do_zero_ampliado')
    parser.add_argument('--animal', choices=['gato', 'cachorro'], default='cachorro')
    args = parser.parse_args()
    torch.set_num_threads(4)
    folder = Path(args.output) / args.animal
    checkpoint = folder / 'best.pt'
    model, classes = load_model(checkpoint)
    manifest = json.loads((folder / 'manifest.json').read_text(encoding='utf-8'))
    if classes != manifest['classes']:
        raise ValueError('Classes do checkpoint incompatíveis com o manifesto.')
    print(f'Avaliando {args.animal}: melhor checkpoint, sem novo treinamento.', flush=True)
    loader = DataLoader(AnimalDataset(manifest['splits']['test']), batch_size=16, num_workers=0)
    report = evaluate(model, loader, classes)
    report.update(checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                  training='continued_own_scratch_all_layers', pretrained=False,
                  source=manifest.get('source'),
                  note='Avaliação do melhor checkpoint após interrupção voluntária. Teste já consultado; não é avaliação independente final.')
    write_json(folder / 'test_metrics.json', report)
    print(json.dumps({k: report[k] for k in ['accuracy', 'macro_f1', 'samples']}), flush=True)


if __name__ == '__main__':
    main()
