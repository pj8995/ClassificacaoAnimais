"""Regressão local de todas as fotos da Paula e controles sem identificação positiva."""
# Verifica o fluxo completo com as fotos adultas locais e controles negativos. As fotos da própria Paula são uma regressão de funcionamento, não um teste independente.
import json
import argparse
from pathlib import Path
import torch
from PIL import Image
from animal_classifier import open_image
from general_classifier import AnimalClassifier


# Executa o modo automático ou o tipo gato informado, registra as falhas e compara os resultados com os controles.
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--modo',choices=['auto','gato'],default='auto')
    args = parser.parse_args()
    torch.set_num_threads(4)
    model = AnimalClassifier()
    results = []
    for path in sorted(Path('data/gato/paula').glob('*')):
        if not path.is_file():
            continue
        result = model.predict(open_image(path), animal_hint=args.modo)
        results.append({'arquivo':path.name,'animal':result['previsao']['animal'],
                        'paula':result.get('identidade',{}).get('identificada',False),
                        'origem_tipo':result.get('origem_tipo'), 'deteccao':result.get('deteccao')})
        if len(results) % 5 == 0:
            print(f"Fotos verificadas: {len(results)}; identificadas como Paula: {sum(r['paula'] for r in results)}",flush=True)
    manifest = json.loads(Path('outputs/paula/manifest.json').read_text(encoding='utf-8'))
    negatives = []
    for path in manifest['negative_paths']['test'][:24]:
        result = model.predict(open_image(path), animal_hint=args.modo)
        negatives.append(result.get('identidade',{}).get('identificada',False))
    objects = [model.predict(Image.new('RGB',(320,480),color)) for color in ['white','gray','black']] if args.modo=='auto' else []
    report = {'modo':args.modo,'fotos':len(results),'gatos':sum(r['animal']=='gato' for r in results),
              'paula_identificada':sum(r['paula'] for r in results),
              'outros_gatos_testados':len(negatives),'falsos_paula':sum(negatives),
              'imagens_vazias_recusadas':sum(r['fora_do_escopo'] for r in objects),
              'falhas':[r for r in results if r['animal']!='gato' or not r['paula']], 'resultados':results}
    suffix = '_informado' if args.modo=='gato' else ''
    Path(f'outputs/paula/regressao_fluxo{suffix}.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='resultados'},ensure_ascii=False,indent=2),flush=True)


if __name__ == '__main__':
    main()
