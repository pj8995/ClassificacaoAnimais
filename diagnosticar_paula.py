"""Diagnóstico local do filtro de espécie e do reconhecimento individual."""
# Diagnóstico do filtro geral original com as fotos da Paula na pasta principal. Não substitui uma avaliação independente nem percorre as fotos de filhote.
import json
from pathlib import Path
import torch
from animal_classifier import open_image
from general_classifier import AnimalClassifier, summarize
from own_cat_classifier import OwnCatClassifier


# Compara a identificação geral com o reconhecimento direto da Paula e grava os resultados para investigar falhas.
def main():
    torch.set_num_threads(4)
    classifier = AnimalClassifier()
    cat = OwnCatClassifier()
    results = []
    for path in sorted(Path('data/gato/paula').glob('*')):
        if not path.is_file():
            continue
        image = open_image(path)
        with torch.inference_mode():
            probabilities = classifier.general(classifier.transform(image).unsqueeze(0).to(classifier.device)).softmax(-1)[0].cpu().tolist()
        gate = summarize(probabilities, classifier.categories, .6)
        identity = cat.identity.predict(image, cat.model, cat.transform, cat.device)
        results.append({'arquivo': path.name, 'gato_score': gate['scores_por_tipo'].get('gato',0),
                        'tipo': gate['previsao']['animal'], 'recusada': gate['fora_do_escopo'],
                        'paula':identity['identificada'], 'score_identidade':identity['score']})
    Path('outputs/paula/diagnostico_filtro.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'fotos':len(results),'recusadas_no_filtro':sum(r['recusada'] for r in results),
        'tipo_incorreto':sum(r['tipo']!='gato' for r in results),'paula_sem_filtro':sum(r['paula'] for r in results),
        'falhas':[r for r in results if r['tipo']!='gato' or not r['paula']]},ensure_ascii=False,indent=2),flush=True)


if __name__ == '__main__':
    main()
