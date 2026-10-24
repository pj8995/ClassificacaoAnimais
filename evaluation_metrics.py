"""Métricas de avaliação multiclasse; não são scores de uma imagem individual."""
# As métricas vêm de fotos de teste com rótulo conhecido; elas não são a confiança da foto enviada. A média macro dá o mesmo peso a cada classe.
import hashlib
import json
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPORT_PATH = ROOT / 'outputs' / 'do_zero_ampliado' / 'cachorro' / 'test_metrics.json'
CHECKPOINT_PATH = REPORT_PATH.with_name('best.pt')
CAT_REPORT_PATH = ROOT / 'outputs' / 'do_zero_ampliado' / 'gato' / 'test_metrics.json'
CAT_CHECKPOINT_PATH = CAT_REPORT_PATH.with_name('best.pt')


# Usa a matriz de confusão para calcular acurácia, precisão, recall, F1 e F2. Divisões sem amostras recebem zero; F2 dá mais peso ao recall.
def summarize_confusion(matrix, classes):
    n = len(classes)
    if not n or len(matrix) != n or any(len(row) != n for row in matrix):
        raise ValueError('Matriz de confusão incompatível com as classes.')
    if any(not isinstance(x, int) or x < 0 for row in matrix for x in row):
        raise ValueError('A matriz deve conter contagens inteiras não negativas.')
    total = sum(sum(row) for row in matrix)
    if not total:
        raise ValueError('Não há exemplos avaliados.')
    per_class = []
    for i, label in enumerate(classes):
        tp = matrix[i][i]
        support = sum(matrix[i])
        predicted = sum(row[i] for row in matrix)
        precision = tp / predicted if predicted else 0.
        recall = tp / support if support else 0.
        per_class.append({**label, 'precision': precision, 'recall': recall,
                          'f1': 2*precision*recall/(precision+recall) if precision+recall else 0.,
                          'f2': 5*precision*recall/(4*precision+recall) if 4*precision+recall else 0.,
                          'support': support})
    return {'accuracy': sum(matrix[i][i] for i in range(n))/total,
            **{f'macro_{key}': sum(c[key] for c in per_class)/n for key in ['precision', 'recall', 'f1', 'f2']},
            'classes': per_class, 'samples': total}


# Lê o relatório e verifica se seu hash corresponde ao checkpoint atual. O cache evita reler arquivos que não mudaram.
@lru_cache(maxsize=4)
def _load(report_path, report_stamp, checkpoint_path, checkpoint_stamp):
    report = json.loads(Path(report_path).read_text(encoding='utf-8'))
    if report.get('checkpoint_sha256') != hashlib.sha256(Path(checkpoint_path).read_bytes()).hexdigest():
        raise ValueError('Relatório não corresponde ao checkpoint atual.')
    return summarize_confusion(report['confusion_matrix'],
                               [{'animal': c['animal'], 'raca': c['raca']} for c in report['classes']])


# Seleciona o relatório da espécie e organiza os campos da API. Relatórios ausentes ou incompatíveis são apresentados como indisponíveis.
def response_metrics(animal, breed, report_path=REPORT_PATH, checkpoint_path=CHECKPOINT_PATH):
    if animal == 'gato' and report_path == REPORT_PATH and checkpoint_path == CHECKPOINT_PATH:
        report_path, checkpoint_path = CAT_REPORT_PATH, CAT_CHECKPOINT_PATH
    scope = 'Classificador felino próprio no teste Oxford Pets' if animal == 'gato' else 'Classificador canino próprio no teste Stanford Dogs'
    context = {'disponivel': False, 'escopo': scope,
               'media': 'macro',
               'observacao': 'Métricas do conjunto de teste, antes dos filtros da interface; não representam a confiança desta foto. Classificador de raça originado do zero; teste já consultado em experimentos anteriores, sem independência final.'}
    if animal not in {'cachorro', 'gato'}:
        return {**context, 'motivo': 'Não há avaliação local deste modelo para o tipo previsto.'}
    try:
        report_path, checkpoint_path = Path(report_path), Path(checkpoint_path)
        values = _load(str(report_path), report_path.stat().st_mtime_ns,
                       str(checkpoint_path), checkpoint_path.stat().st_mtime_ns)
        if any(c['animal'] != animal for c in values['classes']):
            raise ValueError('Relatório de outra espécie.')
    except (OSError, ValueError, KeyError, TypeError):
        return {**context, 'motivo': 'Relatório de teste ausente, inválido ou de outro checkpoint.'}
    selected = next((c for c in values['classes'] if c['raca'] == breed), None)
    return {**context, 'disponivel': True, 'amostras': values['samples'], 'classes': len(values['classes']),
            'precisao': values['macro_precision'], 'acuracia': values['accuracy'],
            'recall': values['macro_recall'], 'f1': values['macro_f1'], 'f2': values['macro_f2'],
            'raca_prevista': selected}
