"""Classificação visual filhote/adulta a partir dos rótulos fornecidos pelo usuário."""
# A fase é aprendida pelos exemplos rotulados nas pastas, não por uma estimativa de idade. Uma resposta incerta pode deixar a fase sem confirmação.
from pathlib import Path
import random


# Considera filhote a imagem dentro de uma pasta chamada Paula filhote, sem distinguir maiúsculas. As demais fotos da Paula são rotuladas como adultas.
def phase_label(path):
    return 'filhote' if any(part.casefold() == 'paula filhote' for part in Path(path).parts[:-1]) else 'adulta'


# Exige que as duas vistas concordem na fase e que ambas superem o limiar; caso contrário, não confirma uma fase.
def phase_decision(probabilities, threshold=.8):
    winners = [max(range(len(row)), key=lambda i: row[i]) for row in probabilities]
    if len(winners) != 2 or winners[0] != winners[1]:
        return None
    index = winners[0]
    return index if min(row[index] for row in probabilities) >= threshold else None


# Separa grupos de fotos por fase e treina apenas uma camada linear. A validação escolhe os pesos; o teste mede o resultado depois dessa escolha.
def train_phases(features, paths, groups):
    import torch
    from torch import nn
    from evaluation_metrics import summarize_confusion
    names = ['adulta','filhote']
    labels = [names.index(phase_label(p)) for p in paths]
    counts = {name: labels.count(i) for i,name in enumerate(names)}
    unavailable = {'disponivel':False,'contagem':counts,
        'motivo':'Adicione pelo menos 8 fotos únicas e variadas de cada fase, em pelo menos 3 grupos visuais, e execute treinar_paula.py.'}
    if min(counts.values()) < 8:
        return None, unavailable
    # Separamos grupos inteiros para evitar que fotos visualmente muito próximas fiquem espalhadas entre treino e teste.
    splits = {'train':[],'val':[],'test':[]}
    rng = random.Random(73)
    for label in range(2):
        selected = [g for g in groups if all(labels[i] == label for i in g)]
        if len(selected) < 3:
            return None, unavailable
        rng.shuffle(selected)
        allocated = {'train':[],'val':[],'test':[]}
        for group in selected:
            split = 'test' if len(allocated['test']) < counts[names[label]]*.2 else 'val' if len(allocated['val']) < counts[names[label]]*.2 else 'train'
            allocated[split].extend(group)
        if min(map(len,allocated.values())) < 2:
            return None, unavailable
        for name in splits:
            splits[name].extend(allocated[name])
    y = torch.tensor(labels)
    x = features[splits['train']].flatten(0,1)
    target = y[splits['train']].repeat_interleave(2)
    val_x = features[splits['val']].flatten(0,1)
    val_y = y[splits['val']].repeat_interleave(2)
    # A saída tem duas classes: adulta e filhote. Os pesos da perda compensam a diferença de quantidade entre as fases.
    head = nn.Linear(x.shape[-1], 2)
    weights = torch.tensor([1/max(1,(target==i).sum().item()) for i in range(2)])
    optimizer = torch.optim.AdamW(head.parameters(), lr=.02, weight_decay=.05)
    best_loss, stale, best = float('inf'), 0, None
    # Cada passo ajusta a camada com o treino; a perda de validação guarda o melhor estado e interrompe após 40 passos sem melhora.
    for epoch in range(500):
        optimizer.zero_grad()
        loss = nn.functional.cross_entropy(head(x),target,weight=weights)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            val_loss = nn.functional.cross_entropy(head(val_x),val_y).item()
        if val_loss < best_loss:
            best_loss, stale = val_loss, 0
            best = {k:v.detach().clone() for k,v in head.state_dict().items()}
        else:
            stale += 1
        if stale >= 40:
            break
    head.load_state_dict(best)
    # A matriz mede a classificação de todas as fotos de teste. Também contamos separadamente as respostas que passaram pelo filtro de confirmação.
    matrix = [[0,0],[0,0]]
    accepted, correct_accepted = 0, 0
    with torch.no_grad():
        scores = head(features[splits['test']]).softmax(-1)
    for actual, views in zip(y[splits['test']].tolist(), scores.tolist()):
        predicted = max(range(2),key=lambda i: sum(row[i] for row in views))
        matrix[actual][predicted] += 1
        decision = phase_decision(views)
        if decision is not None:
            accepted += 1
            correct_accepted += decision == actual
    report = summarize_confusion(matrix,[{'animal':'gato','raca':name} for name in names])
    report.update(disponivel=True,contagem=counts,splits={k:[paths[i] for i in v] for k,v in splits.items()},
                  confirmadas=accepted,acertos_confirmadas=correct_accepted,
                  observacao='Avaliação visual pequena; não estima idade cronológica. Grupos perceptuais não garantem independência por sessão.')
    saved = {'weight':best['weight'],'bias':best['bias'],'classes':names,'threshold':.8}
    return saved, report
