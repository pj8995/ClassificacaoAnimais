"""Critérios de abstenção: heurísticas conservadoras, não calibração estatística."""
# Centraliza a regra de aceitação da raça. Um score alto sozinho não basta: também exigimos separação da segunda opção e concordância entre as duas vistas.
MIN_BREED_SCORE = 0.75
MIN_MARGIN = 0.15


# Compara o primeiro e o segundo score e verifica se a foto original e a espelhada escolheram a mesma raça.
def breed_decision(scores, winners, threshold):
    ordered = sorted(scores, reverse=True)
    margin = ordered[0] - ordered[1] if len(ordered) > 1 else ordered[0]
    stable = len(set(winners)) == 1
    accepted = ordered[0] >= max(MIN_BREED_SCORE, threshold) and margin >= MIN_MARGIN and stable
    return {'raca_aceita': accepted, 'margem': margin, 'consistente': stable}
