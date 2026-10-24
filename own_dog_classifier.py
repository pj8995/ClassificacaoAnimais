"""Inferência das raças com o checkpoint treinado neste projeto."""
# Faz a previsão de raças com os pesos treinados neste projeto. A média entre imagem original e espelhada reduz a dependência de uma única vista.
from pathlib import Path
from PIL import ImageOps
from breed_quality import breed_decision

OWN_CHECKPOINT = Path(__file__).resolve().parent / 'outputs' / 'do_zero_ampliado' / 'cachorro' / 'best.pt'


# Verifica a existência do checkpoint canino antes de tentar carregar os pesos.
def own_model_ready():
    return OWN_CHECKPOINT.is_file()


# Encapsula o carregamento da rede de raças e a transformação usada na inferência.
class OwnDogClassifier:
    # Carrega o checkpoint, confere a espécie das classes e prepara a rede para avaliação.
    def __init__(self, checkpoint=OWN_CHECKPOINT, animal='cachorro'):
        from animal_classifier import load_model, transform, device
        import torch
        self.torch = torch
        self.model, self.classes = load_model(checkpoint)
        if not self.classes or any(label['animal'] != animal for label in self.classes):
            raise ValueError(f'O checkpoint próprio precisa conter somente classes de {animal}.')
        self.transform, self.device = transform(), device()

    # Processa duas vistas da foto, tira a média dos scores e aplica os critérios de aceitação da raça.
    def predict(self, image, threshold=.6):
        batch = self.torch.stack([self.transform(image), self.transform(ImageOps.mirror(image))]).to(self.device)
        with self.torch.inference_mode():
            per_view = self.model(batch).softmax(-1).cpu()
        probabilities = per_view.mean(0)
        quality = breed_decision(probabilities.tolist(), per_view.argmax(-1).tolist(), threshold)
        values, indices = probabilities.topk(min(3, len(self.classes)))
        candidates = [{**self.classes[i], 'score': score} for score, i in zip(values.tolist(), indices.tolist())]
        return {'previsao': candidates[0], 'alternativas': candidates, **quality,
                'modelo_raca': 'EfficientNet-B0 própria: origem do zero e continuação com dados adicionais'}
