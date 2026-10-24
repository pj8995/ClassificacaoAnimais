"""Classificador Stanford Dogs local, com verificação de consistência visual."""
# Implementação anterior com modelo canino pronto, mantida como referência. O fluxo web atual utiliza own_dog_classifier.py.
from pathlib import Path
import re
from PIL import ImageOps
from breed_quality import breed_decision
from pretrained_classifier import MODEL_DIR, FILES, normalize_label

DOG_MODEL_ID = 'firstoff/animalmind-breed-classifier'
DOG_REVISION = 'f0ac4847e5972d57eb9577b7fb0706aba1d01281'
DOG_DIR = MODEL_DIR.parent / 'vit-stanford-dogs'


# Verifica se os arquivos locais do modelo anterior existem.
def dog_downloaded():
    return all((DOG_DIR / name).is_file() for name in FILES)


# Encapsula o classificador canino pré-treinado da implementação anterior.
class DogClassifier:
    # Carrega a configuração e os pesos locais do modelo e prepara sua transformação.
    def __init__(self):
        import torch
        from transformers import AutoImageProcessor, AutoModelForImageClassification
        self.torch = torch
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.processor = AutoImageProcessor.from_pretrained(str(DOG_DIR), local_files_only=True,
                                                            trust_remote_code=False, use_fast=False)
        self.model = AutoModelForImageClassification.from_pretrained(str(DOG_DIR), local_files_only=True,
                                         trust_remote_code=False, use_safetensors=True).to(self.device).eval()
        self.labels = [normalize_label(re.sub(r'^n\d{8}-', '', self.model.config.id2label[i]))
                       for i in range(self.model.config.num_labels)]
        if len(self.labels) != 120 or any(name.startswith('label_') or name == 'test' for name in self.labels):
            raise ValueError('Checkpoint canino sem os 120 rótulos esperados.')

    # Calcula scores em duas vistas, aplica a temperatura configurada e verifica a qualidade da primeira raça.
    def predict(self, image, threshold=0.6):
        image = image.convert('RGB')
        inputs = self.processor(images=[image, ImageOps.mirror(image)], return_tensors='pt')
        with self.torch.inference_mode():
            logits = self.model(**{k: v.to(self.device) for k, v in inputs.items()}).logits
            # Temperatura documentada pelo autor; não representa calibração em fotos do usuário.
            per_view = (logits / 1.7221).softmax(-1).cpu()
        probabilities = per_view.mean(0)
        quality = breed_decision(probabilities.tolist(), per_view.argmax(-1).tolist(), threshold)
        values, indices = probabilities.topk(3)
        candidates = [{'animal': 'cachorro', 'raca': self.labels[i], 'score': score}
                      for score, i in zip(values.tolist(), indices.tolist())]
        return {'previsao': candidates[0], 'alternativas': candidates, **quality, 'modelo_raca': DOG_MODEL_ID}
