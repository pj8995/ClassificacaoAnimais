"""Inferência local com ViT Oxford-IIIT Pets; não requer treinamento próprio."""
# Implementação anterior com modelo pronto de raças de cães e gatos. A lista de raças felinas também é reaproveitada na preparação dos dados.
from pathlib import Path
import math
import re
from PIL import ImageOps
from breed_quality import breed_decision

MODEL_ID = 'rakib730/vit-base-oxford-iiit-pets'
REVISION = '1756354b3aa3170ede5610f3022c32e99849954c'
MODEL_DIR = Path(__file__).resolve().parent / 'modelos' / 'vit-oxford-pets'
FILES = ('config.json', 'preprocessor_config.json', 'model.safetensors')
CAT_BREEDS = {
    'abyssinian', 'bengal', 'birman', 'bombay', 'british_shorthair', 'egyptian_mau',
    'maine_coon', 'persian', 'ragdoll', 'russian_blue', 'siamese', 'sphynx',
}
DOG_BREEDS = {
    'american_bulldog', 'american_pit_bull_terrier', 'basset_hound', 'beagle', 'boxer',
    'chihuahua', 'english_cocker_spaniel', 'english_setter', 'german_shorthaired',
    'great_pyrenees', 'havanese', 'japanese_chin', 'keeshond', 'leonberger',
    'miniature_pinscher', 'newfoundland', 'pomeranian', 'pug', 'saint_bernard',
    'samoyed', 'scottish_terrier', 'shiba_inu', 'staffordshire_bull_terrier',
    'wheaten_terrier', 'yorkshire_terrier',
}


# Padroniza o nome da classe para o formato usado no projeto.
def normalize_label(label):
    return re.sub(r'[\s-]+', '_', str(label).strip().lower())


# Relaciona os rótulos do modelo às espécies e raças conhecidas.
def map_classes(id2label):
    """Usa os rótulos do checkpoint, sem presumir a ordem dos índices."""
    labels = [normalize_label(id2label[i] if i in id2label else id2label[str(i)])
              for i in range(len(id2label))]
    if len(labels) != 37 or set(labels) != CAT_BREEDS | DOG_BREEDS:
        raise ValueError('Os rótulos do checkpoint não correspondem às 37 raças Oxford Pets.')
    return [{'animal': 'gato' if name in CAT_BREEDS else 'cachorro', 'raca': name}
            for name in labels]


# Confere a presença dos arquivos locais antes de carregar o modelo anterior.
def model_downloaded(path=MODEL_DIR):
    return all((Path(path) / name).is_file() for name in FILES)


# Agrupa os recursos do antigo classificador de raças Oxford Pets.
class PetClassifier:
    # Lê configuração e pesos locais e prepara as classes e a transformação das imagens.
    def __init__(self, path=MODEL_DIR):
        import torch
        from transformers import AutoImageProcessor, AutoModelForImageClassification
        self.torch = torch
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.processor = AutoImageProcessor.from_pretrained(
            str(path), local_files_only=True, trust_remote_code=False, use_fast=False)
        self.model = AutoModelForImageClassification.from_pretrained(
            str(path), local_files_only=True, trust_remote_code=False, use_safetensors=True)
        self.classes = map_classes(self.model.config.id2label)
        self.model.to(self.device).eval()

    # Calcula os scores das duas vistas e filtra pela espécie, sem renormalizar os scores das classes restantes.
    def predict(self, image, threshold=0.6, top_k=3, animal=None):
        if not math.isfinite(threshold) or not 0 <= threshold <= 1 or top_k < 1:
            raise ValueError('Limiar deve estar entre 0 e 1 e top_k deve ser positivo.')
        image = image.convert('RGB')
        inputs = self.processor(images=[image, ImageOps.mirror(image)], return_tensors='pt')
        inputs = {name: tensor.to(self.device) for name, tensor in inputs.items()}
        with self.torch.inference_mode():
            logits = self.model(**inputs).logits
            if animal is not None:
                if animal not in {'gato', 'cachorro'}:
                    raise ValueError('Tipo incompatível com o classificador de raças.')
                mask = self.torch.tensor([c['animal'] != animal for c in self.classes], device=self.device)
            per_view = logits.softmax(-1).cpu()
            # Não renormaliza as raças após escolher o tipo: preserva a incerteza entre espécies.
            if animal is not None:
                per_view[:, mask.cpu()] = 0
            probabilities = per_view.mean(0)
        scores, indices = probabilities.topk(min(top_k, len(self.classes)))
        candidates = [{**self.classes[i], 'score': score}
                      for score, i in zip(scores.tolist(), indices.tolist())]
        types = {'gato': 0., 'cachorro': 0.}
        for label, score in zip(self.classes, probabilities.tolist()):
            types[label['animal']] += score
        quality = breed_decision(probabilities.tolist(), per_view.argmax(-1).tolist(), threshold)
        return {'aceito': quality['raca_aceita'], **quality, 'previsao': candidates[0],
                'alternativas': candidates, 'scores_por_tipo': types, 'limiar': threshold,
                'modelo': MODEL_ID, 'revisao': REVISION}
