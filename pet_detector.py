"""Detector COCO de cães/gatos: usa a imagem inteira, sem recorte central."""
# O detector COCO procura cães e gatos na imagem inteira. Ele ajuda a corrigir casos em que o recorte central do classificador geral perde o animal.
from pathlib import Path

DETECTOR_PATH = Path(__file__).resolve().parent/'modelos'/'fasterrcnn_resnet50_fpn_v2_coco-dd69338a.pth'


# Percorre as detecções por score e devolve o cão ou gato mais bem pontuado que supera o limiar.
def select_pet(prediction, categories, threshold=.5):
    for label, score, box in sorted(zip(prediction['labels'], prediction['scores'], prediction['boxes']),
                                    key=lambda item: float(item[1]), reverse=True):
        name = categories[int(label)]
        if name in {'cat', 'dog'} and float(score) >= threshold:
            return {'animal':'gato' if name == 'cat' else 'cachorro', 'score':float(score),
                    'box':[float(v) for v in box]}
    return None


# Encapsula o Faster R-CNN com pesos locais e sua transformação de entrada.
class PetDetector:
    # Monta a arquitetura sem baixar pesos implicitamente e carrega o checkpoint local no dispositivo selecionado.
    def __init__(self, device):
        import torch
        from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
        self.torch, self.device = torch, device
        weights = FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1
        self.categories, self.transform = weights.meta['categories'], weights.transforms()
        self.model = fasterrcnn_resnet50_fpn_v2(weights=None, weights_backbone=None,
                                                  num_classes=len(self.categories))
        self.model.load_state_dict(torch.load(DETECTOR_PATH,map_location='cpu',weights_only=True))
        self.model.to(device).eval()

    # Envia a foto inteira ao detector e seleciona a detecção de cão ou gato aceita.
    def predict(self, image):
        with self.torch.inference_mode():
            prediction = self.model([self.transform(image).to(self.device)])[0]
        return select_pet(prediction, self.categories)


# Baixa os pesos oficiais com verificação de hash e valida seu carregamento para os próximos usos locais.
def prepare():
    import torch
    from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights
    weights = FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1
    # O nome do arquivo vem da URL oficial, mantendo a verificação do hash.
    path = DETECTOR_PATH.parent / weights.url.rsplit('/', 1)[-1]
    if path.name != DETECTOR_PATH.name:
        raise ValueError('URL de pesos inesperada; confira a versão do torchvision.')
    path.parent.mkdir(exist_ok=True)
    torch.hub.load_state_dict_from_url(weights.url,model_dir=str(path.parent),check_hash=True,map_location='cpu')
    PetDetector(torch.device('cpu'))
    print('Detector de cães/gatos preparado e validado.',flush=True)


if __name__ == '__main__':
    prepare()
