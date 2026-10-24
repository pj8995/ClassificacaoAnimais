"""Classificação de raças felinas; identificação individual retirada do produto."""
from pathlib import Path
from own_dog_classifier import OwnDogClassifier

CAT_CHECKPOINT = Path(__file__).resolve().parent/'outputs'/'do_zero_ampliado'/'gato'/'best.pt'


def cat_model_ready():
    """Indica se o checkpoint felino ativo está disponível."""
    return CAT_CHECKPOINT.is_file()


class OwnCatClassifier(OwnDogClassifier):
    """Reutiliza a inferência de raças com pesos exclusivos de gatos."""
    def __init__(self):
        super().__init__(checkpoint=CAT_CHECKPOINT, animal='gato')
