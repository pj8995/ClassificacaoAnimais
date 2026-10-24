"""Prepara os pesos ImageNet usados como backbone e filtro de cães/gatos."""
# Prepara os pesos auxiliares pré-treinados. As camadas de classificação próprias ainda precisam ser treinadas com os scripts do projeto.
from general_classifier import GENERAL_PATH


# Obtém os pesos ImageNet e do detector e mostra os comandos necessários para treinar os classificadores locais.
def main():
    import torch
    from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0
    GENERAL_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not GENERAL_PATH.is_file():
        print('Baixando reconhecedor geral de animais (aproximadamente 21 MB)...', flush=True)
        state = torch.hub.load_state_dict_from_url(EfficientNet_B0_Weights.IMAGENET1K_V1.url,
                    model_dir=str(GENERAL_PATH.parent), file_name=GENERAL_PATH.name,
                    check_hash=True, map_location='cpu')
    else:
        state = torch.load(GENERAL_PATH, map_location='cpu', weights_only=True)
    general = efficientnet_b0(weights=None)
    general.load_state_dict(state)
    from pet_detector import prepare
    prepare()
    print('Backbone e filtro geral validados. O frontend aceita somente cães e gatos.', flush=True)
    print('Para treinar as raças caninas locais: python preparar_dataset.py e python treinar_racas.py', flush=True)
    print('Para gatos: python preparar_gatos.py e python treinar_racas.py --manifest datasets/oxford_cats.json --output outputs/treino_gatos', flush=True)


if __name__ == '__main__':
    main()
