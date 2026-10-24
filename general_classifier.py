"""Reconhecimento geral ImageNet seguido de raça somente para cães e gatos."""
# Coordena a identificação: define cão ou gato, chama o modelo de raças correspondente.
from pathlib import Path
from pretrained_classifier import MODEL_DIR
from own_cat_classifier import OwnCatClassifier
from own_dog_classifier import OwnDogClassifier

GENERAL_PATH = MODEL_DIR.parent / 'efficientnet_b0_rwightman-7f5810bc.pth'
PET_ONLY_MESSAGE = 'Não foi possivel identificar, animal não é um cão ou um gato'


# Traduz os índices do ImageNet em tipos de animal, mantendo animais parecidos, como lobos e grandes felinos, separados de cães e gatos.
def animal_type(index):
    # Índices da taxonomia IMAGENET1K_V1 do torchvision, incluindo classes não animais.
    ranges = [(0, 1, 'peixe'), (2, 4, 'tubarão'), (5, 6, 'arraia'),
              (7, 8, 'galinha'), (9, 9, 'avestruz'), (10, 24, 'ave'),
              (25, 29, 'salamandra'), (30, 32, 'sapo'), (33, 37, 'tartaruga'),
              (38, 48, 'lagarto'), (49, 50, 'crocodilo'), (51, 51, 'dinossauro'),
              (52, 68, 'cobra'), (69, 69, 'trilobita'), (70, 70, 'opilião'),
              (71, 71, 'escorpião'), (72, 77, 'aranha'), (78, 78, 'carrapato'),
              (79, 79, 'centopeia'), (80, 83, 'ave'), (84, 84, 'pavão'),
              (85, 86, 'ave'), (87, 87, 'papagaio'), (88, 88, 'arara'),
              (89, 89, 'cacatua'), (90, 90, 'papagaio'), (91, 96, 'ave'),
              (97, 98, 'pato'), (99, 99, 'ganso'), (100, 100, 'cisne'),
              (101, 101, 'elefante'), (102, 102, 'equidna'), (103, 103, 'ornitorrinco'),
              (104, 104, 'canguru'), (105, 105, 'coala'), (106, 106, 'vombate'),
              (107, 107, 'água-viva'), (108, 109, 'anêmona ou coral'),
              (110, 111, 'verme'), (112, 117, 'molusco'), (118, 121, 'caranguejo'),
              (122, 123, 'lagosta'), (124, 126, 'crustáceo'), (127, 144, 'ave'),
              (145, 145, 'pinguim'), (146, 146, 'albatroz'), (147, 148, 'baleia'),
              (149, 149, 'dugongo'), (150, 150, 'leão-marinho'),
              (151, 268, 'cachorro'), (269, 271, 'lobo'), (272, 272, 'coiote'),
              (273, 275, 'canídeo selvagem'), (276, 276, 'hiena'), (277, 280, 'raposa'),
              (281, 285, 'gato'), (286, 286, 'puma'), (287, 287, 'lince'),
              (288, 289, 'leopardo'), (290, 290, 'onça-pintada'), (291, 291, 'leão'),
              (292, 292, 'tigre'), (293, 293, 'guepardo'), (294, 297, 'urso'),
              (298, 298, 'mangusto'), (299, 299, 'suricato'), (300, 307, 'besouro'),
              (308, 308, 'mosca'), (309, 309, 'abelha'), (310, 310, 'formiga'),
              (311, 311, 'gafanhoto'), (312, 312, 'grilo'), (313, 313, 'bicho-pau'),
              (314, 314, 'barata'), (315, 315, 'louva-a-deus'), (316, 316, 'cigarra'),
              (317, 318, 'inseto'), (319, 320, 'libélula'), (321, 326, 'borboleta'),
              (327, 327, 'estrela-do-mar'), (328, 328, 'ouriço-do-mar'),
              (329, 329, 'pepino-do-mar'), (330, 332, 'coelho'), (333, 333, 'hamster'),
              (334, 334, 'porco-espinho'), (335, 335, 'esquilo'), (336, 336, 'marmota'),
              (337, 337, 'castor'), (338, 338, 'porquinho-da-índia'), (339, 339, 'cavalo'),
              (340, 340, 'zebra'), (341, 341, 'porco'), (342, 343, 'javali'),
              (344, 344, 'hipopótamo'), (345, 345, 'bovino'), (346, 346, 'búfalo'),
              (347, 347, 'bisão'), (348, 349, 'ovelha'), (350, 350, 'íbex'),
              (351, 353, 'antílope'), (354, 354, 'camelo'), (355, 355, 'lhama'),
              (356, 359, 'mustelídeo'), (360, 360, 'lontra'), (361, 361, 'cangambá'),
              (362, 362, 'texugo'), (363, 363, 'tatu'), (364, 364, 'preguiça'),
              (365, 365, 'orangotango'), (366, 366, 'gorila'), (367, 367, 'chimpanzé'),
              (368, 382, 'macaco'), (383, 384, 'lêmure'), (385, 386, 'elefante'),
              (387, 387, 'panda-vermelho'), (388, 388, 'panda'), (389, 397, 'peixe')]
    for start, end, name in ranges:
        if start <= index <= end:
            return name
    return 'não identificado'


# Soma os scores por tipo de animal e aplica a regra que recusa resultados fora do escopo.
def summarize(probabilities, categories, threshold):
    grouped = {}
    for index, score in enumerate(probabilities):
        name = animal_type(index)
        grouped[name] = grouped.get(name, 0.) + float(score)
    ranked = sorted(grouped.items(), key=lambda item: item[1], reverse=True)
    candidates = [{'animal': name, 'raca': 'não identificada' if name in {'gato', 'cachorro'} else 'não se aplica',
                   'score': score} for name, score in ranked[:3]]
    best_index = max(range(len(probabilities)), key=lambda i: probabilities[i])
    if candidates[0]['animal'] not in {'gato', 'cachorro'} or candidates[0]['score'] < max(.35, threshold):
        return {'previsao': {'animal': 'não identificado', 'raca': 'não se aplica', 'score': candidates[0]['score']},
                'alternativas': [], 'aceito': False, 'raca_aceita': False,
                'fora_do_escopo': True, 'codigo': 'cao_ou_gato_nao_identificado',
                'mensagem': PET_ONLY_MESSAGE, 'limiar': threshold,
                'scores_por_tipo': {'gato': grouped.get('gato', 0.), 'cachorro': grouped.get('cachorro', 0.),
                                   'fora_do_escopo': sum(v for k, v in grouped.items() if k not in {'gato', 'cachorro'})}}
    return {'previsao': candidates[0], 'alternativas': candidates, 'scores_por_tipo': grouped,
            'fora_do_escopo': False,
            'aceito': candidates[0]['animal'] != 'não identificado' and candidates[0]['score'] >= threshold,
            'limiar': threshold, 'detalhe': categories[best_index],
            'modelo': 'EfficientNet-B0 + classificadores próprios de cães e gatos'}


# Mantém os modelos auxiliares e encaminha cada foto ao classificador da espécie escolhida.
class AnimalClassifier:
    # Carrega a rede geral e, quando seus pesos locais existem, o detector COCO. Os modelos de raça são carregados conforme o uso.
    def __init__(self, pet_path=MODEL_DIR):
        import torch
        from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
        self.torch = torch
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        from pet_detector import DETECTOR_PATH, PetDetector
        self.detector = PetDetector(self.device) if DETECTOR_PATH.is_file() else None
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1
        self.categories = weights.meta['categories']
        self.transform = weights.transforms()
        self.general = efficientnet_b0(weights=None)
        self.general.load_state_dict(torch.load(GENERAL_PATH, map_location='cpu', weights_only=True))
        self.general.to(self.device).eval()
        self.pet_path = pet_path
        self.pet = None
        self.dog = None

    # No modo automático, combina a classificação geral com a detecção de cães e gatos. Com o tipo informado, usa a escolha do usuário e depois avalia a raça.
    def predict(self, image, threshold=0.6, animal_hint='auto'):
        import math
        if not math.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError('Limiar deve estar entre zero e um.')
        if animal_hint not in {'auto','gato','cachorro'}:
            raise ValueError('Tipo informado inválido.')
        detection = None
        if animal_hint != 'auto':
            result = {'previsao':{'animal':animal_hint,'raca':'não identificada','score':1.},
                      'fora_do_escopo':False,'aceito':False,'alternativas':[],'limiar':threshold,
                      'tipo_informado':True,'origem_tipo':'usuario','scores_por_tipo':{}}
        else:
            with self.torch.inference_mode():
                probabilities = self.general(self.transform(image).unsqueeze(0).to(self.device)).softmax(-1)[0].cpu().tolist()
            result = summarize(probabilities, self.categories, threshold)
            detection = self.detector.predict(image) if getattr(self, 'detector', None) is not None else None
        if detection is not None:
            result.update(fora_do_escopo=False,
                          previsao={'animal':detection['animal'], 'raca':'não identificada', 'score':detection['score']},
                          deteccao=detection, origem_tipo='detector_coco')
            result.pop('mensagem', None)
            result.pop('codigo', None)
        if result['fora_do_escopo']:
            return result
        animal = result['previsao']['animal']
        if animal in {'gato', 'cachorro'} and (detection is not None or result['previsao']['score'] >= max(0.35, threshold)):
            if animal == 'cachorro':
                if self.dog is None:
                    self.dog = OwnDogClassifier()
                breed = self.dog.predict(image, threshold)
            else:
                if self.pet is None:
                    self.pet = OwnCatClassifier()
                breed = self.pet.predict(image, threshold)
            type_score = result['previsao']['score']
            for item in breed['alternativas']:
                item['score'] *= type_score
            confirmed = breed['raca_aceita'] and breed['alternativas'][0]['score'] >= threshold
            prediction = dict(breed['alternativas'][0])
            if not confirmed:
                prediction.update(raca='não identificada', score=None if result.get('tipo_informado') else type_score)
            result.update(previsao=prediction, alternativas=breed['alternativas'], aceito=confirmed,
                          raca_aceita=confirmed, score_tipo=type_score,
                          margem_raca=breed['margem'], consistente=breed['consistente'])
            result['modelo_raca'] = breed.get('modelo_raca', 'EfficientNet-B0 própria')
            if result.get('tipo_informado'):
                result['score_tipo'] = None
            if not confirmed:
                result['aviso'] = 'Tipo reconhecido, mas a raça não foi confirmada. As alternativas são hipóteses. Tente uma foto nítida do rosto e corpo, com um único animal.'
        return result
