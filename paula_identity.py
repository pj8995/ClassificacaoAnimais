"""Reconhecimento experimental Paula/outro gato, separado das raças."""
# O reconhecimento individual usa características extraídas pelo modelo felino e uma camada binária Paula/outros gatos. Isso não transforma Paula em uma raça.
from pathlib import Path
import hashlib
from PIL import ImageOps
from paula_phases import phase_decision

ROOT = Path(__file__).resolve().parent
IDENTITY_PATH = ROOT/'outputs'/'paula'/'identity.pt'


# Só confirma Paula quando as duas vistas atingem o limiar, evitando que uma média esconda uma vista pouco convincente.
def identity_decision(scores, threshold):
    return len(scores) == 2 and min(scores) >= threshold


# Carrega a camada de identidade e, se existir, a camada que distingue filhote de adulta.
class PaulaRecognizer:
    # Confere se os pesos de identidade foram treinados com o mesmo checkpoint felino usado agora, comparando o SHA-256.
    def __init__(self, feature_checkpoint):
        import torch
        self.torch = torch
        self.saved = torch.load(IDENTITY_PATH, map_location='cpu', weights_only=True)
        if self.saved['feature_checkpoint_sha256'] != hashlib.sha256(Path(feature_checkpoint).read_bytes()).hexdigest():
            raise ValueError('Reconhecimento da Paula precisa ser retreinado após mudar o modelo felino.')

    # Extrai características normalizadas das duas vistas e calcula a identidade. A fase só é avaliada depois de confirmar Paula.
    def predict(self, image, model, transform, device):
        torch = self.torch
        batch = torch.stack([transform(image), transform(ImageOps.mirror(image))]).to(device)
        with torch.inference_mode():
            embeddings = model.avgpool(model.features(batch)).flatten(1).cpu()
            embeddings = torch.nn.functional.normalize(embeddings, dim=-1)
            scores = (embeddings @ self.saved['weight'] + self.saved['bias']).sigmoid().tolist()
        identified = identity_decision(scores, self.saved['threshold'])
        # Sem uma camada de fase treinada, a resposta diz que essa etapa está indisponível; nunca presume que Paula é adulta.
        phase = {'disponivel':False,'confirmada':False,'fase':None,
                 'mensagem':'Distinção filhote/adulta ainda não treinada. Adicione fotos em Paula filhote e execute treinar_paula.py.'}
        saved_phase = self.saved.get('phase')
        if not identified and saved_phase is not None:
            phase.update(disponivel=True,mensagem='Identidade da Paula não confirmada; fase não avaliada.')
        # A mesma característica visual alimenta a camada de fase, mas só depois da confirmação de identidade.
        if identified and saved_phase is not None:
            with torch.inference_mode():
                phase_scores = (embeddings @ saved_phase['weight'].T + saved_phase['bias']).softmax(-1).tolist()
            index = phase_decision(phase_scores,saved_phase['threshold'])
            name = saved_phase['classes'][index] if index is not None else None
            phase = {'disponivel':True,'confirmada':index is not None,'fase':name,
                     'rotulo':f'Paula {name}' if name else None,
                     'mensagem':f'Paula {name}' if name else 'Paula reconhecida; fase não confirmada.',
                     'observacao':'Fase visual aprendida com suas fotos; não é uma estimativa da idade cronológica.'}
        return {'disponivel': True, 'identificada': identified, 'nome': 'Paula' if identified else None,
                'fase':phase,
                'score': sum(scores)/2, 'limiar': self.saved['threshold'],
                'mensagem': 'Paula' if identified else 'Não foi possível confirmar se é a Paula',
                'observacao': 'Reconhecimento experimental; pode confundir gatos parecidos. Score não calibrado.'}
