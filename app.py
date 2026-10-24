"""Frontend HTML + API Flask. Execute: python app.py."""
# Este arquivo liga a página web aos modelos. A foto chega pela API, é validada e gera uma resposta JSON; o histórico guarda os resultados, sem salvar a foto.
import csv
import io
import threading
import warnings
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from flask import Flask, jsonify, render_template, request, send_file
from PIL import Image, ImageOps, UnidentifiedImageError
from werkzeug.exceptions import RequestEntityTooLarge
from pretrained_classifier import MODEL_DIR
from general_classifier import GENERAL_PATH
from own_dog_classifier import own_model_ready
from own_cat_classifier import cat_model_ready
from evaluation_metrics import response_metrics, REPORT_PATH, CHECKPOINT_PATH, CAT_REPORT_PATH, CAT_CHECKPOINT_PATH

ROOT = Path(__file__).resolve().parent
FIELDS = ['id', 'data', 'arquivo', 'animal', 'raca', 'score', 'aceito', 'limiar']


# Monta a aplicação Flask. A configuração e o preditor podem ser trocados nos testes, sem depender dos modelos reais.
def create_app(config=None, predictor=None):
    app = Flask(__name__)
    app.config.update(MAX_CONTENT_LENGTH=10 * 1024 * 1024,
                      MODEL_PATH=str(MODEL_DIR),
                      METRICS_PATH=str(REPORT_PATH), METRICS_CHECKPOINT=str(CHECKPOINT_PATH),
                      CAT_METRICS_PATH=str(CAT_REPORT_PATH), CAT_METRICS_CHECKPOINT=str(CAT_CHECKPOINT_PATH),
                      HISTORY_PATH=str(ROOT / 'dados' / 'historico.csv'))
    if config:
        app.config.update(config)
    # Os locks protegem o acesso ao CSV e ao modelo em requisições concorrentes; o dicionário mantém o modelo carregado na memória.
    history_lock = threading.Lock()
    model_lock = threading.Lock()
    cached = {}

    # Lê as linhas do CSV como dicionários. Se ainda não houve análises, devolve uma lista vazia.
    def read_history():
        path = Path(app.config['HISTORY_PATH'])
        if not path.exists():
            return []
        with path.open(encoding='utf-8', newline='') as stream:
            return list(csv.DictReader(stream))

    # Carrega o classificador apenas no primeiro uso e o reutiliza. O lock impede duas requisições de usarem esse trecho ao mesmo tempo.
    def classify(image, threshold, animal_hint='auto'):
        if predictor:
            return predictor(image, threshold)
        from general_classifier import AnimalClassifier
        path = Path(app.config['MODEL_PATH'])
        with model_lock:
            if 'classifier' not in cached:
                cached['classifier'] = AnimalClassifier(path)
            return cached['classifier'].predict(image, threshold, animal_hint=animal_hint)

    # Entrega o HTML inicial; o JavaScript buscará os dados e enviará as fotos pela API.
    @app.get('/')
    def index():
        return render_template('index.html')

    # Informa se os arquivos necessários existem e devolve o histórico do mais recente para o mais antigo.
    @app.get('/api/estado')
    def state():
        with history_lock:
            history = read_history()
        return jsonify(modelo_disponivel=bool(predictor) or (GENERAL_PATH.is_file() and own_model_ready() and cat_model_ready()),
                       modelo='EfficientNet com classificadores próprios canino e felino', classes=132,
                       historico=list(reversed(history)), total=len(history))

    # Recebe a foto, valida os campos e o conteúdo da imagem, executa a análise e registra o resultado.
    @app.post('/api/classificar')
    def classify_upload():
        # O navegador envia a foto como multipart/form-data. Campos ausentes ou inválidos recebem uma resposta de erro antes da inferência.
        uploaded = request.files.get('foto')
        if uploaded is None or not uploaded.filename:
            return jsonify(erro='Selecione uma foto para continuar.'), 400
        # O modo automático tenta detectar a espécie. Nos outros modos, o usuário informa o tipo; isso não é uma confirmação automática do modelo.
        animal_hint = request.form.get('tipo_animal', 'auto')
        if animal_hint not in {'auto','gato','cachorro'}:
            return jsonify(erro='Escolha identificação automática, gato ou cachorro.'),400
        try:
            threshold = float(request.form.get('limiar', '0.6'))
            if not 0 <= threshold <= 1:
                raise ValueError()
        except ValueError:
            return jsonify(erro='O limiar deve ser um número entre 0 e 1.'), 400
        # Não confiamos apenas na extensão do arquivo: o Pillow abre o conteúdo, limita os pixels e corrige a orientação antes da análise.
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('error', Image.DecompressionBombWarning)
                with Image.open(uploaded.stream) as original:
                    if original.format not in {'JPEG', 'PNG', 'WEBP'}:
                        return jsonify(erro='Use uma imagem JPG, PNG ou WEBP.'), 400
                    if original.width * original.height > 20_000_000:
                        return jsonify(erro='A imagem deve ter até 20 megapixels.'), 400
                    original.load()
                    image = ImageOps.exif_transpose(original).convert('RGB')
        except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError,
                Image.DecompressionBombWarning):
            return jsonify(erro='Arquivo inválido. Envie uma foto JPG, PNG ou WEBP válida.'), 400
        if not predictor and not own_model_ready():
            return jsonify(erro='Treine o modelo próprio: python preparar_dataset.py e depois python treinar_racas.py.'), 503
        if not predictor and not cat_model_ready():
            return jsonify(erro='Treine os gatos: python preparar_gatos.py e python treinar_racas.py --manifest datasets/oxford_cats.json --output outputs/treino_gatos.'), 503
        if not predictor and not GENERAL_PATH.is_file():
            return jsonify(erro='Prepare os modelos auxiliares com: python preparar_modelo.py.'), 503
        try:
            result = classify(image, threshold, animal_hint)
        except ImportError:
            app.logger.exception('Dependências de classificação ausentes')
            return jsonify(erro='Instale as dependências do modelo com pip install -r requirements.txt.'), 503
        except Exception:
            app.logger.exception('Falha ao executar o modelo')
            return jsonify(erro='Não foi possível analisar a foto. Verifique o modelo e o terminal do servidor.'), 500
        best = result['previsao']
        prefix = 'CAT_' if best['animal'] == 'gato' else ''
        result['metricas'] = response_metrics(best['animal'], best['raca'],
                         app.config[prefix + 'METRICS_PATH'], app.config[prefix + 'METRICS_CHECKPOINT'])
        row = {'id': str(uuid4()), 'data': datetime.now(timezone.utc).isoformat(),
               'arquivo': uploaded.filename.replace('\\', '/').split('/')[-1][:200],
               'animal': best['animal'], 'raca': best['raca'], 'score': best['score'],
               'aceito': result['aceito'], 'limiar': threshold}
        try:
            with history_lock:
                path = Path(app.config['HISTORY_PATH'])
                path.parent.mkdir(parents=True, exist_ok=True)
                header = not path.exists() or path.stat().st_size == 0
                with path.open('a', encoding='utf-8', newline='') as stream:
                    writer = csv.DictWriter(stream, fieldnames=FIELDS)
                    if header:
                        writer.writeheader()
                    writer.writerow(row)
        except OSError:
            app.logger.exception('Falha ao salvar histórico')
            result['aviso'] = 'Análise concluída, mas não foi possível salvar o histórico.'
        return jsonify(**result, registro=row)

    # Exporta o histórico como CSV. O tratamento dos campos evita que planilhas interpretem texto como fórmulas.
    @app.get('/download/historico')
    def download_history():
        with history_lock:
            rows = read_history()
        stream = io.StringIO(newline='')
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: ("'" + str(value) if str(value).lstrip().startswith(('=', '+', '-', '@'))
                                   else value) for key, value in row.items()})
        return send_file(io.BytesIO(stream.getvalue().encode('utf-8-sig')),
                         mimetype='text/csv; charset=utf-8', as_attachment=True,
                         download_name='historico_animais.csv')

    # Transforma o erro de upload acima do limite em uma mensagem JSON que a interface consegue mostrar.
    @app.errorhandler(RequestEntityTooLarge)
    def too_large(error):
        return jsonify(erro='A foto deve ter no máximo 10 MB.'), 413

    return app


app = create_app()

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=False)
