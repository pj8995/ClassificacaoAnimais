# Animalia — classificação de cães e gatos

O projeto classifica raças de cães e gatos com EfficientNet-B0. A versão atual
carrega os melhores pesos próprios de `outputs/do_zero_ampliado/`, originados de
inicialização aleatória e continuados com fotos públicas adicionais. São 120 classes
caninas e 12 felinas. O detector COCO e o filtro geral ImageNet permanecem
pré-treinados. A identificação individual da Paula e suas fases foram retiradas da aplicação.

O treino canino ampliado foi interrompido voluntariamente após 21 épocas completas;
a aplicação usa o melhor checkpoint pela perda de validação. Veja a seção
“Versão integrada após interrupção voluntária” ao final deste documento.
As seções de transferência de aprendizado abaixo documentam o experimento anterior.

## Pastas

```text
data/cachorro/
  labrador_retriever/
  golden_retriever/
  standard_poodle/
  miniature_poodle/
  toy_poodle/
  beagle/
  chihuahua/
  ... outras raças ...
datasets/
  stanford_dogs.json       Manifesto com classes e caminhos dos conjuntos
  racas.csv               Quantidade de fotos por raça e partição
  resumo.json             Resumo da importação
downloads/                Cópia versionada da fonte e downloads
outputs/treino_proprio/
  best.pt                 Checkpoint canino treinado neste projeto
  history.json            Evolução do treino e validação
  test_metrics.json       Avaliação final no teste reservado
  manifest.json           Manifesto utilizado na execução
  features_*.pt           Características calculadas, para acelerar novas execuções
```

As pastas contêm imagens reais baixadas da base, não imagens geradas. Os nomes das raças vêm dos rótulos publicados. Stanford Dogs tem 120 classes, incluindo algumas de canídeos selvagens; isso não representa todas as raças existentes nem inclui uma classe confiável de mestiços.

## Instalação e execução

PowerShell, Python 3.11 ou 3.12:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe preparar_modelo.py
.\.venv\Scripts\python.exe preparar_dataset.py
.\.venv\Scripts\python.exe treinar_racas.py
.\.venv\Scripts\python.exe app.py
```

Se o ambiente e os modelos auxiliares já existem, pule suas etapas de preparação. Abra http://127.0.0.1:5000. Após um novo treinamento, reinicie `app.py` para carregar o checkpoint atualizado.

`preparar_dataset.py` baixa por padrão um espelho público versionado de Stanford Dogs no Hugging Face (cerca de 776 MB):
https://huggingface.co/datasets/Donghyun99/Stanford-Dogs/tree/4246493e3d3e3a15a43c291ec5e394a003a2e10a

A ficha identifica o espelho como não oficial. Preservamos a divisão treino/teste publicada nesse espelho; a correspondência imagem a imagem com as listas originais não foi auditada. A fonte acadêmica original é http://vision.stanford.edu/aditya86/ImageNetDogs/. Imagens têm origem no ImageNet; consulte as condições dos autores antes de redistribuir ou explorar comercialmente. A ficha da fonte é preservada na pasta de download.

O script verifica se os arquivos são imagens válidas, cria as pastas e gera o manifesto. A validação usa 20% do treino publicado. Duplicatas exatas são identificadas por SHA256: o teste tem prioridade sobre treino e validação; imagens idênticas com raças conflitantes são excluídas da divisão. As imagens permanecem nas pastas e somente as entradas usadas pelo treinamento são filtradas. Fotos quase idênticas e indivíduos repetidos não são detectados.

A importação é reexecutável e não sobrescreve imagens locais diferentes das baixadas. O manifesto e os relatórios são regenerados. Uma alternativa de importação direta dos arquivos originais é `preparar_dataset.py --source stanford`.

## Treinamento

```powershell
.\.venv\Scripts\python.exe -u treinar_racas.py --epochs 60 --patience 8
```

A primeira execução extrai as características das imagens em CPU ou GPU. Essa é a etapa mais demorada. Elas ficam em cache; novas execuções podem reutilizá-las quando os dados e pesos-base continuam iguais. O cache é invalidado se mudarem o manifesto, o conteúdo das imagens ou os pesos-base.

A camada de classificação é treinada com AdamW e dropout. A melhor época é escolhida pela menor perda de validação, com parada antecipada após oito épocas sem melhora. O teste é avaliado apenas depois da escolha do checkpoint. Não escolha parâmetros com base no resultado do teste.

O treinamento atual usa imagens redimensionadas com o pré-processamento oficial da EfficientNet, sem aumento aleatório de dados na extração congelada. É uma linha de base; ajuste fino da rede e aumento de dados podem ser incorporados em uma próxima etapa. `--threads` controla as threads de CPU; `--batch-size` controla o lote de extração. Para repetir outra experiência, preserve antes os relatórios e o checkpoint, pois os arquivos de saída são atualizados.

**Limite da avaliação:** Stanford Dogs deriva de imagens ImageNet. Como o backbone foi pré-treinado no ImageNet, pode ter visto imagens que agora estão no teste. Portanto, mesmo sem duplicatas entre os conjuntos locais, a métrica não representa um teste totalmente independente do pré-treinamento. Para medir generalização real, use fotos próprias rotuladas e nunca utilizadas nos modelos-base.

`animal_classifier.py` mantém um fluxo alternativo de ajuste de toda a rede:

```powershell
.\.venv\Scripts\python.exe animal_classifier.py train --data data --output outputs/experimento_completo
```

Esse fluxo alternativo faz uma nova divisão aleatória e **não usa o manifesto reservado**. Não use seus resultados para comparar diretamente com o protocolo atual. Seu checkpoint não substitui automaticamente `outputs/treino_proprio/best.pt`.

## Identificação na interface

1. A EfficientNet ImageNet identifica o tipo de animal.
2. Para cachorro, `own_dog_classifier.py` carrega o checkpoint treinado aqui.
3. Para gato, `own_cat_classifier.py` carrega o checkpoint felino treinado aqui, com 12 raças.
4. Outros animais, objetos e imagens sem reconhecimento suficiente de cão ou gato são recusados com a mensagem solicitada, sem sugestões de raça.

A confirmação de raça usa a foto original e uma versão espelhada, score mínimo de 0,75 (ou o limiar da interface, se maior), margem mínima de 0,15 e concordância entre as duas versões. O score final também considera o score do tipo. Quando a raça não passa nesses critérios, a tela informa `não identificada` e apresenta hipóteses. Scores não são probabilidades calibradas e não comprovam ancestralidade.

Os antigos modelos prontos de raça foram preservados em disco para referência, mas o canino pronto não é usado pelo frontend. `preparar_modelo.py` prepara somente o backbone e filtro ImageNet; os classificadores de cães e gatos são treinados localmente.

## Frontend, API e histórico

A arquitetura usa Flask (`app.py`), HTML (`templates/index.html`), CSS (`static/styles.css`) e JavaScript (`static/app.js`). O navegador envia a foto por `fetch()` ao Python. Uploads aceitam JPG, PNG e WEBP de até 10 MB e 20 megapixels.

| Rota | Função |
|---|---|
| `GET /` | Interface |
| `GET /api/estado` | Estado dos modelos e histórico |
| `POST /api/classificar` | Análise do arquivo `foto`, com `limiar` |
| `GET /download/historico` | Exportação CSV |

As fotos enviadas na interface não são salvas. O histórico das respostas fica em `dados/historico.csv`, compartilhado por quem acessa o servidor local. Datas são registradas em UTC e exibidas no horário do navegador. O servidor está configurado para uso local, em um único processo.

## Testes

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Testes verificam regras, importação, API e execução dos modelos; não comprovam precisão em fotos novas. Consulte as métricas do treinamento e avalie separadamente uma base independente antes de tirar conclusões sobre acurácia.


## Métricas no retorno da análise

`POST /api/classificar` retorna o campo `metricas` e a interface exibe cinco indicadores: `precisao`, `acuracia`, `recall`, `f1` e `f2`. Os valores da API estão entre 0 e 1; a tela apresenta percentuais. Precisão, recall, F1 e F2 são médias macro (cada classe tem o mesmo peso); acurácia é o total de acertos dividido pelo total de imagens. F2 dá peso quatro vezes maior aos falsos negativos do que aos falsos positivos na sua fórmula por classe: `5*TP / (5*TP + 4*FN + FP)`.

Esses indicadores descrevem o classificador da espécie reconhecida no seu conjunto de teste, antes do roteamento por tipo, dos filtros de aceitação e da análise espelhada da aplicação. Não representam a qualidade de uma única foto. O retorno também inclui as métricas da raça prevista quando há um rótulo correspondente no relatório. Para gatos, usa-se o relatório do teste felino; para outros tipos, `disponivel` é falso. Métricas de uma espécie nunca são reaproveitadas para outra.

O relatório contém um SHA256 do checkpoint. Se o arquivo de métricas estiver ausente ou não corresponder ao checkpoint atual, o retorno informa a indisponibilidade. Novos treinamentos geram os cinco indicadores e a assinatura automaticamente. Para a execução já concluída, os indicadores adicionais foram calculados a partir da matriz de confusão salva, sem novo treinamento.


## Escopo restrito a cães e gatos

O identificador agora aceita somente cães e gatos. A rede geral é usada como filtro: se o tipo mais provável não for cão/gato, ou seu score ficar abaixo do limiar de reconhecimento (mínimo de 0,35 e o limiar escolhido na tela), a resposta contém `fora_do_escopo: true`, `aceito: false`, `alternativas: []` e `mensagem: "Não foi possivel identificar, animal não é um cão ou um gato"`. A requisição válida retorna HTTP 200 com a indicação de recusa; arquivos inválidos continuam retornando erro HTTP 400.

A interface mostra somente a mensagem e pede outra foto, sem exibir raça ou métricas caninas para a imagem recusada. O filtro é estatístico e pode errar; uma imagem de cão/gato também pode ser recusada se não houver reconhecimento suficiente. O modelo não foi retreinado para essa alteração de escopo.


## Modelo próprio de gatos

O frontend agora utiliza dois classificadores treinados neste projeto. Para gatos, o ViT pronto foi substituído por uma EfficientNet-B0 com nova cabeça de 12 classes, treinada nas imagens locais de Oxford-IIIT Pets. O método é transferência de aprendizado com backbone ImageNet congelado, assim como no treinamento canino. A restrição a cães/gatos e as regras de confirmação de raça permanecem.

```powershell
.\.venv\Scripts\python.exe preparar_gatos.py
.\.venv\Scripts\python.exe -u treinar_racas.py --manifest datasets/oxford_cats.json --output outputs/treino_gatos
.\.venv\Scripts\python.exe app.py
```

- Fotos: `data/gato/<raca>/`.
- Manifesto: `datasets/oxford_cats.json`.
- Contagem: `datasets/racas_gatos.csv` e `datasets/resumo_gatos.json`.
- Checkpoint: `outputs/treino_gatos/best.pt`.
- Histórico e avaliação: `outputs/treino_gatos/history.json` e `test_metrics.json`.

Foram organizadas 2.371 fotos reais em 12 pastas: Abyssinian, Bengal, Birman, Bombay, British Shorthair, Egyptian Mau, Maine Coon, Persian, Ragdoll, Russian Blue, Siamese e Sphynx. A divisão resultou em 950 imagens de treino, 238 de validação e 1.183 de teste, sem exclusões por duplicação exata/conflitos. A divisão de teste publicada foi preservada; a validação usa 20% do treino, por raça.

Fonte: https://huggingface.co/datasets/timm/oxford-iiit-pet/tree/089695c834a7deb60505b7cc506672db1c31a6aa . Esse espelho preserva as divisões padronizadas de Oxford Pets. O download inclui o pacote de cães e gatos (cerca de 790 MB), mas somente as imagens das classes felinas são organizadas em `data/gato`. A ficha declara licença CC BY-SA 4.0 e fica preservada em `downloads/oxford_pets/README.md`; a publicação original é “Cats and Dogs”, Parkhi et al., CVPR 2012: https://www.robots.ox.ac.uk/~vgg/data/pets/ .

O parâmetro `--output` do treinamento separa os arquivos de gatos dos caninos; nenhum checkpoint canino é sobrescrito. Após treinar novamente qualquer modelo, reinicie o servidor para carregar a nova versão.

O retorno de gatos também inclui precisão, acurácia, recall, F1 e F2 calculados no teste felino. O relatório é selecionado por espécie e validado pelo SHA256 do checkpoint. Esses números avaliam o classificador de raça antes dos filtros da interface, não a confiança de uma foto nem a acurácia do sistema inteiro. A avaliação em fotos próprias independentes continua necessária.


## Reconhecimento individual da Paula

As fotos originais permanecem em `data/gato/paula`. Essa pasta **não é uma raça** e não entra no manifesto Oxford usado para o modelo de raças. O reconhecedor individual é um classificador binário separado, treinado com características congeladas do modelo felino: Paula versus outros gatos.

```powershell
.\.venv\Scripts\python.exe -u treinar_paula.py
```

A execução encontrou 40 arquivos e 39 fotos únicas, excluindo uma cópia idêntica somente da divisão de treinamento. As fotos não foram removidas nem enviadas a serviços externos. Imagens próximas pelo hash perceptual são mantidas no mesmo grupo; os grupos resultaram em 23 fotos de treino, 8 de validação e 8 de teste. O agrupamento visual é aproximado e não comprova separação por momento ou sessão fotográfica.

Foram usados também 240 outros gatos no treino, 60 na validação e 120 no teste, obtidos das respectivas partições do dataset Oxford já preparado. O limiar é escolhido na validação, com piso de 0,80, e precisa ser ultrapassado tanto na foto original quanto na espelhada. O teste final não é usado para escolher esse limiar.

Artefatos: `outputs/paula/identity.pt`, `manifest.json`, `history.json` e `test_metrics.json`. O reconhecedor verifica o SHA256 do checkpoint felino utilizado como extrator; após retreinar o modelo de raças de gatos, execute novamente `treinar_paula.py`. Reinicie `app.py` após treinar a identidade.

O retorno da API inclui `identidade` quando o filtro reconhece um gato: `nome: "Paula"` quando confirmada, ou `nome: null` quando não confirmada. A interface exibe a identidade separadamente da raça, e as métricas de raça continuam sendo as do classificador felino. O CSV histórico atual continua registrando tipo e raça, sem campo de identidade.

No pequeno teste local, foram reconhecidas 8/8 fotos positivas e rejeitadas 120/120 negativas. Esses números são exploratórios: fundos, poses repetidas e gatos visualmente distintos dos negativos podem facilitar o teste. Fotos novas de outras sessões, especialmente outros gatos brancos semelhantes, são necessárias para medir generalização. O reconhecimento pode errar; score não é probabilidade calibrada e não comprova identidade.

## Correção do filtro antes do reconhecimento da Paula

O diagnóstico das 40 fotos fornecidas mostrou que o filtro de classificação da imagem inteira reconhecia somente três delas como gato: 32 eram recusadas e cinco eram encaminhadas como cachorro. O reconhecedor de identidade, executado diretamente, confirmava as 40. Portanto, testar apenas uma foto positiva no fluxo completo era insuficiente.

O fluxo agora consulta também o detector oficial Faster R-CNN ResNet50 FPN V2 COCO do torchvision, que procura cães e gatos na imagem inteira, sem descartar as bordas em um recorte central. Uma detecção de `cat` ou `dog` com score mínimo de 0,50 tem prioridade sobre a classificação geral de espécie. A identificação de raça e da Paula continuam usando a foto original, preservando o pré-processamento usado no treinamento. Sem detecção confiável, o filtro geral anterior permanece como alternativa; outros tipos continuam recusados.

Os pesos adicionais (aproximadamente 168 MB) ficam em `modelos/fasterrcnn_resnet50_fpn_v2_coco-dd69338a.pth`; `python preparar_modelo.py` os prepara junto ao backbone. O download e a inferência usam apenas o modelo oficial; as fotos privadas continuam sendo processadas localmente. Referência: https://docs.pytorch.org/vision/main/models/generated/torchvision.models.detection.fasterrcnn_resnet50_fpn_v2.html

`python diagnosticar_paula.py` registra os resultados do filtro antigo e da identidade separadamente. `python validar_fluxo_paula.py` registra o novo fluxo completo em todas as fotos fornecidas, mais controles negativos; os relatórios ficam em `outputs/paula/`. Esse ensaio é uma regressão sobre imagens conhecidas e não substitui avaliação com novas fotos e outros gatos parecidos.


Nas fotos difíceis, a interface oferece `Tipo de animal → Gato — informado por mim` (ou Cão). A API aceita `tipo_animal` com valores `auto`, `gato` e `cachorro`. O padrão continua `auto`, preservando a recusa de imagens sem cão/gato reconhecido. No modo informado, o usuário declara a espécie e o sistema pula o filtro automático para consultar a raça e a identidade. O retorno marca `tipo_informado: true` e `origem_tipo: usuario`; não é uma detecção automática. Quando não confirma raça, não inventa um score de espécie: `score_tipo` e o score principal ficam nulos, exibidos como travessão.

Para validar especificamente esse fluxo nas fotos conhecidas: `python validar_fluxo_paula.py --modo gato`. O relatório fica em `outputs/paula/regressao_fluxo_informado.json`. Este modo não comprova a espécie informada pelo usuário e deve ser escolhido apenas para fotos que ele sabe serem de gatos ou cães.

## Paula filhote e Paula adulta

Coloque fotos de filhote em `data/gato/paula/Paula filhote/`. As fotos diretamente em `data/gato/paula/` (e em outras pastas que não se chamem Paula filhote) são tratadas como adultas. Os rótulos são fornecidos por você, não inferidos pela data do arquivo. Não coloque a mesma foto nas duas fases: o treinamento recusa cópias idênticas com rótulos conflitantes.

```powershell
.\.venv\Scripts\python.exe -u treinar_paula.py
```

Esse comando agora lê também as subpastas, treina novamente Paula/outro gato usando fotos das duas fases e, quando houver exemplos suficientes, treina uma cabeça separada filhote/adulta. O mínimo técnico é de 8 fotos únicas por fase em pelo menos 3 grupos visuais, com pelo menos 2 fotos de cada classe em cada partição. Prefira 20 a 40 fotos variadas por fase e sessões diferentes. Grupos perceptuais semelhantes permanecem juntos. Grupos contendo ambas as fases não são usados no treino de fase.

O modelo de fase escolhe a melhor época pela validação e registra sua avaliação separadamente em `outputs/paula/fases_metrics.json`. Seus pesos ficam no campo `phase` de `identity.pt`, junto ao reconhecedor individual. A distinção é visual e depende dos seus rótulos; não determina a idade cronológica da gata.

A fase só é exibida quando a identidade Paula é confirmada, as versões original/espelhada concordam e ambas atingem 0,80. Caso contrário, permanece Paula com fase não confirmada. Se faltarem fotos de filhote ou o treino de fase não for possível, não se presume adulta: a fase é marcada como não disponível. Até treinar com fotos nas duas fases, o modelo de identidade existente continua funcionando sem distinção de fase. Reinicie `app.py` após treinar.

Treinamento de fases concluído com 101 fotos únicas: 39 adultas e 62 filhotes. O teste de fase contém 21 fotos; o classificador top-1 acertou 17 (80,95%). Com os filtros de confirmação, apresentou fase em 16 fotos, todas corretas nesse pequeno teste, e se absteve em 5. Esses resultados são exploratórios e não garantem acerto em fotos novas. A identidade Paula/outro gato foi retreinada incluindo as duas fases, e a versão anterior foi preservada em uma subpasta `backup_*` de `outputs/paula`.


## Treinamento de todas as camadas do zero

O script `treinar_do_zero.py` inicia a EfficientNet-B0 com pesos aleatórios e atualiza todas as suas camadas. Não lê pesos ImageNet nem caches de características. Usar uma arquitetura conhecida não implica usar pesos pré-treinados. O pré-processamento mantém as dimensões e a normalização esperadas pelo carregador existente; isso não carrega conhecimento aprendido de outro modelo.

```powershell
.\.venv\Scripts\python.exe -u treinar_do_zero.py --animal todos
```

A fila começa pelas 12 raças de gatos e depois treina as 120 raças de cães, usando os manifestos já preparados. O padrão é até 60 épocas, com parada após 15 épocas sem redução da perda de validação, batch de 16, taxa inicial 0,001 e quatro threads. Este computador usa CPU; o treino pode ser demorado. O teste reservado só é avaliado depois da escolha do melhor checkpoint pela validação.

Os arquivos novos ficam separados em `outputs/do_zero/gato/` e `outputs/do_zero/cachorro/`: `best.pt` (melhor rede), `last.pt` (retomada com otimizador), `history.json`, `status.json`, `manifest.json` e, ao terminar, `test_metrics.json` com acurácia, precisão macro, recall macro, F1 e F2. O campo `training` dos pesos identifica `random_initialization_all_layers` e `pretrained` é falso. `status.json` informa época e lote; o estado concluído só aparece após a avaliação final.

O processo iniciado pelo assistente escreve `outputs/do_zero/treinamento.log`, `erros.log` e `processo.pid`. Para acompanhar:

```powershell
Get-Content .\outputs\do_zero\treinamento.log -Tail 20 -Wait
```

Se o processo parar, retome com os mesmos parâmetros e dados (não execute em paralelo com o treino ainda ativo):

```powershell
.\.venv\Scripts\python.exe -u treinar_do_zero.py --animal todos --resume
```

A retomada parte da última época completa; um lote interrompido será refeito. Os arquivos existentes exigem `--resume` ou uma nova pasta de saída para evitar sobrescrever outro experimento.

A aplicação web continua consultando os checkpoints anteriores enquanto esse experimento treina. Não há promoção automática dos novos pesos: primeiro devem ser avaliados. Este script substitui o treinamento de raças; o detector COCO, o filtro geral e o reconhecedor da Paula continuam no fluxo anterior. Ao futuramente trocar o checkpoint felino, será necessário retreinar a identidade e a fase da Paula com essa nova base. Portanto, iniciar este treino não torna imediatamente toda a aplicação livre de modelos pré-treinados.


## Continuação com fotos públicas adicionais

`baixar_dados_extras.py` baixa revisões fixas de três fontes: AtharvaTaras/Cat-Breeds-Dataset, CatInBeard/cat-breeds-dataset (Wikimedia com metadados individuais) e o espelho giacomov/tsinghua_dogs. Somente a divisão train de Tsinghua é usada. A base Tsinghua inclui imagens Stanford já existentes; os scripts não a tratam como uma fonte inteiramente independente.

`preparar_dados_extras.py` mantém as classes e a ordem originais, acrescenta exemplos apenas ao treino e conserva integralmente validação e teste. Exclui duplicatas SHA-256 e candidatos com distância pHash de até 6 bits em relação a qualquer imagem original ou nova já aceita. O filtro reduz cópias recomprimidas, mas não garante detectar todos os recortes, fotos do mesmo indivíduo ou sessão. Não há auditoria humana integral dos rótulos das fontes. Raças sem correspondência explícita são descartadas. Cada raça recebe no máximo 300 imagens adicionais.

As fotos aceitas ficam em `data/extras/gato/<raca>/` e `data/extras/cachorro/<raca>/`. Os manifestos ficam em `datasets/ampliado/`; `proveniencia_gato.json` e `proveniencia_cachorro.json` preservam origem, revisão, rótulo e SHA-256 por arquivo. Licenças e documentos das fontes ficam em `datasets/ampliado/fontes/`. As fotos são copiadas sem alteração de conteúdo; as transformações são feitas somente em memória durante o treinamento. A inclusão não significa que uma raça declarada pela fonte tenha sido confirmada geneticamente.

O novo experimento aproveita os nossos melhores pesos originados do zero e continua atualizando todas as camadas. Não usa ImageNet nem detector para criar rótulos. A referência inicial é avaliada na validação e permanece candidata a best.pt: só é substituída quando a perda de validação diminui. Acurácia de validação pode variar independentemente dessa perda. O otimizador é novo porque os dados e a agenda mudaram; isso é uma nova etapa de treinamento, não uma retomada idêntica do experimento antigo.

Comando usado (até 30 épocas adicionais, paciência 10, taxa inicial 0,0002):

```powershell
.\.venv\Scripts\python.exe -u treinar_do_zero.py --animal todos --manifest-dir datasets/ampliado --init-root outputs/do_zero --output outputs/do_zero_ampliado --epochs 30 --patience 10 --lr 0.0002
```

Para retomar uma etapa interrompida, use o mesmo comando acrescentando `--resume`. Os dois modelos são executados sequencialmente, gatos primeiro. Checkpoints, métricas e logs dessa etapa ficam em `outputs/do_zero_ampliado/`; os arquivos do primeiro experimento são preservados. O PID da fila está em `outputs/do_zero_ampliado/processo.pid`.

```powershell
Get-Content .\outputs\do_zero_ampliado\treinamento.log -Tail 20 -Wait
```

O teste original já foi consultado no diagnóstico anterior; serve para comparação de experimentos, mas não deve ser apresentado como uma avaliação final inteiramente independente após novas decisões. Uma confirmação de generalização requer novas fotos independentes. A aplicação web mantém os checkpoints anteriores até uma avaliação explícita da nova versão.

A revisão visual por amostragem encontrou dois exemplos inadequados (uma capa de livro e uma montagem com texto). Seus hashes e motivos estão em `datasets/ampliado/exclusoes_visuais.json`, respeitado pelo preparador em execuções futuras. As imagens brutas podem continuar no diretório de download ou na pasta extras, mas as excluídas não entram no manifesto de treinamento. A amostragem não equivale a uma revisão integral da base.


## Versão integrada após interrupção voluntária

A configuração atual de inferência usa `outputs/do_zero_ampliado/cachorro/best.pt`
e `outputs/do_zero_ampliado/gato/best.pt`. As descrições anteriores deste documento
registram o histórico dos experimentos; os caminhos `treino_proprio` e `treino_gatos`
são da versão anterior. Os classificadores de raça ativos têm origem no treino do
zero e foram continuados com dados adicionais. O filtro geral ImageNet e o detector
COCO continuam pré-treinados.

O treinamento canino ampliado foi interrompido por solicitação do usuário, com
21 épocas completas e a 22ª incompleta. O checkpoint `best.pt` é escolhido pela
menor perda de validação. A avaliação desse checkpoint não exige continuar o treino:

```powershell
.\.venv\Scripts\python.exe -u avaliar_checkpoint.py --animal cachorro
```

As métricas correspondentes ficam na pasta de cada checkpoint e são validadas
por SHA-256. A identificação da Paula e suas fases foram retiradas da API e da interface
por solicitação do usuário. Fotos, pesos, scripts e documentação histórica foram
preservados; as seções anteriores sobre esse experimento não descrevem um recurso
ativo. O retreino com a nova base apresentou regressão; seu relatório foi preservado
em `outputs/paula/tentativa_nova_base_metrics.json`.

Para executar a versão atual:

```powershell
.\.venv\Scripts\python.exe app.py
```

Abra http://127.0.0.1:5000. Reinicie um servidor que já estivesse aberto antes da
integração. Consulte `ROTEIRO_APRESENTACAO.txt` e `RELATORIO_INTEGRACAO.txt` para
a apresentação e os resultados verificados. O teste original já foi consultado e
não representa uma avaliação final independente após decisões entre experimentos.


Validação da integração: 45 testes sem falhas (3 testes de integração histórica
da Paula ignorados); sintaxe JavaScript validada. Acurácia no teste: cães 48,37%,
gatos 64,84%. Relatórios conferidos com SHA-256 dos checkpoints ativos.
Consulte `RELATORIO_INTEGRACAO.txt` para escopo e limitações.
