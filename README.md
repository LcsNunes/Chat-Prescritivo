# Chat Prescritivo

Projeto simples de manutencao prescritiva com IA Generativa, RAG real e modelos locais via Ollama.

O foco desta versao e demonstrar entendimento de arquitetura de IA, embeddings, recuperacao semantica, prompt engineering, guardrails e fluxo de decisao. A interface web e propositalmente simples e fica separada do backend.

## Objetivo

Analisar eventos de manutencao vindos do `data/banner.csv`, identificar a falha informada pelo operador, buscar eventos historicos similares, recuperar documentos tecnicos relacionados e gerar uma resposta prescritiva com uma LLM local.

Se a falha nao possuir documentacao suficiente, o sistema bloqueia a prescricao e recomenda cadastrar um novo procedimento tecnico.

## Como Executar

Pre-requisitos:

- Python 3.10+
- Ollama rodando em `http://localhost:11434`
- Tesseract OCR, caso deseje extrair texto de imagens dentro dos PDFs
- Modelos locais:
  - `qwen3:8b`
  - `qwen3-embedding:4b`

Instalacao:

```bash
pip install -r requirements.txt
```

Execucao:

```bash
uvicorn src.app:app --reload
```

Acesse:

- Frontend demo: `http://127.0.0.1:8000`
- Swagger: `http://127.0.0.1:8000/docs`
- Health: `http://127.0.0.1:8000/health`

Exemplo de analise por evento:

```json
{
  "event_id": 114387,
  "top_k_chunks": 3,
  "similar_events_limit": 3
}
```

Exemplo de chat em linguagem natural:

```json
{
  "question": "falta de fase tem documento?",
  "top_k_chunks": 3
}
```

## Organizacao Frontend/Backend

Backend:

- `src/app.py`: API FastAPI.
- `src/rag.py`: embeddings, indice vetorial, busca e chamada da LLM.
- `src/chunking.py`: extracao e chunking de PDFs.
- `src/fault_mapping.py`: normalizacao de `fault`, mapeamento semantico e busca de eventos similares.
- `src/guardrails.py`: regras anti-alucinacao.
- `src/prompts.py`: prompts da analise e do chat.

Frontend:

- `frontend/index.html`
- `frontend/styles.css`
- `frontend/app.js`

O backend serve os arquivos estaticos em `/assets`, mas o codigo da interface nao fica misturado ao codigo Python.

## Fluxo de IA

1. Carrega o evento do CSV ou recebe um JSON pela API.
2. Preserva `fault_raw` e cria `fault_normalized`.
3. Mapeia a falha normalizada para uma classe canonica por similaridade semantica.
4. Busca eventos historicos similares usando variaveis numericas normalizadas.
5. Extrai textos dos PDFs e divide em chunks com metadados.
6. Gera embeddings reais dos chunks via Ollama.
7. Recupera os chunks mais relevantes por similaridade cosseno.
8. Aplica guardrails de cobertura documental e confianca da recuperacao.
9. Quando aprovado, monta prompt com evento, historico e chunks.
10. Chama a LLM local para sintetizar a resposta final.

## RAG e Embeddings

O RAG esta implementado em `src/chunking.py` e `src/rag.py`.

- `chunking.py` extrai texto dos PDFs com `pdfplumber`.
- PDFs sem texto extraivel ou paginas com imagens podem usar OCR com `pytesseract` quando `ENABLE_OCR=true`.
- Quando uma pagina possui texto embutido e tambem imagem com texto, o sistema combina o texto do PDF com linhas adicionais recuperadas por OCR, evitando duplicacoes obvias.
- Cada chunk guarda `document`, `page`, `chunk_index`, `chunk_id`, texto e metodo de extracao.
- `rag.py` chama o endpoint real do Ollama `/api/embed`.
- O indice vetorial e local, simples e salvo em `cache/`.
- A busca usa NumPy e similaridade cosseno.

Nao ha hashing, TF-IDF ou embeddings simulados.

### OCR

O OCR e controlado por variaveis de ambiente:

```env
ENABLE_OCR=true
OCR_STRATEGY=auto
OCR_LANG=por+eng
TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe
```

Estrategias:

- `auto`: aplica OCR quando a pagina nao tem texto suficiente ou quando possui imagens.
- `missing_text`: aplica OCR somente quando a pagina quase nao tem texto extraivel.
- `always`: aplica OCR em todas as paginas.

No Windows, instale o Tesseract OCR e configure `TESSERACT_CMD` se o executavel nao estiver no `PATH`.

O endpoint `GET /health` mostra `ocr.available`. O endpoint `GET /documents` mostra paginas com imagem, paginas com OCR e paginas em que o OCR esta indisponivel.

## Tratamento da Coluna `fault`

A coluna `fault` vem de escrita humana, nao diretamente do sensor. Por isso existe uma etapa de limpeza antes do mapeamento semantico.

O sistema preserva:

- `fault_raw`: valor original do CSV ou JSON.
- `fault_normalized`: valor limpo para analise.

Exemplos:

- `normal_2`, `normal_6`, `new_normal_0`, `normal_novo_teste` -> `normal`
- `mortor_desligado_novo` -> `motor_desligado`
- `desbanlanceado_carga_3_2` -> `desbalanceado`
- `cockecocked_adxl_0` -> `cocked_rotor`
- `new_falta_fase_0` -> `falta_fase`

A limpeza remove ruido operacional como `new`, numeros, `novo`, `carga`, `adxl` e corrige typos comuns. A classe tecnica final continua sendo escolhida por embeddings contra descricoes canonicas.

Classes canonicas principais:

- `bearing_fault`
- `misalignment`
- `unbalance`
- `belt_fault`
- `pulley_fault`
- `cocked_rotor`
- `fan_fault`
- `phase_loss`
- `undocumented_eccentric_rotor`
- `operational_state`

## Eventos Historicos Similares

A busca de similares usa colunas numericas do `banner.csv`, como vibracao, aceleracao, temperatura e RPM.

Processo:

- converte colunas numericas;
- preenche ausencias com media;
- padroniza por media e desvio padrao;
- calcula distancia euclidiana;
- retorna vizinhos mais proximos, periodo, falhas mais comuns e exemplos.

Isso permite explicar se o novo evento se parece com ocorrencias anteriores.

## Guardrails

Os guardrails ficam em `src/guardrails.py`.

A LLM so e chamada quando:

- a classe nao e estado operacional;
- existe documento relacionado;
- foram recuperados chunks do documento relacionado;
- a similaridade do melhor chunk passa do threshold configurado.

Se qualquer regra falhar, o sistema retorna uma resposta segura:

- nao gera procedimento tecnico inventado;
- informa que falta documentacao;
- recomenda cadastrar um novo documento tecnico.

## Prompt Engineering

Os prompts ficam em `src/prompts.py`.

O system prompt define o assistente como especialista em manutencao prescritiva e exige:

- responder somente com base nos documentos recuperados;
- citar documentos e trechos usados;
- separar diagnostico de acao corretiva;
- declarar incerteza;
- nao inventar ferramentas, causas ou criterios.

A chamada da LLM usa:

```python
temperature = 0.1
top_p = 0.8
num_ctx = 8192
```

## API

Endpoints principais:

- `GET /health`
- `GET /document-report`
- `GET /documents`
- `POST /documents`
- `DELETE /documents/{filename}`
- `GET /events/{event_id}`
- `GET /sample-events?fault=cocked_rotor&limit=5`
- `POST /chat`
- `POST /analyze`

### `POST /chat`

Endpoint para perguntas em linguagem natural. Ele tenta mapear semanticamente a pergunta para uma classe canonica, recupera documentos relacionados e responde somente com base nos chunks.

Exemplos de perguntas:

- `rotor inclinado tem procedimento?`
- `qual procedimento para rolamento?`
- `falta de fase tem documento?`
- `normal_6 precisa de manutencao?`

### `POST /documents`

Adiciona um novo manual PDF na base documental.

Formato: `multipart/form-data`, campo `file`.

Parametro opcional:

- `overwrite=true`: substitui um PDF existente com o mesmo nome.

Quando um documento e adicionado, o cache em memoria de chunks e indice vetorial e invalidado. O indice sera recriado na proxima consulta.

### `DELETE /documents/{filename}`

Remove um manual PDF obsoleto da base documental e invalida o indice RAG em memoria.

Resposta de `/analyze` inclui:

- evento analisado;
- mapeamento da falha;
- eventos similares;
- chunks recuperados;
- decisao dos guardrails;
- validacao simples de citacao;
- resposta final.

## Limitacoes

- `Doc1.pdf` e `Doc7.pdf` sao PDFs de imagem. Sem OCR instalado, eles nao entram no indice textual.
- O indice vetorial e local e simples, adequado para demo, nao para escala industrial.
- Os thresholds foram calibrados para este conjunto de dados e devem ser avaliados em producao.
- A validacao da resposta e simples; ela verifica citacao de documentos, mas nao avalia exatidao tecnica automaticamente.
- O sistema nao substitui validacao de uma equipe de manutencao.

## Melhorias Futuras

- RAG hibrido com busca lexical e vetorial.
- Reranking dos chunks recuperados.
- Banco vetorial dedicado.
- Graph RAG para relacionar falhas, sintomas, ativos e procedimentos.
- LangChain ou LangGraph para orquestracao com estados.
- Avaliacao automatica de respostas.
- OCR robusto e pipeline de qualidade documental.
- Integracao com banco industrial real.
- Monitoramento de respostas, latencia e cobertura documental em producao.
