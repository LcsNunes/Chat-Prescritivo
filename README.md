# Chat Prescritivo

Projeto simples de manutenção prescritiva com IA Generativa, RAG real e modelos locais via Ollama.

O foco desta versão é demonstrar entendimento de arquitetura de IA, embeddings, recuperação semântica, prompt engineering, guardrails e fluxo de decisão. A interface web é propositalmente simples e fica separada do backend.

## Objetivo

Analisar eventos de manutenção vindos do `data/banner.csv` ou do PostgreSQL, identificar a falha informada pelo operador, buscar eventos históricos similares, recuperar documentos técnicos relacionados e gerar uma resposta prescritiva com uma LLM local.

Se a falha não possuir documentação suficiente, o sistema bloqueia a prescrição e recomenda cadastrar um novo procedimento técnico.

## Como Executar

Pré-requisitos:

- Python 3.10+
- Ollama rodando em `http://localhost:11434`
- PostgreSQL opcional para persistir novos eventos operacionais
- Tesseract OCR, caso deseje extrair texto de imagens dentro dos PDFs
- Modelos locais:
  - `qwen3:8b`
  - `qwen3-embedding:4b`

Instalação:

```bash
pip install -r requirements.txt
```

Execução:

```bash
uvicorn src.app:app --reload
```

PostgreSQL local opcional:

```bash
docker compose up -d postgres
```

Para ativar o banco como store de eventos, configure:

```env
DATABASE_URL=postgresql://chat_prescritivo:chat_prescritivo@localhost:5432/chat_prescritivo
EVENTS_TABLE=maintenance_events
POSTGRES_SEED_FROM_CSV=true
```

Quando `DATABASE_URL` não é informado, o sistema continua usando `data/banner.csv`.

Acesse:

- Frontend demo: `http://127.0.0.1:8000`
- Swagger: `http://127.0.0.1:8000/docs`
- Health: `http://127.0.0.1:8000/health`

Exemplo de análise por evento:

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

## Organização Frontend/Backend

Backend:

- `src/app.py`: API FastAPI.
- `src/rag.py`: embeddings, índice vetorial, busca e chamada da LLM.
- `src/chunking.py`: extração e chunking de PDFs.
- `src/fault_mapping.py`: normalização de `fault`, mapeamento semântico e busca de eventos similares.
- `src/event_store.py`: persistência opcional de eventos em PostgreSQL.
- `src/guardrails.py`: regras anti-alucinação.
- `src/prompts.py`: prompts da análise e do chat.

Frontend:

- `frontend/index.html`
- `frontend/styles.css`
- `frontend/app.js`

O backend serve os arquivos estáticos em `/assets`, mas o código da interface não fica misturado ao código Python.

## Fluxo de IA

1. Carrega o evento do CSV, do PostgreSQL ou recebe um JSON pela API.
2. Preserva `fault_raw` e cria `fault_normalized`.
3. Mapeia a falha normalizada para uma classe canônica por similaridade semântica.
4. Busca eventos históricos similares usando variáveis numéricas normalizadas.
5. Extrai textos dos PDFs e divide em chunks com metadados.
6. Gera embeddings reais dos chunks via Ollama.
7. Recupera os chunks mais relevantes por similaridade cosseno.
8. Aplica guardrails de cobertura documental e confiança da recuperação.
9. Quando aprovado, monta prompt com evento, histórico e chunks.
10. Chama a LLM local para sintetizar a resposta final.

## RAG e Embeddings

O RAG está implementado em `src/chunking.py` e `src/rag.py`.

- `chunking.py` extrai texto dos PDFs com `pdfplumber`.
- PDFs sem texto extraível ou páginas com imagens podem usar OCR com `pytesseract` quando `ENABLE_OCR=true`.
- Quando uma página possui texto embutido e também imagem com texto, o sistema combina o texto do PDF com linhas adicionais recuperadas por OCR, evitando duplicações óbvias.
- Cada chunk guarda `document`, `page`, `chunk_index`, `chunk_id`, texto e método de extração.
- `rag.py` chama o endpoint real do Ollama `/api/embed`.
- O índice vetorial é local, simples e salvo em `cache/`.
- A busca usa NumPy e similaridade cosseno.

Não há hashing, TF-IDF ou embeddings simulados.

### OCR

O OCR é controlado por variáveis de ambiente:

```env
ENABLE_OCR=true
OCR_STRATEGY=auto
OCR_LANG=por+eng
TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe
```

No Windows, o projeto tenta detectar automaticamente `C:\Program Files\Tesseract-OCR\tesseract.exe`. Configure `TESSERACT_CMD` somente se o executável estiver em outro caminho.

Estratégias:

- `auto`: aplica OCR quando a página não tem texto suficiente ou quando possui imagens.
- `missing_text`: aplica OCR somente quando a página quase não tem texto extraível.
- `always`: aplica OCR em todas as páginas.

No Windows, instale o Tesseract OCR se `GET /health` retornar `ocr.available=false`.

O endpoint `GET /health` mostra `ocr.available`. O endpoint `GET /documents` mostra páginas com imagem, páginas com OCR e páginas em que o OCR está indisponível.

## Tratamento da Coluna `fault`

A coluna `fault` vem de escrita humana, não diretamente do sensor. Por isso existe uma etapa de limpeza antes do mapeamento semântico.

O sistema preserva:

- `fault_raw`: valor original do CSV ou JSON.
- `fault_normalized`: valor limpo para análise.

Exemplos:

- `normal_2`, `normal_6`, `new_normal_0`, `normal_novo_teste` -> `normal`
- `mortor_desligado_novo` -> `motor_desligado`
- `desbanlanceado_carga_3_2` -> `desbalanceado`
- `cockecocked_adxl_0` -> `cocked_rotor`
- `new_falta_fase_0` -> `falta_fase`

A limpeza remove ruído operacional como `new`, números, `novo`, `carga`, `adxl` e corrige erros de digitação comuns. A classe técnica final continua sendo escolhida por embeddings contra descrições canônicas.

Classes canônicas principais:

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

## Persistência de Eventos

O `data/banner.csv` continua sendo a base histórica inicial e o fallback mais simples para a demo.

Quando `DATABASE_URL` está configurado, os endpoints de eventos usam PostgreSQL:

- `GET /events/{event_id}` busca no banco;
- `POST /events` cria ou atualiza o evento pelo `id`;
- `POST /analyze` usa o banco para buscar evento por `id`;
- `GET /sample-events` lista eventos carregados do banco.

A tabela `maintenance_events` é criada automaticamente na primeira consulta. O evento completo é salvo em `payload JSONB`, enquanto os campos principais ficam normalizados:

- `event_id`;
- `created_at`;
- `fault`;
- `fault_normalized`;
- `fault_is_operational_state`;
- `updated_at`.

Se `POSTGRES_SEED_FROM_CSV=true`, o banco é carregado com os registros do `banner.csv` quando a tabela ainda está vazia.

## Eventos Históricos Similares

A busca de similares usa colunas numéricas da base ativa de eventos, seja PostgreSQL ou `banner.csv`, como vibração, aceleração, temperatura e RPM.

Processo:

- converte colunas numéricas;
- preenche ausências com média;
- padroniza por média e desvio padrão;
- calcula distância euclidiana;
- retorna vizinhos mais próximos, período, falhas mais comuns e exemplos.

Isso permite explicar se o novo evento se parece com ocorrências anteriores.

## Guardrails

Os guardrails ficam em `src/guardrails.py`.

A LLM só é chamada quando:

- a classe não é estado operacional;
- existe documento relacionado;
- foram recuperados chunks do documento relacionado;
- a similaridade do melhor chunk passa do threshold configurado.

Se qualquer regra falhar, o sistema retorna uma resposta segura:

- não gera procedimento técnico inventado;
- informa que falta documentação;
- recomenda cadastrar um novo documento técnico.

## Prompt Engineering

Os prompts ficam em `src/prompts.py`.

O system prompt define o assistente como especialista em manutenção prescritiva e exige:

- responder somente com base nos documentos recuperados;
- citar documentos e trechos usados;
- separar diagnóstico de ação corretiva;
- declarar incerteza;
- não inventar ferramentas, causas ou critérios.

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
- `POST /events`
- `GET /sample-events?fault=cocked_rotor&limit=5`
- `POST /chat`
- `POST /analyze`

### `POST /chat`

Endpoint para perguntas em linguagem natural. Ele tenta mapear semanticamente a pergunta para uma classe canônica, recupera documentos relacionados e responde somente com base nos chunks.

Exemplos de perguntas:

- `rotor inclinado tem procedimento?`
- `qual procedimento para rolamento?`
- `falta de fase tem documento?`
- `normal_6 precisa de manutenção?`

### `POST /documents`

Adiciona um novo manual PDF na base documental.

Formato: `multipart/form-data`, campo `file`.

Parametro opcional:

- `overwrite=true`: substitui um PDF existente com o mesmo nome.

Quando um documento é adicionado, o cache em memória de chunks e índice vetorial é invalidado. O índice será recriado na próxima consulta.

### `DELETE /documents/{filename}`

Remove um manual PDF obsoleto da base documental e invalida o índice RAG em memória.

### `POST /events`

Registra ou atualiza um evento na base ativa.

Formato:

```json
{
  "event": {
    "id": 114387,
    "fault": "cocked_rotor_2",
    "rpm": 1000.0
  }
}
```

Regras:

- Se `DATABASE_URL` estiver configurado, grava no PostgreSQL.
- Se `DATABASE_URL` não estiver configurado, grava no `data/banner.csv`.
- Se `id` existir, atualiza o evento existente.
- Se `id` não existir, cria um novo registro com esse `id`.
- Se `id` não for enviado, cria um novo registro com o próximo id numérico.
- Campos derivados como `fault_normalized` não são aceitos no payload de entrada; eles são recalculados pelo backend.
- No PostgreSQL, o evento completo fica em `payload JSONB` e os campos normalizados ficam em colunas próprias.
- No fallback CSV, campos que não existem no arquivo são ignorados e retornados em `ignored_fields`.
- No PostgreSQL, campos extras são preservados dentro de `payload JSONB`.

Resposta de `/analyze` inclui:

- evento analisado;
- mapeamento da falha;
- eventos similares;
- chunks recuperados;
- decisão dos guardrails;
- validação simples de citação;
- resposta final.

## Limitações

- `Doc1.pdf` e `Doc7.pdf` são PDFs de imagem. Sem OCR instalado, eles não entram no índice textual.
- O PostgreSQL é usado apenas para eventos; documentos, chunks e embeddings continuam em arquivos/cache local.
- O índice vetorial é local e simples, adequado para demo, não para escala industrial.
- Os thresholds foram calibrados para este conjunto de dados e devem ser avaliados em produção.
- A validação da resposta é simples; ela verifica citação de documentos, mas não avalia exatidão técnica automaticamente.
- O sistema não substitui validação de uma equipe de manutenção.

## Melhorias Futuras

- RAG híbrido com busca lexical e vetorial.
- Reranking dos chunks recuperados.
- Banco vetorial dedicado.
- Graph RAG para relacionar falhas, sintomas, ativos e procedimentos.
- LangChain ou LangGraph para orquestração com estados.
- Avaliação automática de respostas.
- OCR robusto e pipeline de qualidade documental.
- Integração com banco industrial real.
- Monitoramento de respostas, latência e cobertura documental em produção.
