from __future__ import annotations

import json
from typing import Any

from src.fault_mapping import FaultMappingResult


SYSTEM_PROMPT = """Voce e um assistente tecnico de manutencao prescritiva industrial.

Sua tarefa e analisar eventos de sensores, historico de ocorrencias e documentos tecnicos recuperados por RAG para sugerir acoes de inspecao, diagnostico e correcao.

Regras obrigatorias:
1. Responda somente com base nos documentos tecnicos fornecidos no contexto.
2. Nao invente procedimentos, causas, ferramentas ou criterios de aceitacao.
3. Se os documentos recuperados nao cobrirem a falha, informe que nao existe documentacao suficiente para orientar a correcao.
4. Sempre destaque quais documentos ou trechos sustentam a resposta.
5. Diferencie diagnostico provavel de acao corretiva recomendada.
6. Quando houver incerteza, declare a incerteza.
7. A resposta deve ser objetiva, tecnica e util para uma equipe de manutencao.

Formato esperado:
- Tipo de falha identificado
- Evidencias nos dados
- Eventos historicos similares
- Documentos consultados
- Diagnostico provavel
- Acoes recomendadas
- Cuidados de seguranca
- Limitacoes ou necessidade de novo documento
"""


CHAT_SYSTEM_PROMPT = """Voce e um assistente tecnico de manutencao prescritiva industrial.

Responda perguntas em linguagem natural usando somente os documentos tecnicos recuperados por RAG.

Regras obrigatorias:
1. Responda somente com base nos trechos fornecidos.
2. Nao infira causas, sintomas, ferramentas, riscos ou procedimentos que nao estejam escritos nos trechos.
3. Se o documento for curto ou incompleto, diga exatamente que a documentacao disponivel e limitada.
4. Cite os documentos consultados.
5. Para perguntas do tipo "tem documento?", responda primeiro se ha ou nao documentacao recuperada.
6. Se nao houver documentacao suficiente, recomende cadastrar ou complementar o documento tecnico.

Formato esperado:
- Resposta direta
- Documentos consultados
- Orientacao permitida pelos documentos
- Limitacoes
"""


def _json_block(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, default=str)


def format_retrieved_chunks(chunks: list[dict[str, Any]]) -> str:
    formatted: list[str] = []
    for chunk in chunks:
        formatted.append(
            "\n".join(
                [
                    f"Fonte: {chunk['document']} | pagina {chunk['page']} | chunk {chunk['chunk_index']} | score {chunk.get('score', 0):.3f}",
                    chunk["text"],
                ]
            )
        )
    return "\n\n---\n\n".join(formatted)


def build_rag_messages(
    event: dict[str, Any],
    fault_mapping: FaultMappingResult,
    similar_events: dict[str, Any],
    retrieved_chunks: list[dict[str, Any]],
) -> list[dict[str, str]]:
    user_prompt = f"""Analise o evento de manutencao abaixo e gere uma resposta prescritiva somente com base nos documentos recuperados.

Dados do evento:
{_json_block(event)}

Mapeamento da falha:
{_json_block(fault_mapping.__dict__)}

Resumo de eventos historicos similares:
{_json_block(similar_events)}

Documentos tecnicos recuperados por RAG:
{format_retrieved_chunks(retrieved_chunks)}

Instrucoes finais:
- Use apenas os documentos recuperados.
- Cite explicitamente os documentos consultados.
- Se alguma informacao nao estiver nos documentos, declare a limitacao.
- Nao inclua conhecimento externo ou procedimento nao documentado.
"""

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def build_chat_messages(
    question: str,
    fault_mapping: FaultMappingResult,
    retrieved_chunks: list[dict[str, Any]],
) -> list[dict[str, str]]:
    user_prompt = f"""Responda a pergunta em linguagem natural usando somente os documentos recuperados.

Pergunta do usuario:
{question}

Mapeamento semantico da pergunta:
{_json_block(fault_mapping.__dict__)}

Documentos tecnicos recuperados por RAG:
{format_retrieved_chunks(retrieved_chunks)}

Instrucoes finais:
- Responda de forma direta e tecnica.
- Cite os documentos consultados.
- Se os documentos nao cobrirem a pergunta, diga que nao ha documentacao suficiente.
- Nao crie procedimentos fora do contexto recuperado.
"""

    return [
        {"role": "system", "content": CHAT_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
