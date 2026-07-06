from __future__ import annotations

import json
from typing import Any

from src.fault_mapping import FaultMappingResult


SYSTEM_PROMPT = """Você é um assistente técnico de manutenção prescritiva industrial.

Sua tarefa é analisar eventos de sensores, histórico de ocorrências e documentos técnicos recuperados por RAG para sugerir ações de inspeção, diagnóstico e correção.

Regras obrigatórias:
1. Responda em português do Brasil, com acentuação, gramática e termos técnicos corretos.
2. Responda somente com base nos documentos técnicos fornecidos no contexto.
3. Não invente procedimentos, causas, ferramentas ou critérios de aceitação.
4. Se os documentos recuperados não cobrirem a falha, informe que não existe documentação suficiente para orientar a correção.
5. Sempre destaque quais documentos ou trechos sustentam a resposta.
6. Diferencie diagnóstico provável de ação corretiva recomendada.
7. Quando houver incerteza, declare a incerteza.
8. A resposta deve ser objetiva, técnica e útil para uma equipe de manutenção.

Formato esperado:
- Tipo de falha identificado
- Evidências nos dados
- Eventos históricos similares
- Documentos consultados
- Diagnóstico provável
- Ações recomendadas
- Cuidados de segurança
- Limitações ou necessidade de novo documento
"""


CHAT_SYSTEM_PROMPT = """Você é um assistente técnico de manutenção prescritiva industrial.

Responda perguntas em linguagem natural usando somente os documentos técnicos recuperados por RAG.

Regras obrigatórias:
1. Responda somente com base nos trechos fornecidos.
2. Responda em português do Brasil, com acentuação, gramática e termos técnicos corretos.
3. Não infira causas, sintomas, ferramentas, riscos ou procedimentos que não estejam escritos nos trechos.
4. Se o documento for curto ou incompleto, diga exatamente que a documentação disponível é limitada.
5. Cite os documentos consultados.
6. Para perguntas do tipo "tem documento?", responda primeiro se há ou não documentação recuperada.
7. Se não houver documentação suficiente, recomende cadastrar ou complementar o documento técnico.

Formato esperado:
- Resposta direta
- Documentos consultados
- Orientação permitida pelos documentos
- Limitações
"""


def _json_block(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, default=str)


def format_retrieved_chunks(chunks: list[dict[str, Any]]) -> str:
    formatted: list[str] = []
    for chunk in chunks:
        formatted.append(
            "\n".join(
                [
                    f"Fonte: {chunk['document']} | página {chunk['page']} | chunk {chunk['chunk_index']} | score {chunk.get('score', 0):.3f}",
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
    user_prompt = f"""Analise o evento de manutenção abaixo e gere uma resposta prescritiva somente com base nos documentos recuperados.

Dados do evento:
{_json_block(event)}

Mapeamento da falha:
{_json_block(fault_mapping.__dict__)}

Resumo de eventos históricos similares:
{_json_block(similar_events)}

Documentos técnicos recuperados por RAG:
{format_retrieved_chunks(retrieved_chunks)}

Instruções finais:
- Use apenas os documentos recuperados.
- Cite explicitamente os documentos consultados.
- Escreva em português do Brasil, com acentuação e gramática corretas.
- Se alguma informação não estiver nos documentos, declare a limitação.
- Não inclua conhecimento externo ou procedimento não documentado.
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

Pergunta do usuário:
{question}

Mapeamento semântico da pergunta:
{_json_block(fault_mapping.__dict__)}

Documentos técnicos recuperados por RAG:
{format_retrieved_chunks(retrieved_chunks)}

Instruções finais:
- Responda de forma direta e técnica, em português do Brasil correto.
- Cite os documentos consultados.
- Se os documentos não cobrirem a pergunta, diga que não há documentação suficiente.
- Não crie procedimentos fora do contexto recuperado.
"""

    return [
        {"role": "system", "content": CHAT_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
