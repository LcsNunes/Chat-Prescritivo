from __future__ import annotations

from typing import Any

from src.fault_mapping import FaultMappingResult


def has_document_coverage(
    fault_mapping: FaultMappingResult,
    retrieved_chunks: list[dict[str, Any]],
) -> bool:
    if not fault_mapping.has_documentation or not fault_mapping.related_documents:
        return False

    related = set(fault_mapping.related_documents)
    return any(chunk.get("document") in related for chunk in retrieved_chunks)


def is_retrieval_confident(
    retrieved_chunks: list[dict[str, Any]],
    min_score: float,
) -> bool:
    if not retrieved_chunks:
        return False
    return max(float(chunk.get("score", 0.0)) for chunk in retrieved_chunks) >= min_score


def evaluate_guardrails(
    fault_mapping: FaultMappingResult,
    retrieved_chunks: list[dict[str, Any]],
    min_chunk_score: float,
) -> dict[str, Any]:
    """Return a clear decision about whether the LLM may prescribe actions."""
    if fault_mapping.is_operational_state:
        return {
            "allowed": False,
            "reason": "operational_state",
            "message": "O evento foi classificado como estado operacional, nao como falha.",
        }

    if not fault_mapping.has_documentation:
        return {
            "allowed": False,
            "reason": "missing_documentation",
            "message": "Nao existe documento tecnico cadastrado para a classe de falha identificada.",
        }

    if not has_document_coverage(fault_mapping, retrieved_chunks):
        return {
            "allowed": False,
            "reason": "missing_related_chunks",
            "message": "Nao foram recuperados chunks do documento relacionado a falha.",
        }

    if not is_retrieval_confident(retrieved_chunks, min_chunk_score):
        return {
            "allowed": False,
            "reason": "low_retrieval_score",
            "message": "A similaridade dos chunks recuperados ficou abaixo do minimo configurado.",
        }

    return {
        "allowed": True,
        "reason": "approved",
        "message": "Ha cobertura documental suficiente para sintetizar a resposta com LLM.",
    }


def build_undocumented_response(
    fault_mapping: FaultMappingResult,
    guardrail_decision: dict[str, Any],
) -> str:
    if guardrail_decision.get("reason") == "operational_state":
        return (
            "Tipo de evento identificado: Estado operacional sem falha\n\n"
            "O registro foi classificado como uma condicao operacional, nao como um defeito de manutencao.\n\n"
            "Por seguranca, o sistema nao ira gerar um procedimento corretivo para este evento.\n\n"
            "Recomendacao:\n"
            "- Verificar se o estado operacional esta coerente com o contexto da maquina.\n"
            "- Caso haja sintomas reais de falha, registrar um novo evento com descricao tecnica mais especifica."
        )

    return (
        "Nao foi encontrada documentacao tecnica suficiente para orientar a correcao deste tipo de falha.\n\n"
        "Por seguranca, o sistema nao ira gerar um procedimento prescritivo sem base documental.\n\n"
        f"Falha identificada: {fault_mapping.display_name}\n"
        f"Classe normalizada: {fault_mapping.fault_normalized}\n"
        f"Motivo do bloqueio: {guardrail_decision.get('message')}\n\n"
        "Recomendacao:\n"
        "Cadastrar um novo documento orientativo de manutencao para essa falha, contendo sintomas, "
        "diagnostico, ferramentas necessarias, procedimento de correcao, criterios de aceitacao e "
        "recomendacoes preventivas."
    )


def validate_llm_answer(answer: str, retrieved_chunks: list[dict[str, Any]]) -> dict[str, Any]:
    """Simple post-check to keep source citation visible in the final answer."""
    documents = sorted({chunk["document"] for chunk in retrieved_chunks})
    cited_documents = [document for document in documents if document in answer]
    return {
        "ok": bool(answer.strip()) and bool(cited_documents),
        "documents": documents,
        "cited_documents": cited_documents,
    }

