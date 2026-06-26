from __future__ import annotations

from pathlib import Path
from typing import Any

from config import CONFIG
from privacy import redact_identifiers


def _load_vector_store():
    """Lazy-load optional LangChain/Chroma dependencies only for RAG operations."""
    try:
        from langchain_chroma import Chroma
        from langchain_ollama import OllamaEmbeddings
    except ImportError as exc:
        raise RuntimeError(
            "RAG retrieval dependencies are missing. Install requirements.txt or run without --use-kb."
        ) from exc

    embeddings = OllamaEmbeddings(model=CONFIG.embedding_model)
    return Chroma(
        collection_name=CONFIG.collection_name,
        persist_directory=str(CONFIG.vector_db_dir),
        embedding_function=embeddings,
    )


def retrieve_supporting_knowledge(question: str, case_context: dict[str, Any] | None = None, *, k: int = 8) -> list[dict[str, Any]]:
    """Retrieve general knowledge-base evidence.

    This function should not be used to fill patient-specific case facts. It is
    only for policies, coding references, guidelines, and appeal support.
    """
    if not CONFIG.vector_db_dir.exists():
        return []

    vector_store = _load_vector_store()
    safe_question = redact_identifiers(question)
    docs = vector_store.similarity_search(safe_question, k=k)
    results: list[dict[str, Any]] = []
    seen = set()

    for doc in docs:
        metadata = getattr(doc, "metadata", {}) or {}
        key = (
            metadata.get("source_id") or metadata.get("source"),
            metadata.get("page_number") or metadata.get("page"),
            metadata.get("chunk_index"),
            getattr(doc, "page_content", "")[:200],
        )
        if key in seen:
            continue
        seen.add(key)
        results.append(
            {
                "text": getattr(doc, "page_content", ""),
                "metadata": metadata,
                "use_limit": "general knowledge only; do not copy patient-specific facts from this evidence",
            }
        )
    return results


# Backward-compatible alias for older imports.
def retrieve_balanced(question: str) -> list[dict[str, Any]]:
    return retrieve_supporting_knowledge(question)
