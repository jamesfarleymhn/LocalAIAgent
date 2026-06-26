from __future__ import annotations

import logging
import os
import warnings
from typing import Any

# Quiet Hugging Face / transformer startup noise in the app.
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
warnings.filterwarnings("ignore")
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings


DB_DIR = "./chroma_denials_db"
COLLECTION_NAME = "denials_knowledge_base"
EMBEDDING_MODEL = "bge-m3"
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"

# If this is True, the app will try to use the reranker only when a RAG/policy
# question actually needs retrieval. If the reranker is unavailable, the app
# automatically falls back to normal Chroma similarity retrieval instead of
# crashing at startup.
USE_RERANKER = True

embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

vector_store = Chroma(
    collection_name=COLLECTION_NAME,
    persist_directory=DB_DIR,
    embedding_function=embeddings,
)

_reranker: Any | None = None
_reranker_failed = False
_reranker_warning_printed = False


def _warn_reranker_unavailable(error: Exception) -> None:
    """Print one concise warning instead of failing the whole app."""
    global _reranker_warning_printed

    if _reranker_warning_printed:
        return

    _reranker_warning_printed = True
    print(
        "\nWARNING: The Hugging Face reranker could not be loaded. "
        "The app will continue using normal vector similarity retrieval.\n"
        f"Reranker error: {type(error).__name__}: {error}\n"
    )


def get_reranker():
    """
    Lazy-load the reranker.

    Important: do NOT create HuggingFaceCrossEncoder at import time. If the model
    is not cached locally or the machine has restricted internet, startup should
    still work for case-only denial summaries.
    """
    global _reranker, _reranker_failed

    if _reranker_failed:
        return None

    if _reranker is not None:
        return _reranker

    try:
        from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
        from langchain_community.cross_encoders import HuggingFaceCrossEncoder

        cross_encoder = HuggingFaceCrossEncoder(model_name=RERANKER_MODEL)
        _reranker = CrossEncoderReranker(
            model=cross_encoder,
            top_n=10,
        )
        return _reranker

    except Exception as e:
        _reranker_failed = True
        _warn_reranker_unavailable(e)
        return None


def get_retriever(filter_dict: dict | None = None, k: int = 30, rerank_top_n: int = 10, use_reranker: bool = USE_RERANKER):
    search_kwargs = {"k": k}

    if filter_dict:
        search_kwargs["filter"] = filter_dict

    base_retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs=search_kwargs,
    )

    if not use_reranker:
        return base_retriever

    reranker = get_reranker()

    if reranker is None:
        return base_retriever

    try:
        from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever

        # top_n is set when the reranker is created. If you want different top_n
        # per retriever, create separate rerankers; for now keep it simple/stable.
        return ContextualCompressionRetriever(
            base_retriever=base_retriever,
            base_compressor=reranker,
        )
    except Exception as e:
        _warn_reranker_unavailable(e)
        return base_retriever


def _safe_invoke(retriever, question: str) -> list:
    try:
        return list(retriever.invoke(question))
    except Exception as e:
        print(f"WARNING: Retrieval failed for one knowledge-base category: {type(e).__name__}: {e}")
        return []


def dedupe_retrieved_docs(docs: list) -> list:
    seen = set()
    unique_docs = []

    for doc in docs:
        metadata = getattr(doc, "metadata", {}) or {}
        key = (
            metadata.get("source"),
            metadata.get("file_name"),
            metadata.get("page"),
            metadata.get("sheet_name"),
            metadata.get("row_number"),
            metadata.get("chunk_index"),
            getattr(doc, "page_content", "")[:250],
        )

        if key in seen:
            continue

        seen.add(key)
        unique_docs.append(doc)

    return unique_docs


def retrieve_supporting_knowledge(question: str) -> list:
    """
    Retrieve general supporting knowledge only.

    Patient-specific facts must come from the user-submitted case file, not from
    the knowledge base. This intentionally avoids patient-specific folders such
    as denial_letters, submitted_cases, uploaded_cases, and appeal_letters.
    """
    filter_dicts = [
        {"doc_category": "payer_policies"},
        {"doc_category": "guides"},
        {"doc_category": "coding_guidelines"},
        {"doc_category": "clinical_guidelines"},
        {"doc_category": "cdi_guides"},
        {"doc_category": "reference_materials"},
        {"file_type": "excel"},
    ]

    docs = []

    for filter_dict in filter_dicts:
        retriever = get_retriever(filter_dict=filter_dict, k=25, rerank_top_n=5)
        docs.extend(_safe_invoke(retriever, question))

    # If the category names in the vector DB do not match the newer safe names,
    # do one broad fallback. The answer prompt should still use these only for
    # policy/guideline/deadline support, never for patient-specific facts.
    if not docs:
        retriever = get_retriever(filter_dict=None, k=12, rerank_top_n=5)
        docs.extend(_safe_invoke(retriever, question))

    return dedupe_retrieved_docs(docs)


# Backward-compatible alias for older main.py versions.
def retrieve_balanced(question: str) -> list:
    return retrieve_supporting_knowledge(question)
