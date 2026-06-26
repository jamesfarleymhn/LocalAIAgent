from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AppConfig:
    """Configuration for the local denial-document RAG workflow.

    The scripts intentionally do not contain patient examples, real identifiers,
    or embedded case content. Submitted documents are read at runtime only.
    """

    case_chunk_chars: int = 9000
    case_chunk_overlap: int = 700
    min_pdf_page_text_chars: int = 80
    ollama_url: str = "http://localhost:11434"
    generation_model: str = "llama3.1:latest"
    embedding_model: str = "bge-m3"
    knowledge_base_dir: Path = Path("knowledge_base")
    vector_db_dir: Path = Path("chroma_denials_db")
    collection_name: str = "denials_knowledge_base"
    ocr_zoom: float = 2.5


CONFIG = AppConfig()

# Folder names that are never allowed to be ingested into the reusable RAG store.
# Patient/case documents must be analyzed at runtime only, not embedded as reusable evidence.
PROHIBITED_KB_FOLDER_NAMES = {
    "appeal_letters",
    "case_files",
    "denial_letters",
    "patient_files",
    "submitted_cases",
    "submitted_denials",
    "uploaded_cases",
    "user_input",
}
