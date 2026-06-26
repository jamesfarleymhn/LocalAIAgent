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

# Folders that should never be ingested into the reusable vector database.
# These are for live/user-submitted patient cases, not reusable knowledge.
ALWAYS_BLOCKED_CASE_FOLDER_NAMES = {
    "case_files",
    "patient_files",
    "submitted_cases",
    "submitted_denials",
    "uploaded_cases",
    "user_input",
    "raw_patient_cases",
    "production_cases",
    "live_cases",
}

# Folders that may be useful for appeal drafting, but only as de-identified / sanitized knowledge.
# The ingest pipeline sanitizes these before embedding. It does not store raw text in Chroma.
SANITIZE_REQUIRED_FOLDER_NAMES = {
    "appeal_letters",
    "appeal_examples",
    "appeal_samples",
    "case_studies",
    "denial_letters",
    "denial_examples",
    "sample_cases",
}

# Folder names that signal the contents were already intentionally prepared as reusable, non-PHI knowledge.
SAFE_KB_MARKERS = {
    "deidentified",
    "de-identified",
    "de_id",
    "deid",
    "redacted",
    "sanitized",
    "template",
    "templates",
    "public",
    "sample",
    "samples",
    "examples",
}
