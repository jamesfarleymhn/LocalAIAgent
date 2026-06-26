from __future__ import annotations

import hashlib
import re
from pathlib import Path

from config import PROHIBITED_KB_FOLDER_NAMES


DATE_RE = re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b")
LONG_NUMBER_RE = re.compile(r"\b\d{6,}\b")
MEMBERISH_RE = re.compile(r"\b[A-Z]{1,4}\d{5,}[A-Z0-9]*\b", re.IGNORECASE)


def stable_file_id(path: Path) -> str:
    """Return a stable non-PHI source id without embedding the file name in code or logs."""
    stat = path.stat()
    raw = f"{path.resolve()}|{stat.st_size}|{int(stat.st_mtime)}".encode("utf-8", errors="ignore")
    return hashlib.sha256(raw).hexdigest()[:16]


def redact_identifiers(text: str) -> str:
    """Best-effort redaction for using a user question as a knowledge-base search query."""
    text = DATE_RE.sub("[DATE]", text or "")
    text = MEMBERISH_RE.sub("[IDENTIFIER]", text)
    text = LONG_NUMBER_RE.sub("[NUMBER]", text)
    return text


def is_prohibited_knowledge_path(path: Path, kb_root: Path) -> bool:
    try:
        parts = [part.lower() for part in path.relative_to(kb_root).parts]
    except ValueError:
        parts = [part.lower() for part in path.parts]
    return any(part in PROHIBITED_KB_FOLDER_NAMES for part in parts)


def validate_knowledge_path(path: Path, kb_root: Path) -> None:
    if is_prohibited_knowledge_path(path, kb_root):
        raise ValueError(
            "This path looks like patient-specific case material and must not be ingested into the reusable RAG store. "
            "Analyze it at runtime with main.py instead."
        )
