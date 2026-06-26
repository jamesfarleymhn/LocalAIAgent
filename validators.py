from __future__ import annotations

import re
from typing import Any


def _compact(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip().lower()


def text_contains_loosely(needle: Any, haystack: str) -> bool:
    """Return True when a model value is plainly supported by source text.

    This is not the primary extraction mechanism. It is a validation layer used
    after the local model reads the chunk. It helps keep model-first extraction
    grounded without relying on regex to understand the denial document.
    """
    n = _compact(needle)
    h = _compact(haystack)
    if not n:
        return True
    if n in h:
        return True

    # Ignore punctuation differences for OCR/table text.
    n2 = re.sub(r"[^a-z0-9]+", " ", n).strip()
    h2 = re.sub(r"[^a-z0-9]+", " ", h).strip()
    if n2 and n2 in h2:
        return True

    # For longer facts, allow most meaningful tokens to be present.
    tokens = [token for token in n2.split() if len(token) >= 3]
    if len(tokens) >= 5:
        hits = sum(1 for token in tokens if token in h2)
        return hits / len(tokens) >= 0.75

    return False


def validate_llm_field(name: str, value: Any, evidence_excerpt: Any, source_text: str) -> tuple[bool, str]:
    """Validate a model-extracted field against the same source chunk."""
    if text_contains_loosely(evidence_excerpt, source_text):
        return True, "Model evidence excerpt was found in the source chunk."
    if text_contains_loosely(value, source_text):
        return True, "Model value was found in the source chunk."
    return False, "Model value/evidence was not directly found in the source chunk; review before relying on it."
