from __future__ import annotations

from config import CONFIG
from schemas import LoadedCase, TextChunk


def chunk_loaded_case(
    loaded_case: LoadedCase,
    *,
    chunk_chars: int = CONFIG.case_chunk_chars,
    overlap: int = CONFIG.case_chunk_overlap,
) -> list[TextChunk]:
    """Chunk every page of the submitted document without discarding later pages."""
    chunks: list[TextChunk] = []

    for page in loaded_case.pages:
        text = page.text or ""
        if not text.strip():
            continue

        start = 0
        part = 1
        while start < len(text):
            end = min(len(text), start + chunk_chars)
            chunk_text = text[start:end]
            chunks.append(
                TextChunk(
                    chunk_id=f"{loaded_case.document_id}:p{page.page_number}:c{part}",
                    source_id=page.source_id,
                    source_name=page.source_name,
                    page_numbers=[page.page_number],
                    text=chunk_text,
                )
            )
            if end >= len(text):
                break
            start = max(0, end - overlap)
            part += 1

    return chunks
