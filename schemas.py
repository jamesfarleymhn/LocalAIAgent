from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class PageText:
    source_id: str
    source_name: str | None
    page_number: int
    text: str
    extraction_method: str
    warnings: list[str] = field(default_factory=list)


@dataclass
class TextChunk:
    chunk_id: str
    source_id: str
    source_name: str | None
    page_numbers: list[int]
    text: str


@dataclass
class Evidence:
    source_id: str | None = None
    source_name: str | None = None
    page_number: int | None = None
    chunk_id: str | None = None
    excerpt: str | None = None


@dataclass
class ExtractedField:
    name: str
    value: Any
    category: str = "general"
    confidence: float | None = None
    evidence: Evidence = field(default_factory=Evidence)
    validated: bool | None = None
    validation_note: str | None = None


@dataclass
class LoadedCase:
    document_id: str
    pages: list[PageText]
    warnings: list[str] = field(default_factory=list)

    @property
    def full_text(self) -> str:
        return "\n\n".join(
            f"--- PAGE {page.page_number} ---\n{page.text}" for page in self.pages if page.text.strip()
        )

    @property
    def page_count(self) -> int:
        return len(self.pages)


def to_plain_json(value: Any) -> Any:
    """Convert dataclasses and nested structures into JSON-serializable values."""
    if hasattr(value, "__dataclass_fields__"):
        return {key: to_plain_json(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): to_plain_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_plain_json(item) for item in value]
    if isinstance(value, tuple):
        return [to_plain_json(item) for item in value]
    return value
