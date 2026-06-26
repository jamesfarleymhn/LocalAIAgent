from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path

from config import ALWAYS_BLOCKED_CASE_FOLDER_NAMES, SAFE_KB_MARKERS, SANITIZE_REQUIRED_FOLDER_NAMES


DATE_RE = re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b")
PHONE_RE = re.compile(r"(?<!\d)(?:\+?1[\s.-]?)?(?:\(?\d{3}\)?[\s.-]?)\d{3}[\s.-]?\d{4}(?!\d)")
EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
IP_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
URL_RE = re.compile(r"\bhttps?://\S+|\bwww\.\S+", re.IGNORECASE)
MEMBERISH_RE = re.compile(r"\b[A-Z]{1,4}\d{5,}[A-Z0-9]*\b", re.IGNORECASE)
LONG_IDENTIFIER_RE = re.compile(r"\b\d{7,}\b")
ZIP_RE = re.compile(r"\b\d{5}(?:-\d{4})\b")

# Label-value identifiers are safer than trying to globally redact every capitalized name.
LABELED_IDENTIFIER_RE = re.compile(
    r"\b(?P<label>"
    r"patient\s+name|member\s+name|subscriber\s+name|date\s+of\s+birth|dob|"
    r"member\s*(?:id|number)|subscriber\s*(?:id|number)|claim\s*(?:id|number|no)|"
    r"account\s*(?:id|number|no)|patient\s+account\s*(?:id|number|no)|mrn|medical\s+record\s*(?:number|no)|"
    r"authorization\s*(?:id|number|no)|auth\s*(?:id|number|no)"
    r")\b\s*[:#-]?\s*(?P<value>[^\n\r|;]{1,120})",
    re.IGNORECASE,
)

ADDRESSISH_RE = re.compile(
    r"\b\d{1,6}\s+[A-Z0-9.'-]+(?:\s+[A-Z0-9.'-]+){0,5}\s+"
    r"(?:ST|STREET|AVE|AVENUE|RD|ROAD|DR|DRIVE|LN|LANE|BLVD|BOULEVARD|CT|COURT|WAY|HWY|HIGHWAY)\b",
    re.IGNORECASE,
)

DEAR_NAME_RE = re.compile(r"\bDear\s+[A-Z][A-Za-z'\-]+(?:\s+[A-Z][A-Za-z'\-]+){0,3}\s*[:,]", re.IGNORECASE)


@dataclass
class PhiScanResult:
    counts: dict[str, int] = field(default_factory=dict)

    @property
    def total(self) -> int:
        return sum(self.counts.values())

    @property
    def has_risk(self) -> bool:
        return self.total > 0


def stable_file_id(path: Path) -> str:
    """Return a stable non-PHI source id without embedding the file name in code or Chroma text."""
    stat = path.stat()
    raw = f"{path.resolve()}|{stat.st_size}|{int(stat.st_mtime)}".encode("utf-8", errors="ignore")
    return hashlib.sha256(raw).hexdigest()[:16]


def path_parts_lower(path: Path, root: Path | None = None) -> list[str]:
    try:
        if root is not None:
            return [part.lower() for part in path.relative_to(root).parts]
    except ValueError:
        pass
    return [part.lower() for part in path.parts]


def has_safe_marker(path: Path, root: Path | None = None) -> bool:
    parts = path_parts_lower(path, root)
    joined = " ".join(parts)
    return any(marker in parts or marker in joined for marker in SAFE_KB_MARKERS)


def is_always_blocked_case_path(path: Path, kb_root: Path) -> bool:
    parts = path_parts_lower(path, kb_root)
    return any(part in ALWAYS_BLOCKED_CASE_FOLDER_NAMES for part in parts)


def requires_sanitization(path: Path, kb_root: Path) -> bool:
    parts = path_parts_lower(path, kb_root)
    return any(part in SANITIZE_REQUIRED_FOLDER_NAMES for part in parts)


def classify_knowledge_path(path: Path, kb_root: Path) -> tuple[str, str]:
    """Classify whether a KB file may be ingested.

    Returns: (decision, reason)
      - block: never ingest live/submitted patient case folders
      - sanitize: ingest only after sanitizer is applied
      - allow: normal reusable knowledge; sanitizer may still run as defense-in-depth
    """
    if is_always_blocked_case_path(path, kb_root):
        return "block", "Path is under a live/submitted patient-case folder. Analyze at runtime with main.py instead."

    if requires_sanitization(path, kb_root):
        return "sanitize", "Appeal/example/case-study material is allowed only after de-identification/sanitization."

    if has_safe_marker(path, kb_root):
        return "allow", "Path contains a safe/de-identified/template marker."

    return "allow", "General knowledge-base material."


def _count(pattern: re.Pattern[str], text: str) -> int:
    return len(pattern.findall(text or ""))


def _count_unredacted_labeled_identifiers(text: str) -> int:
    count = 0
    for match in LABELED_IDENTIFIER_RE.finditer(text or ""):
        value = (match.group("value") or "").strip().lower()
        if value in {"[redacted]", "[date]", "[identifier]", "[number]", "[name]"}:
            continue
        if value.startswith("[") and value.endswith("]"):
            continue
        count += 1
    return count


def scan_phi_indicators(text: str) -> PhiScanResult:
    """Best-effort scanner for common direct identifiers.

    This is a safety screen, not a legal HIPAA de-identification certification.
    """
    text = text or ""
    counts = {
        "ssn": _count(SSN_RE, text),
        "email": _count(EMAIL_RE, text),
        "phone": _count(PHONE_RE, text),
        "url": _count(URL_RE, text),
        "ip_address": _count(IP_RE, text),
        "full_date": _count(DATE_RE, text),
        "labeled_identifier": _count_unredacted_labeled_identifiers(text),
        "member_or_claim_like_identifier": _count(MEMBERISH_RE, text),
        "long_numeric_identifier": _count(LONG_IDENTIFIER_RE, text),
        "zip_plus_four": _count(ZIP_RE, text),
        "street_address_like": _count(ADDRESSISH_RE, text),
        "dear_name": _count(DEAR_NAME_RE, text),
    }
    counts = {key: value for key, value in counts.items() if value > 0}
    return PhiScanResult(counts=counts)


def sanitize_text(text: str) -> tuple[str, dict[str, int]]:
    """Redact common PHI indicators before text is written to Chroma.

    The goal is to preserve reusable appeal logic, denial rationale, criteria, and argument patterns
    while removing identifiers that should not become persistent embeddings.
    """
    text = text or ""
    counts: dict[str, int] = {}

    def sub(name: str, pattern: re.Pattern[str], repl: str, value: str) -> str:
        new_value, count = pattern.subn(repl, value)
        if count:
            counts[name] = counts.get(name, 0) + count
        return new_value

    def replace_labeled(match: re.Match[str]) -> str:
        counts["labeled_identifier"] = counts.get("labeled_identifier", 0) + 1
        label = re.sub(r"\s+", " ", match.group("label")).strip()
        return f"{label}: [REDACTED]"

    text = LABELED_IDENTIFIER_RE.sub(replace_labeled, text)
    text = sub("ssn", SSN_RE, "[SSN]", text)
    text = sub("email", EMAIL_RE, "[EMAIL]", text)
    text = sub("phone", PHONE_RE, "[PHONE]", text)
    text = sub("url", URL_RE, "[URL]", text)
    text = sub("ip_address", IP_RE, "[IP_ADDRESS]", text)
    text = sub("street_address_like", ADDRESSISH_RE, "[ADDRESS]", text)
    text = sub("zip_plus_four", ZIP_RE, "[ZIP]", text)
    text = sub("full_date", DATE_RE, "[DATE]", text)
    text = sub("member_or_claim_like_identifier", MEMBERISH_RE, "[IDENTIFIER]", text)
    text = sub("long_numeric_identifier", LONG_IDENTIFIER_RE, "[NUMBER]", text)
    text = sub("dear_name", DEAR_NAME_RE, "Dear [NAME],", text)
    return text, counts


def redact_identifiers(text: str) -> str:
    """Best-effort redaction for user questions before using them as retrieval queries."""
    return sanitize_text(text or "")[0]
