from __future__ import annotations

import re
from collections import OrderedDict, defaultdict
from typing import Any

from chunking import chunk_loaded_case
from json_utils import json_dumps
from llm_client import LocalLLM
from prompting import render_prompt
from privacy import redact_identifiers
from schemas import Evidence, ExtractedField, LoadedCase, TextChunk, to_plain_json
from validators import validate_llm_field
from vector import retrieve_supporting_knowledge
from case_review import build_case_review

DATE_RE = re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b")
MONEY_RE = re.compile(r"(?<!\w)\$\s?\d[\d,]*(?:\.\d{2})?\b")
DRG_RE = re.compile(r"\b(?:MS\s*-?\s*)?DRG\s*#?\s*[A-Z0-9]{2,6}\b", re.IGNORECASE)
ICD_RE = re.compile(r"\b[A-TV-Z][0-9][0-9A-Z](?:\.[0-9A-Z]{1,4})?\b")
CPT_HCPCS_RE = re.compile(r"\b(?:[A-Z]\d{4}|\d{5})\b")

LABELS: dict[str, list[str]] = {
    "patient_name": ["patient name", "patient"],
    "date_of_birth": ["date of birth", "dob"],
    "member_id": ["member id", "member number", "subscriber id"],
    "claim_number": ["claim number", "claim id", "claim no"],
    "account_number": ["account number", "patient account number", "provider patient account number"],
    "date_of_service": ["date of service", "dates of service", "service date", "dos"],
    "admission_date": ["admission date", "admit date"],
    "discharge_date": ["discharge date"],
    "payer": ["payer", "health plan", "insurance company", "plan"],
    "provider": ["provider", "facility", "legal entity"],
    "denial_reason": ["denial reason", "reason for denial", "rationale", "review findings"],
    "denial_type": ["denial type", "type of denial"],
    "appeal_deadline": ["appeal deadline", "file an appeal by", "appeal must be received by"],
    "amount": ["amount", "overpayment", "allowed amount", "denied amount"],
}
# Labels commonly seen in denial-letter tables. These are useful for rejecting
# OCR/table header text that was accidentally captured as a value.
LABEL_GARBAGE_TERMS = [
    "request id",
    "patient name",
    "humana member identification number",
    "member identification number",
    "member id",
    "subscriber id",
    "patient date of birth",
    "date of birth",
    "provider's patient account number",
    "provider patient account number",
    "patient account number",
    "account number",
    "service date",
    "date of service",
    "claim number",
    "claim id",
    "legal entity",
    "health plan",
    "payer",
    "provider",
]


def _label_hit_count(value: str | None) -> int:
    text = (value or "").lower()
    return sum(1 for term in LABEL_GARBAGE_TERMS if term in text)


def looks_like_label_or_header_value(value: str | None) -> bool:
    """Return True when a supposed value is really OCR/table header text.

    Example of bad text this rejects:
      Patient name: Humana member identification number: Patient date of birth:
      Provider's patient account number: Service date(s): Claim number(s}; Legal entity
    """
    value = clean_scalar(value)
    if not value:
        return True
    lowered = value.lower().strip()
    if lowered in LABEL_GARBAGE_TERMS:
        return True
    if _label_hit_count(lowered) >= 2:
        return True
    if lowered.endswith(":") or lowered.endswith(";"):
        return True
    # OCR often mangles Claim number(s): into Claim number(s};. If that appears
    # inside another field, it is almost certainly a header row, not a value.
    if re.search(r"claim\s+number\s*\(?s?\)?\s*[};:]", lowered):
        return True
    if re.search(r"service\s+date\s*\(?s?\)?\s*[};:]", lowered):
        return True
    return False


def _contains_date(value: str | None) -> bool:
    return bool(value and DATE_RE.search(value))


def _contains_money(value: str | None) -> bool:
    return bool(value and MONEY_RE.search(value))


def _contains_identifier(value: str | None, *, min_len: int = 4) -> bool:
    if not value:
        return False
    return bool(re.search(rf"\b[A-Z0-9][A-Z0-9-]{{{min_len - 1},}}\b", value, flags=re.I))


def is_acceptable_extracted_value(field_name: str, value: str | None) -> bool:
    """Field-specific guardrail before accepting extracted facts.

    This intentionally rejects suspicious label/header text. It is better to show
    "Not found / needs manual review" than to display a table header as a
    patient, claim, DOB, or account value.
    """
    value = clean_scalar(value)
    if not value:
        return False
    name = (field_name or "").lower()
    if looks_like_label_or_header_value(value):
        return False

    if name in {"date_of_birth", "dob", "date_of_service", "service_date", "dates_of_service", "admission_date", "discharge_date", "appeal_deadline"}:
        return _contains_date(value)

    if name in {"claim_number", "claim_id", "claim_no", "account_number", "patient_account_number", "provider_patient_account_number", "member_id", "subscriber_id", "mrn"}:
        # Do not accept long prose or any leftover table labels as identifiers.
        if len(value) > 80:
            return False
        if _label_hit_count(value) >= 1:
            return False
        return _contains_identifier(value, min_len=4)

    if name in {"amount", "money_amount", "overpayment", "denied_amount", "allowed_amount"}:
        return _contains_money(value)

    if name in {"patient_name"}:
        if len(value) > 90:
            return False
        if re.search(r"\d", value):
            return False
        return bool(re.search(r"[A-Za-z]", value))

    return True




def clean_scalar(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, (list, tuple, set)):
        parts = [clean_scalar(item) for item in value]
        parts = [item for item in parts if item]
        return "; ".join(parts) if parts else None
    if isinstance(value, dict):
        parts = [clean_scalar(item) for item in value.values()]
        parts = [item for item in parts if item]
        return "; ".join(parts) if parts else None
    text = re.sub(r"\s+", " ", str(value)).strip(" \t\r\n:;,-|[]{}")
    if not text or text.lower() in {"none", "null", "n/a", "unknown", "not found"}:
        return None
    return text


def excerpt_around(text: str, start: int, end: int, window: int = 180) -> str:
    left = max(0, start - window)
    right = min(len(text), end + window)
    return re.sub(r"\s+", " ", text[left:right]).strip()


def field_from_match(chunk: TextChunk, name: str, value: str, category: str, match_start: int, match_end: int) -> ExtractedField:
    return ExtractedField(
        name=name,
        value=value,
        category=category,
        confidence=0.65,
        evidence=Evidence(
            source_id=chunk.source_id,
            source_name=chunk.source_name,
            page_number=chunk.page_numbers[0] if chunk.page_numbers else None,
            chunk_id=chunk.chunk_id,
            excerpt=excerpt_around(chunk.text, match_start, match_end),
        ),
        validated=True,
        validation_note="Regex fallback value came directly from the source chunk.",
    )


def regex_extract_from_chunk(chunk: TextChunk) -> list[ExtractedField]:
    """Broad, non-template extraction fallback. No patient examples are embedded."""
    fields: list[ExtractedField] = []
    text = chunk.text
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    for line in lines:
        line_start = text.find(line)
        for canonical, labels in LABELS.items():
            for label in labels:
                pattern = re.compile(rf"\b{re.escape(label)}\b\s*[:#-]?\s*(?P<value>.+)$", re.IGNORECASE)
                match = pattern.search(line)
                if not match:
                    continue
                value = clean_scalar(match.group("value"))
                if not value:
                    continue
                if len(value) > 500:
                    value = value[:500].rstrip()
                if not is_acceptable_extracted_value(canonical, value):
                    # Most commonly this means OCR collapsed a table header into
                    # the value position, e.g. Patient name: Member ID: DOB:.
                    continue
                fields.append(
                    field_from_match(
                        chunk,
                        canonical,
                        value,
                        "labeled_field",
                        max(0, line_start + match.start("value")),
                        max(0, line_start + match.end("value")),
                    )
                )
                break

    for regex, name, category in [
        (DATE_RE, "date", "date"),
        (MONEY_RE, "money_amount", "financial"),
        (DRG_RE, "drg_code", "coding"),
        (ICD_RE, "diagnosis_or_procedure_code", "coding"),
        (CPT_HCPCS_RE, "cpt_hcpcs_or_numeric_code", "coding"),
    ]:
        for match in regex.finditer(text):
            fields.append(field_from_match(chunk, name, match.group(0), category, match.start(), match.end()))

    return fields


# -----------------------------
# Denial/coding-specific deterministic extractors
# -----------------------------

def _short_clean(text: str | None, limit: int = 220) -> str | None:
    value = clean_scalar(text)
    if not value:
        return None
    value = re.sub(r"\s+", " ", value).strip(" :;,-|[]{}")
    return value[:limit].rstrip(" :;,-") or None


def _make_special_field(chunk: TextChunk, name: str, value: str, category: str, start: int, end: int, confidence: float = 0.82) -> ExtractedField:
    return ExtractedField(
        name=name,
        value=value,
        category=category,
        confidence=confidence,
        evidence=Evidence(
            source_id=chunk.source_id,
            source_name=chunk.source_name,
            page_number=chunk.page_numbers[0] if chunk.page_numbers else None,
            chunk_id=chunk.chunk_id,
            excerpt=excerpt_around(chunk.text, start, end, window=260),
        ),
        validated=True,
        validation_note="Specialized denial/coding extraction matched source text.",
    )


def _first_drg_value(section: str) -> str | None:
    section = re.sub(r"\s+", " ", section or " ").strip()
    # Handles: DRG 438 Description, MS-DRG #438 Description, or table row: 438 Description.
    match = re.search(r"\b(?:MS\s*-?\s*)?DRG\s*#?\s*(?P<code>[O0]*\d{3,5})\b(?P<desc>[^.;\n]{0,160})", section, re.I)
    if not match:
        match = re.search(r"(?<![\d/])(?P<code>[O0]*\d{3,5})(?![\d/])\s+(?P<desc>[A-Za-z][A-Za-z0-9,\-/&'(). ]{4,160})", section, re.I)
    if not match:
        return None
    code = match.group("code").replace("O", "0").replace("o", "0")
    digits = re.sub(r"\D", "", code).lstrip("0") or re.sub(r"\D", "", code)
    if not digits or len(digits) > 3:
        return None
    digits = digits.zfill(3) if len(digits) < 3 else digits
    desc = _short_clean(match.groupdict().get("desc"), 130)
    if desc:
        desc = re.split(r"\b(?:the\s+new\s+coding\s+assignment|new coding assignment|following review|according to|provider assigned|review findings|claim number|patient name|service date)\b", desc, maxsplit=1, flags=re.I)[0]
        desc = _short_clean(desc, 120)
    return f"DRG {digits}" + (f" {desc}" if desc else "")




def extract_review_findings_summary_table(chunk: TextChunk) -> list[ExtractedField]:
    """Parse common payer review-summary grids when OCR preserves a value row.

    The header often contains labels like Request ID, Patient name, Member ID,
    DOB, Account, Service date(s), Claim number(s), and Legal entity. Generic
    label-value parsing is unsafe for this layout because OCR can collapse the
    header row into one line. This parser only accepts a nearby value row that
    contains date/identifier-shaped values.
    """
    fields: list[ExtractedField] = []
    text = chunk.text
    lines = [re.sub(r"\s+", " ", line).strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return fields

    header_indexes: list[int] = []
    for i, line in enumerate(lines):
        low = line.lower()
        nearby = " ".join(lines[i:min(len(lines), i + 3)]).lower()
        has_header = (
            "review findings summary" in nearby
            or ("request id" in nearby and "patient name" in nearby)
        ) and (
            "patient" in nearby
            and "account" in nearby
            and "service" in nearby
            and "claim" in nearby
        )
        if has_header:
            header_indexes.append(i)

    date_pat = DATE_RE.pattern
    value_pattern = re.compile(
        rf"(?P<request_id>\d{{4,}})\s+"
        rf"(?P<patient_name>[A-Za-z][A-Za-z' .,-]{{2,90}}?)\s+"
        rf"(?P<member_id>[A-Z0-9][A-Z0-9-]{{4,}})\s+"
        rf"(?P<dob>{date_pat})\s+"
        rf"(?P<account>[A-Z0-9][A-Z0-9-]{{3,}})\s+"
        rf"(?P<dos_start>{date_pat})(?:\s*(?:-|to|–|—)\s*(?P<dos_end>{date_pat}))?\s+"
        rf"(?P<claim>[A-Z0-9][A-Z0-9-]{{4,}})\s+"
        rf"(?P<legal_entity>.+)$",
        flags=re.I,
    )

    for header_i in header_indexes:
        # Try the next few lines individually and as joined wrapped lines.
        for start_i in range(header_i + 1, min(len(lines), header_i + 7)):
            if looks_like_label_or_header_value(lines[start_i]):
                continue
            candidates = [lines[start_i]]
            if start_i + 1 < len(lines):
                candidates.append(lines[start_i] + " " + lines[start_i + 1])
            if start_i + 2 < len(lines):
                candidates.append(lines[start_i] + " " + lines[start_i + 1] + " " + lines[start_i + 2])

            for candidate in candidates:
                if looks_like_label_or_header_value(candidate):
                    continue
                match = value_pattern.search(candidate)
                if not match:
                    continue

                mapping = [
                    ("request_id", match.group("request_id"), "identifier"),
                    ("patient_name", match.group("patient_name"), "party"),
                    ("member_id", match.group("member_id"), "identifier"),
                    ("date_of_birth", match.group("dob"), "date"),
                    ("account_number", match.group("account"), "identifier"),
                    ("date_of_service", match.group("dos_start") + (f" - {match.group('dos_end')}" if match.group("dos_end") else ""), "date"),
                    ("claim_number", match.group("claim"), "identifier"),
                    ("provider_or_legal_entity", match.group("legal_entity"), "party"),
                ]

                loc = text.find(lines[start_i])
                if loc < 0:
                    loc = 0
                for field_name, value, category in mapping:
                    value = _short_clean(value, 180)
                    if value and is_acceptable_extracted_value(field_name, value):
                        fields.append(_make_special_field(chunk, field_name, value, category, loc, loc + len(lines[start_i]), 0.94))
                return fields

    return fields


def extract_special_coding_fields_from_chunk(chunk: TextChunk) -> list[ExtractedField]:
    """Target the facts users care about most: before/after DRG and coding changes."""
    fields: list[ExtractedField] = []
    fields.extend(extract_review_findings_summary_table(chunk))
    text = chunk.text
    flat = re.sub(r"\s+", " ", text)

    # Common claim/service-date labels, with special handling for "date(s)" so the
    # generic label scanner does not return only "(s): ...".
    for label_re, field_name in [
        (r"service\s+date(?:\(s\)|s)?", "date_of_service"),
        (r"date(?:s)?\s+of\s+service", "date_of_service"),
        (r"claim\s+(?:number|id|no\.?)", "claim_number"),
        (r"provider(?:'s)?\s+patient\s+account\s+number", "account_number"),
        (r"patient\s+account\s+number", "account_number"),
    ]:
        m = re.search(rf"\b{label_re}\b\s*[:;#-]?\s*(?P<value>[^\n]{{2,180}})", text, flags=re.I)
        if m:
            value = _short_clean(m.group("value"), 180)
            if value and field_name == "date_of_service":
                value = re.sub(r"^\(?s\)?\s*[:;#-]?\s*", "", value, flags=re.I)
            if value and is_acceptable_extracted_value(field_name, value):
                fields.append(_make_special_field(chunk, field_name, value, "date" if "service" in field_name else "identifier", m.start("value"), m.end("value"), 0.86))

    # DRG table pattern: original codes billed were ... new coding assignment is ...
    table = re.search(
        r"(?P<before_label>original\s+codes?\s+billed\s+w(?:e|a)re)\s*[:;]?\s*(?P<before>.{0,1200}?)"
        r"(?:the\s+)?(?P<after_label>new\s+coding\s+assignment\s*(?:is)?)\s*[:;]?\s*(?P<after>.{0,1200})",
        flat,
        flags=re.I,
    )
    if table:
        before = _first_drg_value(table.group("before"))
        after = _first_drg_value(table.group("after"))
        base_start = table.start()
        if before:
            fields.append(_make_special_field(chunk, "drg_before_value", before, "coding", base_start, table.start("after_label"), 0.9))
        if after:
            fields.append(_make_special_field(chunk, "drg_after_value", after, "coding", table.start("after_label"), table.end(), 0.9))

    # Generic DRG before/after language.
    for pattern in [
        r"(?:changed|revised|downgraded|downcoded)?\s*from\s+(?:MS\s*-?\s*)?DRG\s*#?\s*(?P<before>[O0]*\d{3,5})(?P<before_desc>[^.;\n]{0,120})\s+(?:to|into)\s+(?:MS\s*-?\s*)?DRG\s*#?\s*(?P<after>[O0]*\d{3,5})(?P<after_desc>[^.;\n]{0,120})",
        r"(?:billed|submitted|reported|assigned|provider assigned|original(?:ly)? billed|requested)\s+(?:as\s+)?(?:MS\s*-?\s*)?DRG\s*#?\s*(?P<before>[O0]*\d{3,5})(?P<before_desc>[^.;\n]{0,140}).{0,260}?(?:recommended|revised|changed|downgraded|downcoded|approved)\s+(?:to\s+|as\s+)?(?:MS\s*-?\s*)?DRG\s*#?\s*(?P<after>[O0]*\d{3,5})(?P<after_desc>[^.;\n]{0,120})",
    ]:
        match = re.search(pattern, text, flags=re.I | re.S)
        if match:
            before = _first_drg_value("DRG " + match.group("before") + " " + (match.groupdict().get("before_desc") or ""))
            after = _first_drg_value("DRG " + match.group("after") + " " + (match.groupdict().get("after_desc") or ""))
            if before:
                fields.append(_make_special_field(chunk, "drg_before_value", before, "coding", match.start(), match.end(), 0.88))
            if after:
                fields.append(_make_special_field(chunk, "drg_after_value", after, "coding", match.start(), match.end(), 0.88))
            break

    # Non-DRG coding: ICD-10-CM/PCS provider assigned ... following review, code X is not supported.
    before_code = re.search(r"provider\s+assigned\s+(?P<value>ICD-10-(?:PCS|CM)\s+code\s+[A-Z0-9.]+(?:\s*\([^\n.;]{0,180}\))?)", text, flags=re.I)
    after_finding = re.search(r"following\s+review,?\s+(?P<value>code\s+[A-Z0-9.]+\s+is\s+not\s+supported[^.\n]{0,160})", text, flags=re.I)
    if before_code:
        value = _short_clean(before_code.group("value"), 220)
        if value:
            fields.append(_make_special_field(chunk, "before_non_drg_code", value, "coding", before_code.start(), before_code.end(), 0.88))
            if "PCS" in value.upper():
                fields.append(_make_special_field(chunk, "procedure_code_before", value, "coding", before_code.start(), before_code.end(), 0.86))
            if "CM" in value.upper():
                fields.append(_make_special_field(chunk, "diagnosis_code_before", value, "coding", before_code.start(), before_code.end(), 0.86))
    if after_finding:
        value = _short_clean(after_finding.group("value"), 220)
        if value:
            fields.append(_make_special_field(chunk, "after_non_drg_code_or_finding", value, "coding", after_finding.start(), after_finding.end(), 0.88))

    # Payer/reviewer common phrasing in letter headers if labels are not preserved cleanly.
    for label, fname in [("Legal entity", "provider_or_legal_entity"), ("Health plan", "payer_or_reviewer"), ("Payer", "payer_or_reviewer")]:
        m = re.search(rf"\b{re.escape(label)}\b\s*[:;]\s*(?P<value>[^\n]{{2,120}})", text, flags=re.I)
        if m:
            value = _short_clean(m.group("value"), 160)
            if value:
                fields.append(_make_special_field(chunk, fname, value, "party", m.start("value"), m.end("value"), 0.78))

    # Many denial letters start with the payer/reviewer name as a standalone header line.
    for line in [ln.strip() for ln in text.splitlines() if ln.strip()][:8]:
        if re.search(r"\b(?:insurance company|health plan|humana|aetna|cigna|unitedhealth|united healthcare|anthem|optum|cotiviti)\b", line, flags=re.I):
            if ":" not in line and len(line) <= 160:
                start = text.find(line)
                fields.append(_make_special_field(chunk, "payer_or_reviewer", line, "party", max(0, start), max(0, start + len(line)), 0.8))
                break

    return fields


CHUNK_EXTRACTION_PROMPT = """
You are a local healthcare denial document extraction engine.
Return ONLY valid JSON.
Do not include markdown.
Do not invent facts.
Do not use examples.
Extract every useful fact present in this chunk, including identifiers, parties, dates, denial rationale, codes, amounts, deadlines, requested actions, appeal rights, and clinically relevant statements.
For each extracted field, include the exact short evidence excerpt from this chunk.
Use null only when needed. Prefer arrays over prose.

Return this JSON shape:
{
  "chunk_summary": null,
  "fields": [
    {
      "name": "short_snake_case_field_name",
      "value": "exact value or concise extracted fact",
      "category": "identifier|date|party|denial|coding|clinical|financial|appeal|general",
      "confidence": 0.0,
      "evidence_excerpt": "short exact excerpt from the chunk"
    }
  ],
  "denial": {
    "type": null,
    "decision": null,
    "reason": null,
    "payer_position": null,
    "requested_or_billed_value": null,
    "revised_or_approved_value": null
  },
  "open_questions": []
}

Chunk metadata:
{metadata_json}

Chunk text:
{chunk_text}
"""


def llm_extract_from_chunk(chunk: TextChunk, llm: LocalLLM) -> dict[str, Any]:
    prompt = render_prompt(
        CHUNK_EXTRACTION_PROMPT,
        metadata_json=json_dumps(
            {
                "chunk_id": chunk.chunk_id,
                "page_numbers": chunk.page_numbers,
                "source_id": chunk.source_id,
            },
            indent=2,
        ),
        chunk_text=chunk.text,
    )
    return llm.generate_json(prompt, temperature=0.0)


def fields_from_llm_chunk(chunk: TextChunk, data: dict[str, Any]) -> list[ExtractedField]:
    fields: list[ExtractedField] = []
    for item in data.get("fields") or []:
        if not isinstance(item, dict):
            continue
        name = clean_scalar(item.get("name"))
        value = clean_scalar(item.get("value"))
        if not name or not value:
            continue
        if not is_acceptable_extracted_value(name, value):
            continue
        confidence = item.get("confidence")
        try:
            confidence = float(confidence) if confidence is not None else None
        except (TypeError, ValueError):
            confidence = None
        evidence_excerpt = clean_scalar(item.get("evidence_excerpt"))
        validated, validation_note = validate_llm_field(name, value, evidence_excerpt, chunk.text)
        if confidence is not None and not validated:
            confidence = min(confidence, 0.4)
        fields.append(
            ExtractedField(
                name=name,
                value=value,
                category=clean_scalar(item.get("category")) or "general",
                confidence=confidence,
                evidence=Evidence(
                    source_id=chunk.source_id,
                    source_name=chunk.source_name,
                    page_number=chunk.page_numbers[0] if chunk.page_numbers else None,
                    chunk_id=chunk.chunk_id,
                    excerpt=evidence_excerpt,
                ),
                validated=validated,
                validation_note=validation_note,
            )
        )
    denial = data.get("denial")
    if isinstance(denial, dict):
        for key, value in denial.items():
            value = clean_scalar(value)
            if value:
                fields.append(
                    ExtractedField(
                        name=f"denial_{key}",
                        value=value,
                        category="denial",
                        confidence=0.75 if validate_llm_field(f"denial_{key}", value, value[:300], chunk.text)[0] else 0.4,
                        evidence=Evidence(
                            source_id=chunk.source_id,
                            source_name=chunk.source_name,
                            page_number=chunk.page_numbers[0] if chunk.page_numbers else None,
                            chunk_id=chunk.chunk_id,
                            excerpt=value[:300],
                        ),
                        validated=validate_llm_field(f"denial_{key}", value, value[:300], chunk.text)[0],
                        validation_note=validate_llm_field(f"denial_{key}", value, value[:300], chunk.text)[1],
                    )
                )
    return fields


def normalize_key(name: str, value: Any) -> str:
    return f"{name.lower().strip()}::{re.sub(r'\\s+', ' ', str(value)).lower().strip()}"


def dedupe_fields(fields: list[ExtractedField]) -> list[ExtractedField]:
    kept: OrderedDict[str, ExtractedField] = OrderedDict()
    for field in fields:
        value = clean_scalar(field.value)
        if not field.name or not value:
            continue
        if not is_acceptable_extracted_value(field.name, value):
            continue
        field.value = value
        key = normalize_key(field.name, field.value)
        existing = kept.get(key)
        if existing is None:
            kept[key] = field
            continue
        old_conf = existing.confidence or 0
        new_conf = field.confidence or 0
        if new_conf > old_conf:
            kept[key] = field
    return list(kept.values())


def first_value(fields: list[ExtractedField], names: list[str]) -> str | None:
    wanted = {name.lower() for name in names}
    for field in fields:
        normalized = field.name.lower()
        if normalized in wanted or any(token in normalized for token in wanted):
            value = clean_scalar(field.value)
            if value:
                return value
    return None


def group_fields(fields: list[ExtractedField]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for field in fields:
        grouped[field.category or "general"].append(to_plain_json(field))
    return dict(grouped)


def build_core_summary(fields: list[ExtractedField]) -> dict[str, Any]:
    return {
        "patient": {
            "name": first_value(fields, ["patient_name"]),
            "date_of_birth": first_value(fields, ["date_of_birth", "dob"]),
            "member_id": first_value(fields, ["member_id"]),
        },
        "claim": {
            "claim_number": first_value(fields, ["claim_number", "claim_id"]),
            "account_number": first_value(fields, ["account_number", "patient_account_number"]),
            "date_of_service": first_value(fields, ["date_of_service", "service_date", "dos"]),
            "admission_date": first_value(fields, ["admission_date"]),
            "discharge_date": first_value(fields, ["discharge_date"]),
        },
        "parties": {
            "payer": first_value(fields, ["payer", "health_plan", "insurance_company"]),
            "provider": first_value(fields, ["provider", "facility", "legal_entity"]),
        },
        "denial": {
            "type": first_value(fields, ["denial_type", "denial_type"]),
            "reason": first_value(fields, ["denial_reason", "denial_reason", "denial_payer_position"]),
            "decision": first_value(fields, ["denial_decision"]),
            "requested_or_billed_value": first_value(fields, ["denial_requested_or_billed_value", "before_value", "before_non_drg_code", "drg_before_value"]),
            "revised_or_approved_value": first_value(fields, ["denial_revised_or_approved_value", "after_value", "after_non_drg_code_or_finding", "drg_after_value"]),
        },
        "appeal": {
            "deadline": first_value(fields, ["appeal_deadline"]),
            "rights_or_instructions": first_value(fields, ["appeal_rights", "appeal_instructions"]),
        },
    }


MERGE_SUMMARY_PROMPT = """
You are summarizing a healthcare denial document from structured extraction only.
Return ONLY valid JSON. Do not include markdown. Do not invent facts.

Return this JSON shape:
{
  "plain_english_summary": null,
  "key_denial_rationale": null,
  "recommended_next_steps": [],
  "missing_or_uncertain_information": []
}

Structured extraction:
{extraction_json}
"""


def summarize_extraction_with_llm(extraction: dict[str, Any], llm: LocalLLM | None) -> dict[str, Any]:
    if llm is None:
        return {
            "plain_english_summary": extraction.get("core", {}).get("denial", {}).get("reason"),
            "key_denial_rationale": extraction.get("core", {}).get("denial", {}).get("reason"),
            "recommended_next_steps": [],
            "missing_or_uncertain_information": [],
        }
    prompt = render_prompt(MERGE_SUMMARY_PROMPT, extraction_json=json_dumps(extraction, indent=2))
    data = llm.generate_json(prompt, temperature=0.0)
    return data or {}


def extract_case_to_json(
    loaded_case: LoadedCase,
    *,
    use_llm: bool = True,
    include_page_text: bool = False,
    include_source_names: bool = False,
    llm_timeout_seconds: int | None = None,
    progress=None,
) -> dict[str, Any]:
    if progress:
        progress.log("Chunking loaded case for full extraction...")
    chunks = chunk_loaded_case(loaded_case)
    if progress:
        progress.log(f"Created {len(chunks)} text chunk(s) for full extraction.")
    llm = LocalLLM(timeout_seconds=llm_timeout_seconds) if use_llm and llm_timeout_seconds else (LocalLLM() if use_llm else None)
    all_fields: list[ExtractedField] = []
    chunk_summaries: list[dict[str, Any]] = []
    warnings = list(loaded_case.warnings)

    for chunk_index, chunk in enumerate(chunks, start=1):
        if progress:
            progress.log(f"Full extraction analyzing chunk {chunk_index}/{len(chunks)} ({chunk.chunk_id}, pages {chunk.page_numbers})...")
        # Model-first extraction: let the local model understand the page/chunk
        # before using regex as a fallback and validation aid. Regex is no longer
        # responsible for understanding the denial letter layout.
        if llm is not None:
            try:
                if progress:
                    progress.log(f"Sending extraction chunk {chunk_index}/{len(chunks)} to Ollama...")
                chunk_data = llm_extract_from_chunk(chunk, llm)
                if progress:
                    progress.log(f"Ollama extraction returned for chunk {chunk_index}/{len(chunks)}.")
                chunk_summaries.append(
                    {
                        "chunk_id": chunk.chunk_id,
                        "page_numbers": chunk.page_numbers,
                        "chunk_summary": chunk_data.get("chunk_summary"),
                        "open_questions": chunk_data.get("open_questions") or [],
                    }
                )
                all_fields.extend(fields_from_llm_chunk(chunk, chunk_data))
            except Exception as exc:
                warnings.append(f"LLM extraction failed for {chunk.chunk_id}: {type(exc).__name__}: {exc}")

        # Regex extraction remains as a deterministic fallback/validator for
        # dates, DRGs, codes, labeled identifiers, and amounts.
        all_fields.extend(regex_extract_from_chunk(chunk))
        all_fields.extend(extract_special_coding_fields_from_chunk(chunk))

    if progress:
        progress.log(f"Deduplicating and validating {len(all_fields)} extracted field candidate(s)...")
    if progress:
        progress.log(f"Deduplicating {len(all_fields)} fast field candidate(s)...")
    fields = dedupe_fields(all_fields)
    extraction = {
        "core": build_core_summary(fields),
        "fields_by_category": group_fields(fields),
        "all_fields": [to_plain_json(field) for field in fields],
        "chunk_summaries": chunk_summaries,
    }
    if use_llm and llm is not None:
        try:
            if progress:
                progress.log("Sending merged extraction to Ollama for summary...")
            summary = summarize_extraction_with_llm(extraction, llm)
            if progress:
                progress.log("Ollama summary returned.")
        except Exception as exc:
            warnings.append(f"LLM summary failed; used deterministic fallback summary: {type(exc).__name__}: {exc}")
            summary = summarize_extraction_with_llm(extraction, None)
    else:
        summary = summarize_extraction_with_llm(extraction, None)

    result = {
        "schema_version": "2.5-case-review",
        "privacy": {
            "phi_in_source_code": False,
            "case_text_handling": "Submitted documents are read at runtime. The code does not contain embedded patient examples or case facts.",
            "raw_page_text_included": include_page_text,
        },
        "document": {
            "document_id": loaded_case.document_id,
            "page_count": loaded_case.page_count,
            "chunk_count": len(chunks),
            "analyzed_all_chunks": True,
            "source_names_included": include_source_names,
        },
        "structured_extraction": extraction,
        "summary": summary,
        "warnings": warnings,
    }
    result["case_review"] = build_case_review(result)

    if include_page_text:
        result["document_pages"] = [to_plain_json(page) for page in loaded_case.pages]

    return result


ANSWER_CHUNK_PROMPT = """
You are answering a question about one submitted healthcare document chunk.
Return ONLY valid JSON. Do not include markdown. Do not invent facts.
If this chunk does not help answer the question, return an empty partial_answer and empty evidence.

Return this JSON shape:
{
  "partial_answer": "",
  "evidence": [
    {"excerpt": "short exact excerpt", "page_number": null}
  ]
}

User question:
{question}

Structured extraction JSON:
{extraction_json}

Chunk metadata:
{metadata_json}

Chunk text:
{chunk_text}
"""

FINAL_ANSWER_PROMPT = """
You are answering a user's question about a submitted healthcare denial document.
Return ONLY valid JSON. Do not include markdown. Do not invent facts.

Source rules:
- Patient-specific facts come only from the submitted case extraction and chunk answers.
- Knowledge-base evidence is reusable general support only. It may include sanitized appeal examples, de-identified case studies, templates, policies, coding references, CDI guidance, or clinical criteria.
- Do not copy any patient/account/claim/member facts from knowledge-base evidence.
- If the user asks for appeal help, use the knowledge-base evidence to suggest appeal themes, strong arguments, missing documentation to look for, and starter appeal language.
- If the user asks to draft an appeal letter, produce a starter draft with placeholders where facts are missing. Do not invent missing facts.

Return this JSON shape:
{
  "answer": "direct answer to the user question",
  "strong_appeal_arguments": [],
  "appeal_letter_starter": null,
  "case_facts_used": [],
  "supporting_evidence": [],
  "limitations": []
}

User question:
{question}

Structured extraction JSON:
{extraction_json}

Chunk-level answers from every analyzed chunk:
{partial_answers_json}

General knowledge-base evidence, if any:
{knowledge_json}
"""


def answer_question_from_case(
    extraction_json: dict[str, Any],
    loaded_case: LoadedCase,
    question: str,
    *,
    use_llm: bool = True,
    use_kb: bool = False,
    llm_timeout_seconds: int | None = None,
    progress=None,
) -> dict[str, Any]:
    if progress:
        progress.log("Chunking loaded case for question answering...")
    chunks = chunk_loaded_case(loaded_case)
    if progress:
        progress.log(f"Created {len(chunks)} answer chunk(s).")
    if not use_llm:
        return {
            "answer": extraction_json.get("summary", {}).get("plain_english_summary")
            or "Regex-only mode completed extraction, but question answering requires the local LLM.",
            "case_facts_used": extraction_json.get("structured_extraction", {}).get("core", {}),
            "supporting_evidence": [],
            "limitations": ["Question answering was run without the local LLM."],
        }

    llm = LocalLLM(timeout_seconds=llm_timeout_seconds) if llm_timeout_seconds else LocalLLM()
    partials: list[dict[str, Any]] = []
    compact_extraction = {
        "core": extraction_json.get("structured_extraction", {}).get("core", {}),
        "summary": extraction_json.get("summary", {}),
    }

    for chunk_index, chunk in enumerate(chunks, start=1):
        if progress:
            progress.log(f"Answering from chunk {chunk_index}/{len(chunks)} ({chunk.chunk_id}, pages {chunk.page_numbers})...")
        prompt = render_prompt(
            ANSWER_CHUNK_PROMPT,
            question=question,
            extraction_json=json_dumps(compact_extraction, indent=2),
            metadata_json=json_dumps({"chunk_id": chunk.chunk_id, "page_numbers": chunk.page_numbers}, indent=2),
            chunk_text=chunk.text,
        )
        try:
            if progress:
                progress.log(f"Sending answer chunk {chunk_index}/{len(chunks)} to Ollama...")
            data = llm.generate_json(prompt, temperature=0.0)
            if progress:
                progress.log(f"Ollama answer returned for chunk {chunk_index}/{len(chunks)}.")
        except Exception as exc:
            partials.append({"chunk_id": chunk.chunk_id, "error": f"{type(exc).__name__}: {exc}"})
            continue
        if data.get("partial_answer") or data.get("evidence"):
            data["chunk_id"] = chunk.chunk_id
            data["page_numbers"] = chunk.page_numbers
            partials.append(data)

    knowledge: list[dict[str, Any]] = []
    if use_kb:
        if progress:
            progress.log("Retrieving sanitized/de-identified knowledge-base support...")
        # Search with the user's question plus non-identifier denial context so appeal examples
        # and case studies can be retrieved by denial type, rationale, codes, and appeal theme.
        core = compact_extraction.get("core", {}) or {}
        denial = core.get("denial", {}) or {}
        summary = compact_extraction.get("summary", {}) or {}
        search_query = "\n".join(
            str(item)
            for item in [
                question,
                denial.get("type"),
                denial.get("reason"),
                denial.get("decision"),
                denial.get("requested_or_billed_value"),
                denial.get("revised_or_approved_value"),
                summary.get("plain_english_summary"),
                summary.get("key_denial_rationale"),
            ]
            if item
        )
        safe_query = redact_identifiers(search_query)
        knowledge = retrieve_supporting_knowledge(safe_query, compact_extraction)
        if progress:
            progress.log(f"Retrieved {len(knowledge)} knowledge-base evidence item(s).")

    final_prompt = render_prompt(
        FINAL_ANSWER_PROMPT,
        question=question,
        extraction_json=json_dumps(compact_extraction, indent=2),
        partial_answers_json=json_dumps(partials, indent=2),
        knowledge_json=json_dumps(knowledge, indent=2),
    )
    try:
        if progress:
            progress.log("Sending final answer synthesis prompt to Ollama...")
        final = llm.generate_json(final_prompt, temperature=0.0)
        if progress:
            progress.log("Final answer synthesis returned from Ollama.")
    except Exception as exc:
        final = {
            "answer": (
                extraction_json.get("summary", {}).get("plain_english_summary")
                or "The document was extracted, but the local model timed out while building the final answer."
            ),
            "strong_appeal_arguments": [],
            "appeal_letter_starter": None,
            "case_facts_used": compact_extraction.get("core", {}),
            "supporting_evidence": partials,
            "limitations": [f"Final answer LLM call failed: {type(exc).__name__}: {exc}"],
        }
    if not final:
        final = {
            "answer": "The local model did not return a valid final JSON answer.",
            "case_facts_used": [],
            "supporting_evidence": partials,
            "limitations": ["Invalid final LLM JSON."],
        }
    final["analyzed_all_document_chunks_for_answer"] = True
    final["chunk_count"] = len(chunks)
    final["knowledge_base_used"] = bool(use_kb)
    return final

# -----------------------------
# Fast / mode-aware helpers
# -----------------------------

SUMMARY_QUESTION_TERMS = {
    "summarize", "summary", "what is this", "what is the denial", "what does the letter say",
    "denial letter", "denial type", "extract", "structured json", "case facts",
}
APPEAL_QUESTION_TERMS = {
    "appeal", "argument", "arguments", "draft", "letter", "policy", "guideline", "criteria",
    "support", "strong", "strategy", "rebuttal", "respond", "medical necessity", "coding guideline",
}

FAST_PAGE_KEYWORDS = [
    "denial", "denied", "not supported", "review findings", "rationale", "claim", "account",
    "service date", "date of service", "dos", "drg", "ms-drg", "diagnosis", "procedure",
    "overpayment", "appeal", "deadline", "medical necessity", "clinical validation", "coding",
]


def choose_analysis_mode(question: str | None, requested_mode: str = "auto", use_kb: bool = False) -> str:
    """Pick the cheapest safe workflow for the user's request."""
    mode = (requested_mode or "auto").lower().strip()
    if mode in {"fast", "full", "appeal"}:
        return mode
    if mode != "auto":
        raise ValueError("mode must be one of: auto, fast, full, appeal")

    q = (question or "").lower()
    if not q or any(term in q for term in SUMMARY_QUESTION_TERMS):
        # A question like "summarize the denial letter" should stay fast; the word
        # "letter" alone should not trigger appeal generation.
        if not use_kb and not any(term in q for term in {"draft appeal", "write appeal", "appeal argument", "strong argument", "strong appeal", "appeal strategy"}):
            return "fast"
    if use_kb or any(term in q for term in APPEAL_QUESTION_TERMS):
        return "appeal"
    return "full"


def _page_keyword_score(text: str) -> int:
    lowered = text.lower()
    return sum(1 for word in FAST_PAGE_KEYWORDS if word in lowered)


def select_fast_pages(loaded_case: LoadedCase, *, max_pages: int = 8) -> list[int]:
    """Select the pages most likely to answer a basic summary/extraction question."""
    if not loaded_case.pages:
        return []
    selected: set[int] = set()

    # Always include the front matter and last page because denial letters often put
    # demographics/header on page 1 and appeal rights/deadlines near the end.
    for page in loaded_case.pages[:2]:
        if page.text.strip():
            selected.add(page.page_number)
    for page in loaded_case.pages[-1:]:
        if page.text.strip():
            selected.add(page.page_number)

    scored = sorted(
        [(_page_keyword_score(page.text or ""), page.page_number) for page in loaded_case.pages if page.text.strip()],
        reverse=True,
    )
    for score, page_number in scored:
        if len(selected) >= max_pages:
            break
        if score > 0:
            selected.add(page_number)

    if not selected:
        selected = {page.page_number for page in loaded_case.pages[:max_pages] if page.text.strip()}

    return sorted(selected)


def make_compact_case_packet(loaded_case: LoadedCase, *, max_pages: int = 8, max_chars: int = 24000) -> tuple[str, list[int]]:
    pages = select_fast_pages(loaded_case, max_pages=max_pages)
    parts: list[str] = []
    used: list[int] = []
    for page in loaded_case.pages:
        if page.page_number not in pages or not page.text.strip():
            continue
        block = f"--- PAGE {page.page_number} ({page.extraction_method}) ---\n{page.text.strip()}"
        if sum(len(x) for x in parts) + len(block) > max_chars and parts:
            break
        parts.append(block)
        used.append(page.page_number)
    return "\n\n".join(parts), used


FAST_CASE_PROMPT = """
You are a local healthcare denial PDF/text extraction model.
Return ONLY valid JSON. Do not include markdown. Do not invent facts.
The document may contain PHI. Do not create examples. Extract only facts found in the submitted text.

Task:
1. Summarize the denial letter.
2. Extract the most important case facts and denial facts.
3. Answer the user's question if one is provided.
4. Include short supporting excerpts and page numbers where possible.

Return this JSON shape:
{
  "plain_english_summary": null,
  "key_denial_rationale": null,
  "answer": null,
  "fields": [
    {
      "name": "short_snake_case_field_name",
      "value": "exact value or concise fact from text",
      "category": "identifier|date|party|denial|coding|clinical|financial|appeal|general",
      "confidence": 0.0,
      "page_number": null,
      "evidence_excerpt": "short exact excerpt"
    }
  ],
  "recommended_next_steps": [],
  "missing_or_uncertain_information": []
}

User question:
{question}

Already-extracted deterministic facts:
{core_json}

Selected denial document text:
{case_packet}
"""


def fields_from_fast_llm(loaded_case: LoadedCase, data: dict[str, Any]) -> list[ExtractedField]:
    out: list[ExtractedField] = []
    source_id = loaded_case.document_id
    for item in data.get("fields") or []:
        if not isinstance(item, dict):
            continue
        name = clean_scalar(item.get("name"))
        value = clean_scalar(item.get("value"))
        if not name or not value:
            continue
        if not is_acceptable_extracted_value(name, value):
            continue
        page_number = item.get("page_number")
        try:
            page_number = int(page_number) if page_number is not None else None
        except (TypeError, ValueError):
            page_number = None
        evidence_excerpt = clean_scalar(item.get("evidence_excerpt"))
        page_text = "\n".join(page.text for page in loaded_case.pages if page_number is None or page.page_number == page_number)
        validated, note = validate_llm_field(name, value, evidence_excerpt, page_text or loaded_case.full_text)
        try:
            confidence = float(item.get("confidence")) if item.get("confidence") is not None else None
        except (TypeError, ValueError):
            confidence = None
        if confidence is not None and not validated:
            confidence = min(confidence, 0.45)
        out.append(
            ExtractedField(
                name=name,
                value=value,
                category=clean_scalar(item.get("category")) or "general",
                confidence=confidence,
                evidence=Evidence(source_id=source_id, page_number=page_number, excerpt=evidence_excerpt),
                validated=validated,
                validation_note=note,
            )
        )
    return out


def extract_case_to_json_fast(
    loaded_case: LoadedCase,
    *,
    question: str | None = None,
    use_llm: bool = True,
    include_page_text: bool = False,
    include_source_names: bool = False,
    llm_timeout_seconds: int | None = None,
    max_fast_pages: int = 8,
    max_fast_chars: int = 24000,
    progress=None,
) -> dict[str, Any]:
    """Fast path: regex all pages, one compact LLM call, no per-chunk model loop."""
    if progress:
        progress.log("Chunking loaded case for fast extraction...")
    chunks = chunk_loaded_case(loaded_case)
    if progress:
        progress.log(f"Created {len(chunks)} chunk(s). Fast mode will use regex across all chunks and one compact model call.")
    warnings = list(loaded_case.warnings)

    all_fields: list[ExtractedField] = []
    for chunk_index, chunk in enumerate(chunks, start=1):
        if progress:
            progress.log(f"Fast deterministic scan chunk {chunk_index}/{len(chunks)} ({chunk.chunk_id})...")
        all_fields.extend(regex_extract_from_chunk(chunk))
        all_fields.extend(extract_special_coding_fields_from_chunk(chunk))

    deterministic_core = build_core_summary(dedupe_fields(all_fields))
    packet, selected_pages = make_compact_case_packet(loaded_case, max_pages=max_fast_pages, max_chars=max_fast_chars)
    if progress:
        progress.log(f"Selected page(s) for fast local-model summary: {selected_pages}")
        progress.log(f"Fast packet length: {len(packet)} character(s).")
    llm_data: dict[str, Any] = {}
    if use_llm:
        llm = LocalLLM(timeout_seconds=llm_timeout_seconds) if llm_timeout_seconds else LocalLLM()
        prompt = render_prompt(
            FAST_CASE_PROMPT,
            question=question or "Summarize the submitted denial letter.",
            core_json=json_dumps(deterministic_core, indent=2),
            case_packet=packet,
        )
        try:
            if progress:
                progress.log("Sending compact fast summary/extraction prompt to Ollama...")
            llm_data = llm.generate_json(prompt, temperature=0.0) or {}
            if progress:
                progress.log("Ollama fast summary/extraction returned.")
            all_fields.extend(fields_from_fast_llm(loaded_case, llm_data))
        except Exception as exc:
            warnings.append(f"Fast LLM summary/extraction failed; returned deterministic extraction only: {type(exc).__name__}: {exc}")

    fields = dedupe_fields(all_fields)
    extraction = {
        "core": build_core_summary(fields),
        "fields_by_category": group_fields(fields),
        "all_fields": [to_plain_json(field) for field in fields],
        "chunk_summaries": [],
    }
    fallback_summary = summarize_extraction_with_llm(extraction, None)
    summary = {
        "plain_english_summary": clean_scalar(llm_data.get("plain_english_summary")) or fallback_summary.get("plain_english_summary"),
        "key_denial_rationale": clean_scalar(llm_data.get("key_denial_rationale")) or fallback_summary.get("key_denial_rationale"),
        "recommended_next_steps": llm_data.get("recommended_next_steps") or [],
        "missing_or_uncertain_information": llm_data.get("missing_or_uncertain_information") or [],
    }

    result = {
        "schema_version": "2.5-case-review",
        "analysis_mode": "fast",
        "privacy": {
            "phi_in_source_code": False,
            "case_text_handling": "Submitted documents are read at runtime. Fast mode sends only selected runtime pages to the local model and does not ingest the case into Chroma.",
            "raw_page_text_included": include_page_text,
        },
        "document": {
            "document_id": loaded_case.document_id,
            "page_count": loaded_case.page_count,
            "chunk_count": len(chunks),
            "selected_pages_for_fast_llm": selected_pages,
            "analyzed_all_chunks_with_regex": True,
            "analyzed_all_chunks_with_llm": False,
            "source_names_included": include_source_names,
        },
        "structured_extraction": extraction,
        "summary": summary,
        "warnings": warnings,
    }
    result["case_review"] = build_case_review(result)
    if question:
        result["fast_answer_hint"] = clean_scalar(llm_data.get("answer"))
    if include_page_text:
        result["document_pages"] = [to_plain_json(page) for page in loaded_case.pages]
    return result


def answer_question_fast(
    extraction_json: dict[str, Any],
    loaded_case: LoadedCase,
    question: str,
    *,
    use_llm: bool = True,
    llm_timeout_seconds: int | None = None,
    progress=None,
) -> dict[str, Any]:
    existing = clean_scalar(extraction_json.get("fast_answer_hint"))
    summary = extraction_json.get("summary", {}) or {}
    if progress:
        progress.log("Using fast answer from compact model call when available.")
    if existing:
        return {
            "answer": existing,
            "strong_appeal_arguments": [],
            "appeal_letter_starter": None,
            "case_facts_used": extraction_json.get("structured_extraction", {}).get("core", {}),
            "supporting_evidence": [],
            "limitations": ["Fast mode used one compact local-model call instead of analyzing every chunk with the model."],
            "knowledge_base_used": False,
        }
    return {
        "answer": summary.get("plain_english_summary") or summary.get("key_denial_rationale") or "Fast mode completed extraction, but no model-generated answer was available.",
        "strong_appeal_arguments": [],
        "appeal_letter_starter": None,
        "case_facts_used": extraction_json.get("structured_extraction", {}).get("core", {}),
        "supporting_evidence": [],
        "limitations": ["Fast mode returned the extracted summary. Use --mode full for deeper Q&A or --mode appeal --use-kb for appeal strategy."],
        "knowledge_base_used": False,
    }
