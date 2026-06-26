from __future__ import annotations

import re
from collections import OrderedDict, defaultdict
from typing import Any

from chunking import chunk_loaded_case
from json_utils import json_dumps
from llm_client import LocalLLM
from privacy import redact_identifiers
from schemas import Evidence, ExtractedField, LoadedCase, TextChunk, to_plain_json
from vector import retrieve_supporting_knowledge

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
    "appeal_deadline": ["appeal deadline", "deadline", "file an appeal by"],
    "amount": ["amount", "overpayment", "allowed amount", "denied amount"],
}


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
    prompt = CHUNK_EXTRACTION_PROMPT.format(
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
        confidence = item.get("confidence")
        try:
            confidence = float(confidence) if confidence is not None else None
        except (TypeError, ValueError):
            confidence = None
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
                    excerpt=clean_scalar(item.get("evidence_excerpt")),
                ),
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
                        confidence=0.75,
                        evidence=Evidence(
                            source_id=chunk.source_id,
                            source_name=chunk.source_name,
                            page_number=chunk.page_numbers[0] if chunk.page_numbers else None,
                            chunk_id=chunk.chunk_id,
                            excerpt=value[:300],
                        ),
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
            "requested_or_billed_value": first_value(fields, ["denial_requested_or_billed_value", "before_value"]),
            "revised_or_approved_value": first_value(fields, ["denial_revised_or_approved_value", "after_value"]),
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
    prompt = MERGE_SUMMARY_PROMPT.format(extraction_json=json_dumps(extraction, indent=2))
    data = llm.generate_json(prompt, temperature=0.0)
    return data or {}


def extract_case_to_json(
    loaded_case: LoadedCase,
    *,
    use_llm: bool = True,
    include_page_text: bool = False,
    include_source_names: bool = False,
) -> dict[str, Any]:
    chunks = chunk_loaded_case(loaded_case)
    llm = LocalLLM() if use_llm else None
    all_fields: list[ExtractedField] = []
    chunk_summaries: list[dict[str, Any]] = []
    warnings = list(loaded_case.warnings)

    for chunk in chunks:
        all_fields.extend(regex_extract_from_chunk(chunk))
        if llm is not None:
            try:
                chunk_data = llm_extract_from_chunk(chunk, llm)
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

    fields = dedupe_fields(all_fields)
    extraction = {
        "core": build_core_summary(fields),
        "fields_by_category": group_fields(fields),
        "all_fields": [to_plain_json(field) for field in fields],
        "chunk_summaries": chunk_summaries,
    }
    summary = summarize_extraction_with_llm(extraction, llm) if use_llm else summarize_extraction_with_llm(extraction, None)

    result = {
        "schema_version": "2.0",
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
Use the structured extraction and chunk-level answers. Knowledge-base evidence is general support only and must not override case facts.

Return this JSON shape:
{
  "answer": "direct answer to the user question",
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
) -> dict[str, Any]:
    chunks = chunk_loaded_case(loaded_case)
    if not use_llm:
        return {
            "answer": extraction_json.get("summary", {}).get("plain_english_summary")
            or "Regex-only mode completed extraction, but question answering requires the local LLM.",
            "case_facts_used": extraction_json.get("structured_extraction", {}).get("core", {}),
            "supporting_evidence": [],
            "limitations": ["Question answering was run without the local LLM."],
        }

    llm = LocalLLM()
    partials: list[dict[str, Any]] = []
    compact_extraction = {
        "core": extraction_json.get("structured_extraction", {}).get("core", {}),
        "summary": extraction_json.get("summary", {}),
    }

    for chunk in chunks:
        prompt = ANSWER_CHUNK_PROMPT.format(
            question=question,
            extraction_json=json_dumps(compact_extraction, indent=2),
            metadata_json=json_dumps({"chunk_id": chunk.chunk_id, "page_numbers": chunk.page_numbers}, indent=2),
            chunk_text=chunk.text,
        )
        try:
            data = llm.generate_json(prompt, temperature=0.0)
        except Exception as exc:
            partials.append({"chunk_id": chunk.chunk_id, "error": f"{type(exc).__name__}: {exc}"})
            continue
        if data.get("partial_answer") or data.get("evidence"):
            data["chunk_id"] = chunk.chunk_id
            data["page_numbers"] = chunk.page_numbers
            partials.append(data)

    knowledge: list[dict[str, Any]] = []
    if use_kb:
        safe_query = redact_identifiers(question)
        knowledge = retrieve_supporting_knowledge(safe_query, compact_extraction)

    final_prompt = FINAL_ANSWER_PROMPT.format(
        question=question,
        extraction_json=json_dumps(compact_extraction, indent=2),
        partial_answers_json=json_dumps(partials, indent=2),
        knowledge_json=json_dumps(knowledge, indent=2),
    )
    final = llm.generate_json(final_prompt, temperature=0.0)
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
