from __future__ import annotations

import base64
import re
from pathlib import Path
from typing import Any

from config import CONFIG
from final_case import build_final_case_json
from case_review import build_case_review
from json_utils import json_dumps
from llm_client import LocalLLM
from privacy import stable_file_id
from schemas import Evidence, ExtractedField, LoadedCase, to_plain_json



DOCUMENT_VISION_PROMPT = """
You are reading an entire scanned healthcare denial-letter PDF as a sequence of page images.
The images are in page order. Read the document as a whole, not as isolated pages.
Return ONLY valid JSON. Do not include markdown or explanation outside JSON.

Your job:
- Find the most definitive case facts in the whole document.
- Do not try to fill every field from every page.
- Use the page that clearly states the fact.
- Do not use table headers as values.
- Do not invent patient, claim, payer, DRG, diagnosis, procedure, or appeal facts.
- If a value is not clearly visible in the provided images, return null.

Important scanned-denial rules:
- Original/before DRG is usually near “The original codes billed were”.
- Updated/after DRG is usually near “The new coding assignment is”.
- Preserve raw DRG codes exactly if shown, such as “00438”, but also provide normalized code “438”.
- Capture procedure/diagnosis findings such as “ICD-10-PCS code ___” and “code ___ is not supported”.
- If a DRG table shows one row under original codes and one row under new coding assignment, that is a DRG reassignment.
- For patient/claim facts, prefer the Review Findings Summary value row. Never return the header labels as values.

Return this JSON shape exactly:
{
  "document_summary": null,
  "key_payer_rationale": null,
  "parties": {
    "payer_or_reviewer": {"value": null, "source_page": null, "evidence": null},
    "payee_provider_or_facility": {"value": null, "source_page": null, "evidence": null}
  },
  "patient": {
    "name": {"value": null, "source_page": null, "evidence": null},
    "date_of_birth": {"value": null, "source_page": null, "evidence": null},
    "member_id": {"value": null, "source_page": null, "evidence": null},
    "account_number": {"value": null, "source_page": null, "evidence": null},
    "mrn": {"value": null, "source_page": null, "evidence": null}
  },
  "claim": {
    "claim_number": {"value": null, "source_page": null, "evidence": null},
    "service_dates": {"value": null, "source_page": null, "evidence": null},
    "admission_date": {"value": null, "source_page": null, "evidence": null},
    "discharge_date": {"value": null, "source_page": null, "evidence": null},
    "amount_at_issue": {"value": null, "source_page": null, "evidence": null}
  },
  "denial": {
    "denial_type": {"value": null, "source_page": null, "evidence": null},
    "payer_decision": {"value": null, "source_page": null, "evidence": null},
    "payer_rationale": {"value": null, "source_page": null, "evidence": null},
    "appeal_deadline_or_rights": {"value": null, "source_page": null, "evidence": null}
  },
  "coding_change": {
    "change_type": null,
    "original_drg": {"code": null, "raw_code": null, "description": null, "source_page": null, "evidence": null},
    "updated_drg": {"code": null, "raw_code": null, "description": null, "source_page": null, "evidence": null},
    "provider_assigned_or_billed_non_drg_code": {"value": null, "source_page": null, "evidence": null},
    "payer_non_drg_finding": {"value": null, "source_page": null, "evidence": null},
    "unsupported_procedure_or_code_findings": [
      {"code": null, "code_system": null, "payer_finding": null, "source_page": null, "evidence": null}
    ],
    "plain_english_coding_summary": null
  },
  "important_evidence": []
}
"""

VISION_PAGE_PROMPT = """
You are reading one scanned healthcare denial-letter PDF page IMAGE.
Return ONLY valid JSON. Do not include markdown or explanation outside JSON.

Critical rules:
- Read the image itself, including tables and labels.
- Extract only values that are visibly present on this page.
- Do not use table headers as values.
- Do not invent patient, claim, payer, DRG, diagnosis, or procedure facts.
- If a value is not clearly visible on this image, return null or an empty list.
- For DRG tables:
  - The original/before DRG is usually near text like "The original codes billed were".
  - The updated/after DRG is usually near text like "The new coding assignment is".
  - Preserve the raw DRG code if shown, such as "00438", but also normalize it when possible, such as "438".
- For procedure/diagnosis findings, capture statements like "code ___ is not supported" or "not supported by documentation".

Return this JSON shape:
{
  "page_summary": null,
  "payer_or_reviewer": null,
  "payee_provider_or_facility": null,
  "patient": {
    "name": null,
    "date_of_birth": null,
    "member_id": null,
    "account_number": null,
    "mrn": null
  },
  "claim": {
    "claim_number": null,
    "service_dates": null,
    "admission_date": null,
    "discharge_date": null,
    "amount_at_issue": null
  },
  "denial": {
    "denial_type": null,
    "payer_rationale": null,
    "appeal_deadline_or_rights": null
  },
  "coding_change": {
    "original_drg": {"code": null, "raw_code": null, "description": null},
    "updated_drg": {"code": null, "raw_code": null, "description": null},
    "provider_assigned_or_billed_non_drg_code": null,
    "payer_non_drg_finding": null,
    "unsupported_procedure_or_code_findings": [
      {"code": null, "code_system": null, "payer_finding": null}
    ],
    "plain_english_coding_summary": null
  },
  "visible_evidence_phrases": []
}
"""


def _clean(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, (list, tuple, set)):
        parts = [_clean(item) for item in value]
        parts = [item for item in parts if item]
        return "; ".join(parts) if parts else None
    if isinstance(value, dict):
        # Avoid flattening whole objects unintentionally. Common explicit fields only.
        for key in ["value", "text", "code", "raw_code", "description", "payer_finding"]:
            if key in value:
                text = _clean(value.get(key))
                if text:
                    return text
        return None
    text = re.sub(r"\s+", " ", str(value)).strip(" \t\r\n:;,-|[]{}")
    if not text or text.lower() in {"null", "none", "n/a", "unknown", "not found"}:
        return None
    return text


def _normalize_drg_code(raw: str | None) -> str | None:
    raw = _clean(raw)
    if not raw:
        return None
    code = raw.replace("O", "0").replace("o", "0")
    digits = re.sub(r"\D", "", code)
    if not digits:
        return None
    if len(digits) > 3 and digits.startswith("0"):
        digits = digits.lstrip("0") or "0"
    if len(digits) > 3:
        return None
    return digits.zfill(3) if len(digits) < 3 else digits


def _drg_value(obj: Any) -> str | None:
    if not isinstance(obj, dict):
        text = _clean(obj)
        if not text:
            return None
        code = _normalize_drg_code(text)
        return f"DRG {code}" if code else text
    raw = _clean(obj.get("raw_code")) or _clean(obj.get("code"))
    code = _normalize_drg_code(obj.get("code") or raw)
    desc = _clean(obj.get("description"))
    if code and desc:
        if raw and raw != code:
            return f"DRG {code} (raw {raw}) - {desc}"
        return f"DRG {code} - {desc}"
    if code:
        if raw and raw != code:
            return f"DRG {code} (raw {raw})"
        return f"DRG {code}"
    if desc:
        return desc
    return None


def _evidence_text(data: dict[str, Any], fallback: str | None = None) -> str | None:
    phrases = data.get("visible_evidence_phrases") or []
    if isinstance(phrases, list) and phrases:
        return " | ".join(str(x) for x in phrases[:4] if x)
    return fallback


def _field(name: str, value: Any, *, category: str, source_id: str, source_name: str | None, page_number: int, evidence: str | None, confidence: float = 0.9) -> ExtractedField | None:
    value = _clean(value)
    if not value:
        return None
    return ExtractedField(
        name=name,
        value=value,
        category=category,
        confidence=confidence,
        evidence=Evidence(source_id=source_id, source_name=source_name, page_number=page_number, excerpt=evidence),
        validated=True,
        validation_note="Extracted from rendered PDF page image by local vision model.",
    )


def _add_field(fields: list[ExtractedField], *args, **kwargs) -> None:
    item = _field(*args, **kwargs)
    if item is not None:
        fields.append(item)


def fields_from_vision_page(data: dict[str, Any], *, source_id: str, source_name: str | None, page_number: int) -> list[ExtractedField]:
    fields: list[ExtractedField] = []
    evidence = _evidence_text(data, data.get("page_summary"))

    _add_field(fields, "payer_or_reviewer", data.get("payer_or_reviewer"), category="parties", source_id=source_id, source_name=source_name, page_number=page_number, evidence=evidence)
    _add_field(fields, "provider_or_legal_entity", data.get("payee_provider_or_facility"), category="parties", source_id=source_id, source_name=source_name, page_number=page_number, evidence=evidence)

    patient = data.get("patient") or {}
    if isinstance(patient, dict):
        _add_field(fields, "patient_name", patient.get("name"), category="patient", source_id=source_id, source_name=source_name, page_number=page_number, evidence=evidence)
        _add_field(fields, "date_of_birth", patient.get("date_of_birth"), category="patient", source_id=source_id, source_name=source_name, page_number=page_number, evidence=evidence)
        _add_field(fields, "member_id", patient.get("member_id"), category="patient", source_id=source_id, source_name=source_name, page_number=page_number, evidence=evidence)
        _add_field(fields, "account_number", patient.get("account_number"), category="patient", source_id=source_id, source_name=source_name, page_number=page_number, evidence=evidence)
        _add_field(fields, "mrn", patient.get("mrn"), category="patient", source_id=source_id, source_name=source_name, page_number=page_number, evidence=evidence)

    claim = data.get("claim") or {}
    if isinstance(claim, dict):
        _add_field(fields, "claim_number", claim.get("claim_number"), category="claim", source_id=source_id, source_name=source_name, page_number=page_number, evidence=evidence)
        _add_field(fields, "date_of_service", claim.get("service_dates"), category="claim", source_id=source_id, source_name=source_name, page_number=page_number, evidence=evidence)
        _add_field(fields, "admission_date", claim.get("admission_date"), category="claim", source_id=source_id, source_name=source_name, page_number=page_number, evidence=evidence)
        _add_field(fields, "discharge_date", claim.get("discharge_date"), category="claim", source_id=source_id, source_name=source_name, page_number=page_number, evidence=evidence)
        _add_field(fields, "amount", claim.get("amount_at_issue"), category="claim", source_id=source_id, source_name=source_name, page_number=page_number, evidence=evidence)

    denial = data.get("denial") or {}
    if isinstance(denial, dict):
        _add_field(fields, "denial_type", denial.get("denial_type"), category="denial", source_id=source_id, source_name=source_name, page_number=page_number, evidence=evidence, confidence=0.75)
        _add_field(fields, "payer_rationale", denial.get("payer_rationale"), category="denial", source_id=source_id, source_name=source_name, page_number=page_number, evidence=evidence, confidence=0.75)
        _add_field(fields, "appeal_rights", denial.get("appeal_deadline_or_rights"), category="appeal", source_id=source_id, source_name=source_name, page_number=page_number, evidence=evidence, confidence=0.75)

    coding = data.get("coding_change") or {}
    if isinstance(coding, dict):
        original_drg = _drg_value(coding.get("original_drg"))
        updated_drg = _drg_value(coding.get("updated_drg"))
        coding_evidence = _evidence_text(data, coding.get("plain_english_coding_summary") or data.get("page_summary"))
        _add_field(fields, "original_drg", original_drg, category="coding", source_id=source_id, source_name=source_name, page_number=page_number, evidence=coding_evidence, confidence=0.95)
        _add_field(fields, "updated_drg", updated_drg, category="coding", source_id=source_id, source_name=source_name, page_number=page_number, evidence=coding_evidence, confidence=0.95)
        _add_field(fields, "before_non_drg_code", coding.get("provider_assigned_or_billed_non_drg_code"), category="coding", source_id=source_id, source_name=source_name, page_number=page_number, evidence=coding_evidence, confidence=0.85)
        _add_field(fields, "after_non_drg_code_or_finding", coding.get("payer_non_drg_finding"), category="coding", source_id=source_id, source_name=source_name, page_number=page_number, evidence=coding_evidence, confidence=0.85)
        _add_field(fields, "coding_change_summary", coding.get("plain_english_coding_summary"), category="coding", source_id=source_id, source_name=source_name, page_number=page_number, evidence=coding_evidence, confidence=0.85)

        unsupported = coding.get("unsupported_procedure_or_code_findings") or []
        if isinstance(unsupported, list):
            for item in unsupported:
                if not isinstance(item, dict):
                    continue
                code = _clean(item.get("code"))
                finding = _clean(item.get("payer_finding"))
                if code and finding:
                    value = f"code {code} - {finding}"
                else:
                    value = finding or code
                _add_field(fields, "not_supported_code", value, category="coding", source_id=source_id, source_name=source_name, page_number=page_number, evidence=coding_evidence, confidence=0.9)

    return fields




def _obj_value(item: Any) -> str | None:
    if isinstance(item, dict):
        return _clean(item.get("value")) or _clean(item.get("text"))
    return _clean(item)


def _obj_page(item: Any, default_page: int = 1) -> int:
    if isinstance(item, dict):
        try:
            page = item.get("source_page") or item.get("page") or item.get("page_number")
            return int(page) if page else default_page
        except Exception:
            return default_page
    return default_page


def _obj_evidence(item: Any, fallback: str | None = None) -> str | None:
    if isinstance(item, dict):
        return _clean(item.get("evidence")) or _clean(item.get("source_text")) or fallback
    return fallback


def _add_doc_field(fields: list[ExtractedField], name: str, item: Any, *, category: str, source_id: str, source_name: str | None, default_page: int = 1, confidence: float = 0.92, note: str = "Extracted from whole-document page images by local vision model.") -> None:
    value = _obj_value(item)
    if not value:
        return
    page_number = _obj_page(item, default_page=default_page)
    evidence = _obj_evidence(item, fallback=value)
    fields.append(
        ExtractedField(
            name=name,
            value=value,
            category=category,
            confidence=confidence,
            evidence=Evidence(source_id=source_id, source_name=source_name, page_number=page_number, excerpt=evidence),
            validated=True,
            validation_note=note,
        )
    )


def _drg_doc_value(item: Any) -> tuple[str | None, int, str | None]:
    if not isinstance(item, dict):
        return _drg_value(item), 1, _clean(item)
    page = _obj_page(item, default_page=1)
    evidence = _obj_evidence(item)
    value = _drg_value(item)
    return value, page, evidence or value


def fields_from_document_vision(data: dict[str, Any], *, source_id: str, source_name: str | None) -> list[ExtractedField]:
    fields: list[ExtractedField] = []
    summary = _clean(data.get("document_summary"))
    rationale = _clean(data.get("key_payer_rationale"))

    parties = data.get("parties") or {}
    if isinstance(parties, dict):
        _add_doc_field(fields, "payer_or_reviewer", parties.get("payer_or_reviewer"), category="parties", source_id=source_id, source_name=source_name)
        _add_doc_field(fields, "provider_or_legal_entity", parties.get("payee_provider_or_facility"), category="parties", source_id=source_id, source_name=source_name)

    patient = data.get("patient") or {}
    if isinstance(patient, dict):
        _add_doc_field(fields, "patient_name", patient.get("name"), category="patient", source_id=source_id, source_name=source_name)
        _add_doc_field(fields, "date_of_birth", patient.get("date_of_birth"), category="patient", source_id=source_id, source_name=source_name)
        _add_doc_field(fields, "member_id", patient.get("member_id"), category="patient", source_id=source_id, source_name=source_name)
        _add_doc_field(fields, "account_number", patient.get("account_number"), category="patient", source_id=source_id, source_name=source_name)
        _add_doc_field(fields, "mrn", patient.get("mrn"), category="patient", source_id=source_id, source_name=source_name)

    claim = data.get("claim") or {}
    if isinstance(claim, dict):
        _add_doc_field(fields, "claim_number", claim.get("claim_number"), category="claim", source_id=source_id, source_name=source_name)
        _add_doc_field(fields, "date_of_service", claim.get("service_dates"), category="claim", source_id=source_id, source_name=source_name)
        _add_doc_field(fields, "admission_date", claim.get("admission_date"), category="claim", source_id=source_id, source_name=source_name)
        _add_doc_field(fields, "discharge_date", claim.get("discharge_date"), category="claim", source_id=source_id, source_name=source_name)
        _add_doc_field(fields, "amount", claim.get("amount_at_issue"), category="claim", source_id=source_id, source_name=source_name)

    denial = data.get("denial") or {}
    if isinstance(denial, dict):
        _add_doc_field(fields, "denial_type", denial.get("denial_type"), category="denial", source_id=source_id, source_name=source_name, confidence=0.85)
        _add_doc_field(fields, "payer_decision", denial.get("payer_decision"), category="denial", source_id=source_id, source_name=source_name, confidence=0.85)
        _add_doc_field(fields, "payer_rationale", denial.get("payer_rationale") or {"value": rationale, "source_page": 1, "evidence": rationale}, category="denial", source_id=source_id, source_name=source_name, confidence=0.85)
        _add_doc_field(fields, "appeal_rights", denial.get("appeal_deadline_or_rights"), category="appeal", source_id=source_id, source_name=source_name, confidence=0.8)

    coding = data.get("coding_change") or {}
    if isinstance(coding, dict):
        for field_name, key in [("original_drg", "original_drg"), ("updated_drg", "updated_drg")]:
            value, page, evidence = _drg_doc_value(coding.get(key))
            if value:
                fields.append(
                    ExtractedField(
                        name=field_name,
                        value=value,
                        category="coding",
                        confidence=0.97,
                        evidence=Evidence(source_id=source_id, source_name=source_name, page_number=page, excerpt=evidence),
                        validated=True,
                        validation_note="Extracted from whole-document page images by local vision model.",
                    )
                )
        _add_doc_field(fields, "before_non_drg_code", coding.get("provider_assigned_or_billed_non_drg_code"), category="coding", source_id=source_id, source_name=source_name, confidence=0.9)
        _add_doc_field(fields, "after_non_drg_code_or_finding", coding.get("payer_non_drg_finding"), category="coding", source_id=source_id, source_name=source_name, confidence=0.9)
        if _clean(coding.get("plain_english_coding_summary")):
            fields.append(
                ExtractedField(
                    name="coding_change_summary",
                    value=_clean(coding.get("plain_english_coding_summary")),
                    category="coding",
                    confidence=0.9,
                    evidence=Evidence(source_id=source_id, source_name=source_name, page_number=1, excerpt=_clean(coding.get("plain_english_coding_summary"))),
                    validated=True,
                    validation_note="Whole-document vision model summary.",
                )
            )
        unsupported = coding.get("unsupported_procedure_or_code_findings") or []
        if isinstance(unsupported, list):
            for item in unsupported:
                if not isinstance(item, dict):
                    continue
                code = _clean(item.get("code"))
                finding = _clean(item.get("payer_finding"))
                if not code and not finding:
                    continue
                value = f"code {code} - {finding}" if code and finding else (finding or code)
                _add_doc_field(fields, "not_supported_code", {"value": value, "source_page": item.get("source_page"), "evidence": item.get("evidence") or value}, category="coding", source_id=source_id, source_name=source_name, confidence=0.92)

    return fields

def render_pdf_page_to_base64(path: Path, page_number: int, *, zoom: float) -> str:
    fitz = __import__("fitz")
    doc = fitz.open(str(path))
    page = doc[page_number - 1]
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
    return base64.b64encode(pix.tobytes("png")).decode("ascii")


def count_pdf_pages(path: Path) -> int:
    fitz = __import__("fitz")
    with fitz.open(str(path)) as doc:
        return len(doc)


def parse_pages_arg(raw: str | None, *, total_pages: int, max_pages: int) -> list[int]:
    if raw:
        pages: list[int] = []
        for part in raw.split(","):
            part = part.strip()
            if not part:
                continue
            if "-" in part:
                start, end = part.split("-", 1)
                pages.extend(range(int(start), int(end) + 1))
            else:
                pages.append(int(part))
        return [p for p in dict.fromkeys(pages) if 1 <= p <= total_pages]
    return list(range(1, min(total_pages, max_pages) + 1))


def extract_case_with_vision(
    loaded_case: LoadedCase,
    case_paths: list[str],
    *,
    vision_model: str,
    timeout_seconds: int | None = None,
    page_spec: str | None = None,
    max_pages: int = 12,
    zoom: float = 2.0,
    include_page_text: bool = False,
    include_source_names: bool = False,
    progress=None,
) -> dict[str, Any]:
    """Extract final case facts by sending rendered PDF page images to a local Ollama vision model."""
    llm = LocalLLM(model=vision_model, timeout_seconds=timeout_seconds or CONFIG.ollama_timeout_seconds)
    all_fields: list[ExtractedField] = []
    page_results: list[dict[str, Any]] = []
    warnings = list(loaded_case.warnings)

    for file_index, raw_path in enumerate(case_paths, start=1):
        path = Path(raw_path).expanduser().resolve()
        if path.suffix.lower() != ".pdf":
            warnings.append(f"Vision extraction only supports PDFs. Skipped non-PDF file: {path.name}")
            continue
        source_id = stable_file_id(path)
        source_name = path.name if include_source_names else None
        try:
            total_pages = count_pdf_pages(path)
        except Exception as exc:
            warnings.append(f"Could not open PDF for vision extraction: {path.name}: {type(exc).__name__}: {exc}")
            continue
        target_pages = parse_pages_arg(page_spec, total_pages=total_pages, max_pages=max_pages)
        if progress:
            progress.log(f"Vision extraction file {file_index}/{len(case_paths)}: {path.name}")
            progress.log(f"Vision target pages: {target_pages}")

        for page_index, page_number in enumerate(target_pages, start=1):
            try:
                if progress:
                    progress.log(f"Rendering page {page_number}/{total_pages} to image for vision model ({page_index}/{len(target_pages)})...")
                image_b64 = render_pdf_page_to_base64(path, page_number, zoom=zoom)
                if progress:
                    progress.log(f"Sending page {page_number} image to Ollama vision model {vision_model}...")
                data = llm.generate_json_with_images(VISION_PAGE_PROMPT, [image_b64], temperature=0.0) or {}
                if progress:
                    progress.log(f"Vision model returned JSON for page {page_number}.")
                data["source_name"] = source_name
                data["page_number"] = page_number
                page_results.append(data)
                all_fields.extend(fields_from_vision_page(data, source_id=source_id, source_name=source_name, page_number=page_number))
            except Exception as exc:
                warnings.append(f"Vision extraction failed for {path.name} page {page_number}: {type(exc).__name__}: {exc}")

    extraction = {
        "core": {},
        "fields_by_category": {},
        "all_fields": [to_plain_json(field) for field in all_fields],
        "vision_page_results": page_results,
    }
    summary_parts = []
    for item in page_results:
        summary = _clean(item.get("page_summary"))
        if summary:
            summary_parts.append(f"Page {item.get('page_number')}: {summary}")
    summary = {
        "plain_english_summary": " ".join(summary_parts[:4]) if summary_parts else None,
        "key_denial_rationale": None,
        "recommended_next_steps": [],
        "missing_or_uncertain_information": [],
    }
    result = {
        "schema_version": "3.1-vision-final-case",
        "analysis_mode": "vision-fact-check",
        "privacy": {
            "phi_in_source_code": False,
            "case_text_handling": "Submitted PDF pages are rendered at runtime and sent only to the configured local Ollama vision model. The case is not ingested into Chroma.",
            "raw_page_text_included": include_page_text,
        },
        "document": {
            "document_id": loaded_case.document_id,
            "page_count": loaded_case.page_count,
            "vision_model": vision_model,
            "vision_zoom": zoom,
            "source_names_included": include_source_names,
        },
        "structured_extraction": extraction,
        "summary": summary,
        "warnings": warnings,
    }
    result["final_case"] = build_final_case_json(result)
    # If the resolver did not create a good summary, prefer coding summary from final_case.
    final_summary = result["final_case"].get("coding_change", {}).get("plain_english_summary")
    if not result["summary"].get("plain_english_summary") and final_summary:
        result["summary"]["plain_english_summary"] = final_summary
    result["final_case"] = build_final_case_json(result)
    result["case_review"] = build_case_review(result)
    if include_page_text:
        result["document_pages"] = [to_plain_json(page) for page in loaded_case.pages]
    return result


def extract_case_with_document_vision(
    loaded_case: LoadedCase,
    case_paths: list[str],
    *,
    vision_model: str,
    timeout_seconds: int | None = None,
    page_spec: str | None = None,
    max_pages: int = 12,
    zoom: float = 1.5,
    include_page_text: bool = False,
    include_source_names: bool = False,
    progress=None,
) -> dict[str, Any]:
    """Extract final case facts by sending the whole PDF page-image set to one local Ollama vision call.

    This is intentionally different from vision-fact-check, which sends one page per call.
    Whole-document vision lets the model compare pages and choose the most definitive source for
    each fact. It is still local-only through the configured Ollama endpoint.
    """
    llm = LocalLLM(model=vision_model, timeout_seconds=timeout_seconds or CONFIG.ollama_timeout_seconds)
    all_fields: list[ExtractedField] = []
    document_results: list[dict[str, Any]] = []
    warnings = list(loaded_case.warnings)

    for file_index, raw_path in enumerate(case_paths, start=1):
        path = Path(raw_path).expanduser().resolve()
        if path.suffix.lower() != ".pdf":
            warnings.append(f"Whole-document vision only supports PDFs. Skipped non-PDF file: {path.name}")
            continue
        source_id = stable_file_id(path)
        source_name = path.name if include_source_names else None
        try:
            total_pages = count_pdf_pages(path)
        except Exception as exc:
            warnings.append(f"Could not open PDF for whole-document vision extraction: {path.name}: {type(exc).__name__}: {exc}")
            continue

        target_pages = parse_pages_arg(page_spec, total_pages=total_pages, max_pages=max_pages)
        if progress:
            progress.log(f"Whole-document vision file {file_index}/{len(case_paths)}: {path.name}")
            progress.log(f"Whole-document vision target pages: {target_pages}")
            progress.log("Rendering all target pages before one document-level vision call...")

        images: list[str] = []
        for page_number in target_pages:
            try:
                if progress:
                    progress.log(f"Rendering page {page_number}/{total_pages} to image for whole-document vision...")
                images.append(render_pdf_page_to_base64(path, page_number, zoom=zoom))
            except Exception as exc:
                warnings.append(f"Could not render page {page_number} for whole-document vision: {type(exc).__name__}: {exc}")

        if not images:
            warnings.append(f"No page images were rendered for whole-document vision: {path.name}")
            continue

        prompt = DOCUMENT_VISION_PROMPT + f"\n\nPage order in this request: {target_pages}\n"
        try:
            if progress:
                progress.log(f"Sending {len(images)} page image(s) together to Ollama vision model {vision_model}. This is one document-level call, not one call per page...")
            data = llm.generate_json_with_images(prompt, images, temperature=0.0) or {}
            if progress:
                progress.log("Whole-document vision model returned JSON.")
            data["source_name"] = source_name
            data["target_pages"] = target_pages
            document_results.append(data)
            all_fields.extend(fields_from_document_vision(data, source_id=source_id, source_name=source_name))
        except Exception as exc:
            warnings.append(
                f"Whole-document vision extraction failed for {path.name}: {type(exc).__name__}: {exc}. "
                "Try reducing --document-vision-max-pages, lowering --document-vision-zoom, or increasing --ollama-timeout."
            )

    extraction = {
        "core": {},
        "fields_by_category": {},
        "all_fields": [to_plain_json(field) for field in all_fields],
        "whole_document_vision_results": document_results,
    }
    summary_texts = []
    rationale_texts = []
    for item in document_results:
        if _clean(item.get("document_summary")):
            summary_texts.append(_clean(item.get("document_summary")))
        if _clean(item.get("key_payer_rationale")):
            rationale_texts.append(_clean(item.get("key_payer_rationale")))
    summary = {
        "plain_english_summary": summary_texts[0] if summary_texts else None,
        "key_denial_rationale": rationale_texts[0] if rationale_texts else None,
        "recommended_next_steps": [],
        "missing_or_uncertain_information": [],
    }
    result = {
        "schema_version": "4.1-whole-document-vision-final-case",
        "analysis_mode": "document-vision",
        "privacy": {
            "phi_in_source_code": False,
            "case_text_handling": "Submitted PDF pages are rendered at runtime and sent only to the configured local Ollama vision model. The case is not ingested into Chroma.",
            "raw_page_text_included": include_page_text,
        },
        "document": {
            "document_id": loaded_case.document_id,
            "page_count": loaded_case.page_count,
            "vision_model": vision_model,
            "vision_zoom": zoom,
            "vision_target_pages": [item.get("target_pages") for item in document_results],
            "source_names_included": include_source_names,
        },
        "structured_extraction": extraction,
        "summary": summary,
        "warnings": warnings,
    }
    result["final_case"] = build_final_case_json(result)
    final_summary = result["final_case"].get("coding_change", {}).get("plain_english_summary")
    if not result["summary"].get("plain_english_summary") and final_summary:
        result["summary"]["plain_english_summary"] = final_summary
    result["final_case"] = build_final_case_json(result)
    result["case_review"] = build_case_review(result)
    if include_page_text:
        result["document_pages"] = [to_plain_json(page) for page in loaded_case.pages]
    return result
