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
