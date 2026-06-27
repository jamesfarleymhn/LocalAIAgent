from __future__ import annotations

import re
from typing import Any

MISSING_NOTE = "Not confidently extracted from the submitted document."

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
]

DATE_RE = re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b")
MONEY_RE = re.compile(r"(?<!\w)\$\s?\d[\d,]*(?:\.\d{2})?\b")
DRG_RE = re.compile(r"\b(?:MS\s*-?\s*)?DRG\s*#?\s*(?:[O0]*\d{3,5}|[A-Z0-9]{3,6})\b", re.I)
ICD_PCS_RE = re.compile(r"\b[A-HJ-NP-Z0-9]{7}\b")
ICD_ANY_RE = re.compile(r"\b[A-TV-Z][0-9][0-9A-Z](?:\.[0-9A-Z]{1,4})?\b|\b[A-HJ-NP-Z0-9]{7}\b")


def _norm(value: Any) -> str:
    return str(value or "").lower().strip().replace(" ", "_").replace("-", "_")


def _clean(value: Any) -> str | None:
    if value is None:
        return None
    text = re.sub(r"\s+", " ", str(value)).strip(" \t\r\n:;,-|[]{}")
    if not text or text.lower() in {"none", "null", "n/a", "unknown", "not found"}:
        return None
    return text


def _label_hit_count(value: str | None) -> int:
    text = (value or "").lower()
    return sum(1 for term in LABEL_GARBAGE_TERMS if term in text)


def _looks_like_label_garbage(value: str | None) -> bool:
    text = (value or "").lower().strip()
    if not text:
        return True
    if _label_hit_count(text) >= 2:
        return True
    if text.endswith(":") or text.endswith(";"):
        return True
    if "claim number" in text and "legal entity" in text:
        return True
    if "service date" in text and "claim number" in text:
        return True
    return False


def _field_value(field: dict[str, Any]) -> str | None:
    return _clean(field.get("value"))


def _page(field: dict[str, Any]) -> int | None:
    ev = field.get("evidence") or {}
    return ev.get("page_number")


def _excerpt(field: dict[str, Any]) -> str | None:
    ev = field.get("evidence") or {}
    return _clean(ev.get("excerpt"))


def _field_is_acceptable(field: dict[str, Any], role: str | None = None) -> bool:
    value = _field_value(field)
    if not value or _looks_like_label_garbage(value):
        return False
    name = _norm(field.get("name"))
    role = _norm(role) if role else name

    if any(token in role for token in ["date_of_birth", "dob", "date_of_service", "service_date", "admission_date", "discharge_date"]):
        return bool(DATE_RE.search(value))
    if any(token in role for token in ["claim", "account", "member_id", "subscriber", "mrn"]):
        return len(value) <= 80 and _label_hit_count(value) == 0 and bool(re.search(r"\b[A-Z0-9][A-Z0-9-]{3,}\b", value, re.I))
    if "amount" in role or "overpayment" in role:
        return bool(MONEY_RE.search(value))
    if "patient_name" in role:
        return len(value) <= 90 and not re.search(r"\d", value) and _label_hit_count(value) == 0
    if "drg" in role:
        return bool(DRG_RE.search(value))
    if "code" in role and ("procedure" in role or "diagnosis" in role or "non_drg" in role):
        return bool(ICD_ANY_RE.search(value) or re.search(r"\bcode\s+[A-Z0-9.]{3,}\b", value, re.I))
    return True


def _score(field: dict[str, Any], names: list[str], role: str | None = None) -> float:
    name = _norm(field.get("name"))
    wanted = [_norm(n) for n in names]
    value = _field_value(field)
    if not value or not _field_is_acceptable(field, role):
        return -999.0

    score = 0.0
    if name in wanted:
        score += 100
    elif any(w and w in name for w in wanted):
        score += 55
    else:
        return -999.0

    try:
        score += float(field.get("confidence") or 0) * 20
    except Exception:
        pass
    if field.get("validated") is True:
        score += 20
    if field.get("validated") is False:
        score -= 30

    excerpt = (_excerpt(field) or "").lower()
    if any(term in excerpt for term in ["review findings summary", "original codes billed", "new coding assignment", "following review", "provider assigned", "not supported"]):
        score += 12
    if _page(field) is not None:
        score += 2
    return score


def _field_obj(field: dict[str, Any] | None, *, role: str | None = None) -> dict[str, Any]:
    if not field or not _field_is_acceptable(field, role):
        return {"value": None, "confidence": "not_found", "source_page": None, "evidence": None, "note": MISSING_NOTE}
    value = _field_value(field)
    conf = field.get("confidence")
    if conf is None:
        conf_label = "medium"
    else:
        try:
            c = float(conf)
            conf_label = "high" if c >= 0.82 else "medium" if c >= 0.6 else "low"
        except Exception:
            conf_label = str(conf)
    return {
        "value": value,
        "confidence": conf_label,
        "source_page": _page(field),
        "evidence": _excerpt(field),
        "validated": field.get("validated"),
        "field_name": field.get("name"),
    }


def _best(fields: list[dict[str, Any]], names: list[str], *, role: str | None = None) -> dict[str, Any] | None:
    ranked = sorted(fields, key=lambda f: _score(f, names, role), reverse=True)
    if not ranked or _score(ranked[0], names, role) < 0:
        return None
    return ranked[0]


def _all(fields: list[dict[str, Any]], names: list[str], *, role: str | None = None, limit: int = 10) -> list[dict[str, Any]]:
    seen: set[tuple[str, str]] = set()
    out: list[dict[str, Any]] = []
    for field in sorted(fields, key=lambda f: _score(f, names, role), reverse=True):
        if _score(field, names, role) < 0:
            continue
        value = _field_value(field)
        key = (_norm(field.get("name")), value or "")
        if key in seen:
            continue
        seen.add(key)
        out.append(_field_obj(field, role=role))
        if len(out) >= limit:
            break
    return out


def _extract_code(value: str | None) -> str | None:
    if not value:
        return None
    m = re.search(r"\bcode\s+([A-Z0-9.]{3,})\b", value, re.I)
    if m:
        return m.group(1)
    m = ICD_ANY_RE.search(value)
    return m.group(0) if m else None


def _has_value(item: dict[str, Any]) -> bool:
    return bool(item and item.get("value"))


def _coding_summary(original_drg: dict[str, Any], updated_drg: dict[str, Any], unsupported: list[dict[str, Any]], before_code: dict[str, Any], after_finding: dict[str, Any]) -> str:
    parts: list[str] = []
    if _has_value(original_drg) or _has_value(updated_drg):
        parts.append(f"DRG reassignment from {original_drg.get('value') or 'not found'} to {updated_drg.get('value') or 'not found'}")
    finding_val = None
    if unsupported:
        finding_val = unsupported[0].get("payer_finding") or unsupported[0].get("value")
    elif _has_value(after_finding):
        finding_val = after_finding.get("value")
    if finding_val:
        code = _extract_code(finding_val)
        if code:
            parts.append(f"payer stated procedure/code {code} was not supported by the submitted documentation")
        else:
            parts.append(str(finding_val))
    elif _has_value(before_code):
        parts.append(f"non-DRG coding issue involving {before_code.get('value')}")
    return "; ".join(parts) + "." if parts else "Coding change not definitively resolved from extracted evidence."


def _unsupported_code_items(fields: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidates = _all(fields, ["after_non_drg_code_or_finding", "not_supported_code", "procedure_code_after", "after_value"], role="procedure_code", limit=10)
    out: list[dict[str, Any]] = []
    for item in candidates:
        value = item.get("value")
        if not value:
            continue
        low = str(value).lower()
        if "not supported" not in low and "unsupported" not in low:
            continue
        out.append({
            "code": _extract_code(value),
            "code_system": "ICD-10-PCS" if re.search(r"\b[A-HJ-NP-Z0-9]{7}\b", value or "") else None,
            "payer_finding": value,
            "confidence": item.get("confidence"),
            "source_page": item.get("source_page"),
            "evidence": item.get("evidence"),
        })
    return out


def build_final_case_json(result: dict[str, Any]) -> dict[str, Any]:
    extraction = result.get("structured_extraction") or {}
    fields = extraction.get("all_fields") or []
    summary = result.get("summary") or {}
    warnings = result.get("warnings") or []

    payer = _field_obj(_best(fields, ["payer_or_reviewer", "payer", "reviewer", "health_plan", "insurance_company"], role="party"))
    payee = _field_obj(_best(fields, ["provider_or_legal_entity", "provider", "facility", "payee", "legal_entity"], role="party"))

    patient = {
        "name": _field_obj(_best(fields, ["patient_name"], role="patient_name"), role="patient_name"),
        "date_of_birth": _field_obj(_best(fields, ["date_of_birth", "dob"], role="date_of_birth"), role="date_of_birth"),
        "member_id": _field_obj(_best(fields, ["member_id", "subscriber_id", "member_number"], role="member_id"), role="member_id"),
        "account_number": _field_obj(_best(fields, ["account_number", "patient_account_number", "provider_patient_account_number"], role="account_number"), role="account_number"),
        "mrn": _field_obj(_best(fields, ["mrn", "medical_record_number"], role="mrn"), role="mrn"),
    }
    claim = {
        "claim_number": _field_obj(_best(fields, ["claim_number", "claim_id", "claim_no"], role="claim_number"), role="claim_number"),
        "service_dates": _field_obj(_best(fields, ["date_of_service", "dates_of_service", "service_date", "dos"], role="date_of_service"), role="date_of_service"),
        "admission_date": _field_obj(_best(fields, ["admission_date", "admit_date"], role="admission_date"), role="admission_date"),
        "discharge_date": _field_obj(_best(fields, ["discharge_date"], role="discharge_date"), role="discharge_date"),
        "amount_at_issue": _field_obj(_best(fields, ["amount", "overpayment", "denied_amount", "allowed_amount", "money_amount"], role="amount"), role="amount"),
    }

    original_drg = _field_obj(_best(fields, ["drg_before_value", "before_drg", "billed_drg", "original_drg", "requested_drg"], role="drg"), role="drg")
    updated_drg = _field_obj(_best(fields, ["drg_after_value", "after_drg", "recommended_drg", "revised_drg", "approved_drg"], role="drg"), role="drg")
    before_code = _field_obj(_best(fields, ["before_non_drg_code", "procedure_code_before", "diagnosis_code_before", "provider_assigned_code", "original_code"], role="procedure_code"), role="procedure_code")
    after_finding = _field_obj(_best(fields, ["after_non_drg_code_or_finding", "not_supported_code", "procedure_code_after", "diagnosis_code_after", "after_value"], role="procedure_code"), role="procedure_code")
    unsupported = _unsupported_code_items(fields)

    coding_change = {
        "change_type": "DRG reassignment with unsupported procedure/code finding" if (_has_value(original_drg) or _has_value(updated_drg)) and unsupported else "DRG reassignment" if (_has_value(original_drg) or _has_value(updated_drg)) else "Coding finding" if unsupported or _has_value(before_code) or _has_value(after_finding) else "not_resolved",
        "plain_english_summary": _coding_summary(original_drg, updated_drg, unsupported, before_code, after_finding),
        "original_drg": original_drg,
        "updated_drg": updated_drg,
        "provider_assigned_or_billed_non_drg_code": before_code,
        "payer_non_drg_finding": after_finding,
        "unsupported_procedure_or_code_findings": unsupported,
        "diagnosis_code_candidates": _all(fields, ["diagnosis", "icd_10_cm"], role="diagnosis_code", limit=8),
        "procedure_code_candidates": _all(fields, ["procedure", "icd_10_pcs", "cpt", "hcpcs"], role="procedure_code", limit=8),
        "drg_candidates": _all(fields, ["drg"], role="drg", limit=8),
    }

    unresolved = []
    for section_name, section in [("patient", patient), ("claim", claim)]:
        for key, item in section.items():
            if isinstance(item, dict) and not item.get("value"):
                unresolved.append(f"{section_name}.{key}")
    for key in ["original_drg", "updated_drg"]:
        if not coding_change[key].get("value"):
            unresolved.append(f"coding_change.{key}")

    return {
        "schema_version": "3.0-concise-final-case",
        "case_summary": {
            "document_type": _field_obj(_best(fields, ["denial_type", "denial_type"], role="denial")),
            "one_sentence_summary": summary.get("plain_english_summary") or coding_change["plain_english_summary"],
            "key_payer_rationale": summary.get("key_denial_rationale"),
        },
        "parties": {
            "payer_or_reviewer": payer,
            "payee_provider_or_facility": payee,
        },
        "patient": patient,
        "claim": claim,
        "denial": {
            "denial_type": _field_obj(_best(fields, ["denial_type"], role="denial")),
            "payer_decision": _field_obj(_best(fields, ["denial_decision", "decision"], role="denial")),
            "payer_rationale": _field_obj(_best(fields, ["denial_reason", "payer_rationale", "review_findings", "denial_payer_position", "rationale"], role="denial")),
            "appeal_deadline_or_rights": _field_obj(_best(fields, ["appeal_deadline", "appeal_rights", "appeal_instructions"], role="appeal")),
        },
        "coding_change": coding_change,
        "confidence": {
            "overall": "needs_review" if unresolved or warnings else "medium_high",
            "unresolved_important_fields": unresolved,
            "warnings": warnings,
            "note": "This concise object contains one resolved value per field. Use debug JSON only when troubleshooting extraction candidates.",
        },
    }
