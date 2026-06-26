from __future__ import annotations

from typing import Any


MISSING = "Not found / needs manual review"


def _norm(text: Any) -> str:
    return str(text or "").lower().strip().replace(" ", "_").replace("-", "_")


def _field_value(field: dict[str, Any]) -> Any:
    value = field.get("value")
    if value is None:
        return None
    value = str(value).strip()
    return value or None


def _field_score(field: dict[str, Any], wanted_names: list[str]) -> float:
    name = _norm(field.get("name"))
    wanted = [_norm(x) for x in wanted_names]

    score = 0.0
    matched = False
    if name in wanted:
        score += 100
        matched = True
    for item in wanted:
        if item and item in name:
            score += 50
            matched = True
    if not matched:
        return -1.0
    if field.get("validated") is True:
        score += 15
    if field.get("validated") is False:
        score -= 10
    try:
        score += float(field.get("confidence") or 0) * 10
    except Exception:
        pass
    return score


def _best_field(fields: list[dict[str, Any]], names: list[str]) -> dict[str, Any] | None:
    candidates = [f for f in fields if _field_value(f)]
    if not candidates:
        return None
    ranked = sorted(candidates, key=lambda f: _field_score(f, names), reverse=True)
    best = ranked[0]
    if _field_score(best, names) <= 0:
        return None
    return best


def _item(fields: list[dict[str, Any]], names: list[str]) -> dict[str, Any]:
    field = _best_field(fields, names)
    if not field:
        return {"value": None, "display": MISSING, "confidence": None, "page": None, "evidence": None, "validated": None}
    evidence = field.get("evidence") or {}
    value = _field_value(field)
    return {
        "value": value,
        "display": value or MISSING,
        "confidence": field.get("confidence"),
        "page": evidence.get("page_number"),
        "evidence": evidence.get("excerpt"),
        "validated": field.get("validated"),
        "validation_note": field.get("validation_note"),
        "field_name": field.get("name"),
    }


def _all_matching(fields: list[dict[str, Any]], name_tokens: list[str], limit: int = 10) -> list[dict[str, Any]]:
    tokens = [_norm(x) for x in name_tokens]
    matches = []
    seen = set()
    for field in sorted(fields, key=lambda f: _field_score(f, name_tokens), reverse=True):
        name = _norm(field.get("name"))
        value = _field_value(field)
        if not value:
            continue
        if not any(token in name for token in tokens):
            continue
        if tokens == ["drg"] and "non_drg" in name:
            continue
        key = (name, value)
        if key in seen:
            continue
        seen.add(key)
        evidence = field.get("evidence") or {}
        matches.append({
            "value": value,
            "page": evidence.get("page_number"),
            "evidence": evidence.get("excerpt"),
            "confidence": field.get("confidence"),
            "validated": field.get("validated"),
            "field_name": field.get("name"),
        })
        if len(matches) >= limit:
            break
    return matches


def build_case_review(result: dict[str, Any]) -> dict[str, Any]:
    extraction = result.get("structured_extraction") or {}
    fields = extraction.get("all_fields") or []
    summary = result.get("summary") or {}
    core = extraction.get("core") or {}

    review = {
        "purpose": "Human-readable fact check of the submitted denial letter. Patient/case facts are taken from the submitted document only.",
        "summary": {
            "plain_english_summary": summary.get("plain_english_summary") or MISSING,
            "key_denial_rationale": summary.get("key_denial_rationale") or MISSING,
        },
        "parties": {
            "payer_or_reviewer": _item(fields, ["payer", "reviewer", "health_plan", "insurance_company", "payer_or_reviewer"]),
            "payee_provider_or_facility": _item(fields, ["provider", "facility", "payee", "legal_entity", "provider_name"]),
        },
        "patient": {
            "patient_name": _item(fields, ["patient_name", "patient"]),
            "date_of_birth": _item(fields, ["date_of_birth", "dob"]),
            "member_id": _item(fields, ["member_id", "subscriber_id", "member_number"]),
            "account_number": _item(fields, ["account_number", "patient_account_number", "provider_patient_account_number"]),
            "mrn": _item(fields, ["mrn", "medical_record_number"]),
        },
        "claim": {
            "claim_number": _item(fields, ["claim_number", "claim_id", "claim_no"]),
            "service_dates": _item(fields, ["date_of_service", "dates_of_service", "service_date", "dos"]),
            "admission_date": _item(fields, ["admission_date", "admit_date"]),
            "discharge_date": _item(fields, ["discharge_date"]),
            "amount_at_issue": _item(fields, ["amount", "overpayment", "denied_amount", "allowed_amount", "money_amount"]),
        },
        "denial": {
            "denial_type": _item(fields, ["denial_type", "type"]),
            "payer_decision": _item(fields, ["denial_decision", "decision"]),
            "payer_rationale": _item(fields, ["denial_reason", "denial_payer_position", "payer_rationale", "key_denial_rationale", "review_findings", "rationale"]),
            "appeal_deadline_or_rights": _item(fields, ["appeal_deadline", "appeal_rights", "appeal_instructions"]),
        },
        "coding_change": {
            "before_drg": _item(fields, ["drg_before_value", "before_drg", "billed_drg", "original_drg", "requested_drg", "denial_requested_or_billed_value"]),
            "after_drg": _item(fields, ["drg_after_value", "after_drg", "recommended_drg", "revised_drg", "approved_drg", "denial_revised_or_approved_value"]),
            "before_non_drg_code": _item(fields, ["before_non_drg_code", "procedure_code_before", "diagnosis_code_before", "before_value", "billed_code", "provider_assigned_code", "original_code", "requested_or_billed_value"]),
            "after_non_drg_code_or_finding": _item(fields, ["after_non_drg_code_or_finding", "procedure_code_after", "diagnosis_code_after", "after_value", "recommended_code", "revised_code", "not_supported_code", "revised_or_approved_value"]),
            "diagnosis_codes_found": _all_matching(fields, ["diagnosis", "icd_10_cm"], limit=12),
            "procedure_codes_found": _all_matching(fields, ["procedure", "icd_10_pcs", "cpt", "hcpcs"], limit=12),
            "all_drg_mentions": _all_matching(fields, ["drg"], limit=12),
        },
        "quality_check": {
            "warnings": result.get("warnings") or [],
            "knowledge_base_used": bool((result.get("answer") or {}).get("knowledge_base_used")),
            "raw_internal_json_available_under": "structured_extraction.all_fields",
            "how_to_verify": "Review each value's page and evidence excerpt. Values shown as Not found / needs manual review were not confidently extracted.",
        },
    }

    # Add a plain-language coding-change statement.
    before_drg = review["coding_change"]["before_drg"].get("value")
    after_drg = review["coding_change"]["after_drg"].get("value")
    before_code = review["coding_change"]["before_non_drg_code"].get("value")
    after_code = review["coding_change"]["after_non_drg_code_or_finding"].get("value")
    if before_drg or after_drg:
        review["coding_change"]["main_change_summary"] = f"DRG change/reassignment: {before_drg or MISSING} -> {after_drg or MISSING}"
    elif before_code or after_code:
        review["coding_change"]["main_change_summary"] = f"Coding change/finding: {before_code or MISSING} -> {after_code or MISSING}"
    else:
        review["coding_change"]["main_change_summary"] = MISSING

    return review


def _fmt_item(item: dict[str, Any]) -> str:
    value = item.get("display") or item.get("value") or MISSING
    page = item.get("page")
    conf = item.get("confidence")
    parts = [str(value)]
    meta = []
    if page is not None:
        meta.append(f"page {page}")
    if conf is not None:
        try:
            meta.append(f"confidence {float(conf):.2f}")
        except Exception:
            meta.append(f"confidence {conf}")
    if meta:
        parts.append(f"({', '.join(meta)})")
    return " ".join(parts)


def _md_section(title: str, rows: list[tuple[str, dict[str, Any]]]) -> list[str]:
    lines = [f"## {title}", "", "| Fact | Extracted value | Evidence |", "|---|---|---|"]
    for label, item in rows:
        evidence = (item.get("evidence") or "").replace("\n", " ").strip()
        if len(evidence) > 220:
            evidence = evidence[:217].rstrip() + "..."
        lines.append(f"| {label} | {_fmt_item(item)} | {evidence or '—'} |")
    lines.append("")
    return lines


def render_case_review_markdown(review: dict[str, Any]) -> str:
    lines: list[str] = ["# Denial Letter Case Review", ""]
    lines.append("This report is meant for human review. It shows the extracted fact, where it came from, and what still needs manual verification.")
    lines.append("")
    lines.append("## Plain-English Summary")
    lines.append("")
    lines.append(str(review.get("summary", {}).get("plain_english_summary") or MISSING))
    lines.append("")
    lines.append("## Key Payer Rationale")
    lines.append("")
    lines.append(str(review.get("summary", {}).get("key_denial_rationale") or MISSING))
    lines.append("")

    lines.extend(_md_section("Parties", [
        ("Payer / reviewer", review["parties"]["payer_or_reviewer"]),
        ("Payee / provider / facility", review["parties"]["payee_provider_or_facility"]),
    ]))
    lines.extend(_md_section("Patient", [
        ("Patient name", review["patient"]["patient_name"]),
        ("Date of birth", review["patient"]["date_of_birth"]),
        ("Member ID", review["patient"]["member_id"]),
        ("Account number", review["patient"]["account_number"]),
        ("MRN", review["patient"]["mrn"]),
    ]))
    lines.extend(_md_section("Claim", [
        ("Claim number", review["claim"]["claim_number"]),
        ("Service dates", review["claim"]["service_dates"]),
        ("Admission date", review["claim"]["admission_date"]),
        ("Discharge date", review["claim"]["discharge_date"]),
        ("Amount at issue", review["claim"]["amount_at_issue"]),
    ]))
    lines.extend(_md_section("Denial", [
        ("Denial type", review["denial"]["denial_type"]),
        ("Payer decision", review["denial"]["payer_decision"]),
        ("Payer rationale", review["denial"]["payer_rationale"]),
        ("Appeal deadline / rights", review["denial"]["appeal_deadline_or_rights"]),
    ]))
    lines.extend(_md_section("Coding Change", [
        ("Main change summary", {"display": review["coding_change"].get("main_change_summary")}),
        ("Before DRG", review["coding_change"]["before_drg"]),
        ("After DRG", review["coding_change"]["after_drg"]),
        ("Before non-DRG code", review["coding_change"]["before_non_drg_code"]),
        ("After non-DRG code / finding", review["coding_change"]["after_non_drg_code_or_finding"]),
    ]))

    lines.append("## Codes Found for Manual Review")
    lines.append("")
    for label, key in [("Diagnosis codes", "diagnosis_codes_found"), ("Procedure codes", "procedure_codes_found"), ("DRG mentions", "all_drg_mentions")]:
        lines.append(f"### {label}")
        items = review["coding_change"].get(key) or []
        if not items:
            lines.append("- None confidently extracted.")
        else:
            for item in items:
                page = f" page {item.get('page')}" if item.get("page") is not None else ""
                lines.append(f"- {item.get('value')} ({item.get('field_name') or 'field'}{page})")
        lines.append("")

    warnings = review.get("quality_check", {}).get("warnings") or []
    lines.append("## Quality Check")
    lines.append("")
    if warnings:
        for warning in warnings:
            lines.append(f"- Warning: {warning}")
    else:
        lines.append("- No runtime warnings were reported.")
    lines.append("- Verify important facts against the evidence excerpts and source PDF before using in an appeal.")
    lines.append("")
    return "\n".join(lines)
