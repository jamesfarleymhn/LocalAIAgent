from __future__ import annotations

from typing import Any

MISSING = "Not found / needs manual review"


def _get(d: dict[str, Any], *path: str) -> Any:
    cur: Any = d
    for key in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def _item(value: Any) -> dict[str, Any]:
    if isinstance(value, dict) and ("value" in value or "confidence" in value or "evidence" in value):
        return value
    if value is None:
        return {"value": None, "confidence": "not_found", "source_page": None, "evidence": None}
    return {"value": value, "confidence": None, "source_page": None, "evidence": None}


def _display(item: dict[str, Any]) -> str:
    value = item.get("value")
    if value is None or str(value).strip() == "":
        return MISSING
    page = item.get("source_page") or item.get("page")
    conf = item.get("confidence")
    meta = []
    if page is not None:
        meta.append(f"page {page}")
    if conf and conf != "not_found":
        meta.append(f"confidence {conf}")
    return str(value) + (f" ({', '.join(meta)})" if meta else "")


def _evidence(item: dict[str, Any]) -> str:
    evidence = item.get("evidence") or ""
    evidence = str(evidence).replace("\n", " ").strip()
    if len(evidence) > 260:
        evidence = evidence[:257].rstrip() + "..."
    return evidence or "—"


def _section(title: str, rows: list[tuple[str, dict[str, Any]]]) -> list[str]:
    lines = [f"## {title}", "", "| Fact | Resolved value | Evidence |", "|---|---|---|"]
    for label, item in rows:
        lines.append(f"| {label} | {_display(_item(item))} | {_evidence(_item(item))} |")
    lines.append("")
    return lines


def build_case_review(result: dict[str, Any]) -> dict[str, Any]:
    """Build the human-readable report data from the concise final case.

    The old versions rendered directly from all extraction candidates. That made
    the report noisy and confusing. This version renders only from final_case,
    which is the resolved/canonical output for downstream workflows.
    """
    final_case = result.get("final_case") or {}
    if final_case:
        return {
            "purpose": "Human-readable fact check of the submitted denial letter. Values come from final_case, not from every raw extraction candidate.",
            "final_case": final_case,
        }

    # Fallback for older result objects.
    summary = result.get("summary") or {}
    return {
        "purpose": "Human-readable fact check of the submitted denial letter.",
        "final_case": {
            "case_summary": {
                "one_sentence_summary": summary.get("plain_english_summary"),
                "key_payer_rationale": summary.get("key_denial_rationale"),
            },
            "confidence": {"overall": "needs_review", "note": "No final_case object was generated."},
        },
    }


def render_case_review_markdown(review: dict[str, Any]) -> str:
    final = review.get("final_case") or review
    lines: list[str] = ["# Denial Letter Case Review", ""]
    lines.append("This report shows the resolved case facts first. It intentionally hides raw extraction candidates unless you create debug output.")
    lines.append("")

    lines.append("## Concise Summary")
    lines.append("")
    lines.append(str(_get(final, "case_summary", "one_sentence_summary") or MISSING))
    lines.append("")
    lines.append("## Key Payer Rationale")
    lines.append("")
    lines.append(str(_get(final, "case_summary", "key_payer_rationale") or _get(final, "denial", "payer_rationale", "value") or MISSING))
    lines.append("")

    parties = final.get("parties") or {}
    lines.extend(_section("Parties", [
        ("Payer / reviewer", parties.get("payer_or_reviewer") or {}),
        ("Payee / provider / facility", parties.get("payee_provider_or_facility") or {}),
    ]))

    patient = final.get("patient") or {}
    lines.extend(_section("Patient", [
        ("Patient name", patient.get("name") or {}),
        ("Date of birth", patient.get("date_of_birth") or {}),
        ("Member ID", patient.get("member_id") or {}),
        ("Account number", patient.get("account_number") or {}),
        ("MRN", patient.get("mrn") or {}),
    ]))

    claim = final.get("claim") or {}
    lines.extend(_section("Claim", [
        ("Claim number", claim.get("claim_number") or {}),
        ("Service dates", claim.get("service_dates") or {}),
        ("Admission date", claim.get("admission_date") or {}),
        ("Discharge date", claim.get("discharge_date") or {}),
        ("Amount at issue", claim.get("amount_at_issue") or {}),
    ]))

    denial = final.get("denial") or {}
    lines.extend(_section("Denial", [
        ("Denial type", denial.get("denial_type") or _get(final, "case_summary", "document_type") or {}),
        ("Payer decision", denial.get("payer_decision") or {}),
        ("Payer rationale", denial.get("payer_rationale") or {}),
        ("Appeal deadline / rights", denial.get("appeal_deadline_or_rights") or {}),
    ]))

    coding = final.get("coding_change") or {}
    lines.append("## Coding Change")
    lines.append("")
    lines.append(f"**Change type:** {coding.get('change_type') or MISSING}")
    lines.append("")
    lines.append(f"**Plain-English coding summary:** {coding.get('plain_english_summary') or MISSING}")
    lines.append("")
    lines.extend(_section("DRG and Coding Facts", [
        ("Original / billed DRG", coding.get("original_drg") or {}),
        ("Updated / recommended DRG", coding.get("updated_drg") or {}),
        ("Provider assigned / billed non-DRG code", coding.get("provider_assigned_or_billed_non_drg_code") or {}),
        ("Payer non-DRG finding", coding.get("payer_non_drg_finding") or {}),
    ]))

    unsupported = coding.get("unsupported_procedure_or_code_findings") or []
    lines.append("## Unsupported Procedure / Code Findings")
    lines.append("")
    if not unsupported:
        lines.append("- None definitively resolved from extracted evidence.")
    else:
        for item in unsupported:
            code = item.get("code") or "code not isolated"
            page = f" page {item.get('source_page')}" if item.get("source_page") else ""
            lines.append(f"- {code}:{page} {item.get('payer_finding') or ''}")
    lines.append("")

    confidence = final.get("confidence") or {}
    lines.append("## Confidence / Manual Review")
    lines.append("")
    lines.append(f"- Overall: {confidence.get('overall') or 'needs_review'}")
    unresolved = confidence.get("unresolved_important_fields") or []
    if unresolved:
        lines.append("- Unresolved important fields:")
        for item in unresolved:
            lines.append(f"  - {item}")
    warnings = confidence.get("warnings") or []
    if warnings:
        lines.append("- Runtime warnings:")
        for warning in warnings:
            lines.append(f"  - {warning}")
    lines.append("- Verify all values against the source PDF before using in an appeal.")
    lines.append("")
    return "\n".join(lines)
