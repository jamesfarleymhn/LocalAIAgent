from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

DATE = r"\b\d{1,2}/\d{1,2}/\d{2,4}\b"
DRG = r"\b(?:MS\s*[- ]?\s*)?DRG\s*#?\s*(?P<drg>[O0]*\d{3,5})\b"
ICD_CM = r"\b[A-TV-Z][0-9][0-9A-Z](?:\.[0-9A-Z]{1,4})?\b"
ICD_PCS = r"\b[0-9A-HJ-NP-Z]{7}\b"

KEY_TERMS = [
    "request id", "patient", "account", "claim", "service date", "date of service",
    "legal entity", "payer", "provider", "review findings", "denial", "not supported",
    "drg table", "drg description", "original codes billed", "new coding assignment",
    "following review", "icd-10-pcs", "icd-10-cm", "overpayment", "medical necessity",
]

@dataclass
class Evidence:
    field: str
    value: str | None
    page: int | None
    confidence: str
    method: str
    excerpt: str


def clean(s: Any) -> str | None:
    if s is None:
        return None
    s = str(s)
    s = re.sub(r"\s+", " ", s).strip(" :;,-|[](){}\n\t")
    return s or None


def normalize_drg(code: str | None) -> str | None:
    code = clean(code)
    if not code:
        return None
    digits = re.sub(r"\D", "", code.replace("O", "0").replace("o", "0"))
    if not digits:
        return None
    if len(digits) > 3 and digits.startswith("0"):
        digits = digits.lstrip("0") or "0"
    if len(digits) > 3:
        return None
    return digits.zfill(3) if len(digits) < 3 else digits


def excerpt_around(text: str, start: int, end: int, radius: int = 260) -> str:
    a = max(0, start - radius)
    b = min(len(text), end + radius)
    return re.sub(r"\s+", " ", text[a:b]).strip()


def page_texts_to_flat(page_texts: list[dict]) -> str:
    parts = []
    for p in page_texts:
        parts.append(f"\n--- PAGE {p.get('page')} ---\n{p.get('text','')}")
    return "\n".join(parts)


def find_labeled_value(page_texts: list[dict], labels: list[str], field: str) -> Evidence | None:
    for p in page_texts:
        text = p.get("text", "")
        lines = [x.strip() for x in text.splitlines() if x.strip()]
        for i, line in enumerate(lines):
            for label in labels:
                m = re.search(rf"\b{re.escape(label)}\b\s*[:;\-]?\s*(.+)?$", line, flags=re.I)
                if not m:
                    continue
                value = clean(m.group(1))
                if value and not re.fullmatch(r"(?i)(patient|claim|account|service date|date of service|legal entity|payer|provider)", value):
                    return Evidence(field, value, p.get("page"), "medium", "label_same_line", line)
                if i + 1 < len(lines):
                    value = clean(lines[i + 1])
                    if value:
                        return Evidence(field, value, p.get("page"), "medium", "label_next_line", line + " / " + lines[i+1])
    return None


def find_all_drg_mentions(page_texts: list[dict]) -> list[Evidence]:
    out: list[Evidence] = []
    for p in page_texts:
        text = p.get("text", "")
        for m in re.finditer(DRG, text, flags=re.I):
            code = normalize_drg(m.group("drg"))
            if code:
                out.append(Evidence("drg_mentions", f"DRG {code}", p.get("page"), "medium", "drg_regex", excerpt_around(text, m.start(), m.end())))
    return out


def first_drg_in_section(section: str) -> str | None:
    # Prefer rows like: 438 DISORDERS OF PANCREAS ...
    lines = [re.sub(r"\s+", " ", x).strip() for x in section.splitlines() if x.strip()]
    for line in lines:
        line2 = re.sub(r"(?i)\b(DRG|DRG Description|Description|Table)\b", " ", line)
        line2 = re.sub(r"\s+", " ", line2).strip()
        m = re.match(r"^(?:DRG\s*)?([O0]*\d{3,5})\s+([A-Za-z][A-Za-z0-9,\-/&'(). ]{3,180})", line2, flags=re.I)
        if m:
            code = normalize_drg(m.group(1))
            desc = clean(m.group(2))
            return f"DRG {code} {desc}" if code and desc else (f"DRG {code}" if code else None)
    # Fallback flattened section.
    flat = re.sub(r"\s+", " ", section)
    flat = re.sub(r"(?i)\b(DRG Table|DRG Description|Description)\b", " ", flat)
    m = re.search(r"(?<!\d)([O0]*\d{3,5})(?!\d)\s+([A-Za-z][A-Za-z0-9,\-/&'(). ]{4,180})", flat, flags=re.I)
    if m:
        code = normalize_drg(m.group(1))
        desc = clean(m.group(2))
        desc = re.split(r"(?i)\b(new coding assignment|following review|according to|provider assigned|claim|patient|service date)\b", desc or "")[0]
        desc = clean(desc)
        return f"DRG {code} {desc}" if code and desc else (f"DRG {code}" if code else None)
    return None


def extract_drg_table(page_texts: list[dict]) -> list[Evidence]:
    evidences: list[Evidence] = []
    for p in page_texts:
        text = p.get("text", "")
        lower = text.lower()
        if not any(term in lower for term in ["original codes billed", "new coding assignment", "drg table", "drg description"]):
            continue
        normalized = text.replace("|", " ")
        normalized = re.sub(r"[_=]{2,}", "\n", normalized)
        original_match = re.search(r"(?i)(original\s+codes?\s+billed\s+w(?:e|a)re\s*[:;]?)", normalized)
        new_match = re.search(r"(?i)(new\s+coding\s+assignment\s*(?:is)?\s*[:;]?)", normalized)
        if original_match:
            end = new_match.start() if new_match and new_match.start() > original_match.end() else min(len(normalized), original_match.end() + 1800)
            section = normalized[original_match.end():end]
            value = first_drg_in_section(section)
            if value:
                evidences.append(Evidence("before_drg", value, p.get("page"), "high", "drg_table_original_section", excerpt_around(normalized, original_match.start(), end, 120)))
        if new_match:
            section = normalized[new_match.end(): min(len(normalized), new_match.end() + 1800)]
            value = first_drg_in_section(section)
            if value:
                evidences.append(Evidence("after_drg", value, p.get("page"), "high", "drg_table_new_assignment_section", excerpt_around(normalized, new_match.start(), min(len(normalized), new_match.end()+500), 120)))
    return evidences


def extract_code_findings(page_texts: list[dict]) -> list[Evidence]:
    evidences: list[Evidence] = []
    for p in page_texts:
        text = p.get("text", "")
        for m in re.finditer(r"(?i)provider\s+assigned\s+((?:ICD-10-(?:PCS|CM)\s+)?code\s+[A-Z0-9.]{3,10}[^.\n]{0,180})", text):
            evidences.append(Evidence("before_code_or_change", clean(m.group(1)), p.get("page"), "high", "provider_assigned_code_regex", excerpt_around(text, m.start(), m.end())))
        for m in re.finditer(r"(?i)(code\s+[A-Z0-9.]{3,10}\s+is\s+not\s+supported[^.\n]{0,160})", text):
            evidences.append(Evidence("after_code_or_finding", clean(m.group(1)), p.get("page"), "high", "not_supported_code_regex", excerpt_around(text, m.start(), m.end())))
    return evidences


def extract_candidate_sections(page_texts: list[dict]) -> list[dict]:
    sections = []
    for p in page_texts:
        text = p.get("text", "")
        lower = text.lower()
        if any(term in lower for term in KEY_TERMS):
            sections.append({
                "page": p.get("page"),
                "text": text[:5000],
                "matched_terms": [term for term in KEY_TERMS if term in lower],
            })
    return sections


def build_case_facts(page_texts: list[dict]) -> dict:
    evidence: list[Evidence] = []
    fields = {
        "payer_or_reviewer": find_labeled_value(page_texts, ["Legal entity", "Payer", "Health plan", "Insurance company"], "payer_or_reviewer"),
        "patient_name": find_labeled_value(page_texts, ["Patient name", "Patient"], "patient_name"),
        "patient_account_number": find_labeled_value(page_texts, ["Provider's patient account number", "Patient account number", "Account number"], "patient_account_number"),
        "claim_number": find_labeled_value(page_texts, ["Claim number(s)", "Claim number", "Claim ID"], "claim_number"),
        "service_dates": find_labeled_value(page_texts, ["Service date(s)", "Service dates", "Date(s) of service", "Dates of service", "DOS"], "service_dates"),
    }
    for ev in fields.values():
        if ev:
            evidence.append(ev)
    evidence.extend(extract_drg_table(page_texts))
    evidence.extend(extract_code_findings(page_texts))
    drg_mentions = find_all_drg_mentions(page_texts)
    evidence.extend(drg_mentions[:20])
    diagnosis_codes = sorted(set(re.findall(ICD_CM, page_texts_to_flat(page_texts))))
    procedure_codes = sorted(set(x for x in re.findall(ICD_PCS, page_texts_to_flat(page_texts)) if not re.fullmatch(r"\d{7}", x)))
    before_drg = next((e.value for e in evidence if e.field == "before_drg"), None)
    after_drg = next((e.value for e in evidence if e.field == "after_drg"), None)
    return {
        "summary_for_human": {
            "payer_or_reviewer": fields["payer_or_reviewer"].value if fields["payer_or_reviewer"] else None,
            "patient_name": fields["patient_name"].value if fields["patient_name"] else None,
            "patient_account_number": fields["patient_account_number"].value if fields["patient_account_number"] else None,
            "claim_number": fields["claim_number"].value if fields["claim_number"] else None,
            "service_dates": fields["service_dates"].value if fields["service_dates"] else None,
            "before_drg": before_drg,
            "after_drg": after_drg,
            "main_code_change_or_finding": next((e.value for e in evidence if e.field in {"before_code_or_change", "after_code_or_finding"}), None),
        },
        "diagnosis_code_candidates": diagnosis_codes[:50],
        "procedure_code_candidates": procedure_codes[:50],
        "evidence": [asdict(e) for e in evidence],
        "candidate_sections": extract_candidate_sections(page_texts),
    }


def write_outputs(out_base: Path, markdown: str, page_texts: list[dict], facts: dict) -> None:
    out_base.parent.mkdir(parents=True, exist_ok=True)
    out_base.with_suffix(".markdown.md").write_text(markdown, encoding="utf-8")
    out_base.with_suffix(".pages.txt").write_text(page_texts_to_flat(page_texts), encoding="utf-8")
    out_base.with_suffix(".facts.json").write_text(json.dumps(facts, indent=2), encoding="utf-8")
    review = build_review(facts)
    out_base.with_suffix(".case_review.md").write_text(review, encoding="utf-8")


def build_review(facts: dict) -> str:
    s = facts.get("summary_for_human", {})
    lines = ["# PDF Extraction Fact Check", "", "## Key Fields", ""]
    for label, key in [
        ("Payer / reviewer", "payer_or_reviewer"),
        ("Patient", "patient_name"),
        ("Patient account", "patient_account_number"),
        ("Claim number", "claim_number"),
        ("Service dates", "service_dates"),
        ("Before / original DRG", "before_drg"),
        ("After / updated DRG", "after_drg"),
        ("Main code change / finding", "main_code_change_or_finding"),
    ]:
        lines.append(f"- **{label}:** {s.get(key) or 'NOT FOUND'}")
    lines += ["", "## Diagnosis Code Candidates", ""]
    lines.append(", ".join(facts.get("diagnosis_code_candidates", [])) or "None found")
    lines += ["", "## Procedure Code Candidates", ""]
    lines.append(", ".join(facts.get("procedure_code_candidates", [])) or "None found")
    lines += ["", "## Evidence", ""]
    for ev in facts.get("evidence", [])[:40]:
        lines.append(f"### {ev.get('field')} — {ev.get('value') or 'NOT FOUND'}")
        lines.append(f"- Page: {ev.get('page')}")
        lines.append(f"- Confidence: {ev.get('confidence')}")
        lines.append(f"- Method: {ev.get('method')}")
        lines.append(f"- Excerpt: {ev.get('excerpt')}")
        lines.append("")
    return "\n".join(lines)
