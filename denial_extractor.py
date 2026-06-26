from __future__ import annotations

import json
import re
from typing import Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM

from schemas import DenialExtraction


extract_model = OllamaLLM(
    model="llama3.1:latest",
    temperature=0,
)


DATE_PATTERN = r"\d{1,2}/\d{1,2}/\d{2,4}"


def scalar_to_text(value) -> Optional[str]:
    """
    Convert LLM outputs into a safe scalar string.

    Ollama can occasionally return a dict/list even when the prompt asks for a string,
    for example {"code": "871", "description": "..."}. The extractor and
    regex validators should never crash because of that.
    """
    if value is None:
        return None

    if isinstance(value, str):
        return value

    if isinstance(value, (int, float)):
        return str(value)

    if isinstance(value, dict):
        # Prefer common explicit fields first.
        preferred_keys = [
            "value",
            "text",
            "drg",
            "ms_drg",
            "ms-drg",
            "code",
            "description",
            "label",
            "name",
        ]

        parts: list[str] = []

        for key in preferred_keys:
            if key in value:
                part = scalar_to_text(value.get(key))
                if part:
                    parts.append(part)

        # If none of the expected keys were present, flatten scalar leaves.
        if not parts:
            for item in value.values():
                part = scalar_to_text(item)
                if part:
                    parts.append(part)

        return " ".join(parts) if parts else None

    if isinstance(value, (list, tuple, set)):
        parts = []
        for item in value:
            part = scalar_to_text(item)
            if part:
                parts.append(part)
        return "; ".join(parts) if parts else None

    return str(value)


def clean_value(value) -> Optional[str]:
    value = scalar_to_text(value)

    if value is None:
        return None

    value = value.strip()
    value = re.sub(r"\s+", " ", value)
    value = value.strip(" :;,-|[](){}")

    if not value or value.lower() in {"null", "none", "n/a", "unknown"}:
        return None

    return value


def normalize_ocr_text(text: str) -> str:
    """Normalize common OCR mistakes without changing meaning too aggressively."""
    replacements = {
        "Provider'$": "Provider's",
        "Claim number(s};": "Claim number(s):",
        "Claim number(s};": "Claim number(s):",
        "Service date(s);": "Service date(s):",
        "Legal entity;": "Legal entity:",
        "Humana:": "Humana",
    }

    for bad, good in replacements.items():
        text = text.replace(bad, good)

    # Standardize dashes between dates.
    text = text.replace("–", "-").replace("—", "-")
    return text


def value_appears_in_text(value: Optional[str], text: str) -> bool:
    if value is None:
        return True

    normalized_value = re.sub(r"\s+", " ", str(value).strip()).lower()
    normalized_text = re.sub(r"\s+", " ", text).lower()

    return normalized_value in normalized_text


def find_labeled_value_same_or_next_line(text: str, labels: list[str]) -> Optional[str]:
    """
    Basic label-value fallback:
    Label: value
    or
    Label:
    value

    This intentionally does not try to parse OCR tables with many labels on one line.
    Those are handled separately by parse_review_findings_summary().
    """
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    for i, line in enumerate(lines):
        for label in labels:
            pattern = rf"\b{re.escape(label)}\b\s*:?[\s\-]*(.*)$"
            match = re.search(pattern, line, flags=re.IGNORECASE)

            if not match:
                continue

            value = clean_value(match.group(1))

            # Avoid returning another label as the value.
            if value and not re.search(r"\b(patient|claim|service|legal entity|account number|date of birth)\b", value, re.IGNORECASE):
                return value

            if i + 1 < len(lines):
                next_line = clean_value(lines[i + 1])
                if next_line:
                    return next_line

    return None


def parse_review_findings_summary(text: str) -> dict:

    result: dict[str, Optional[str]] = {}
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    for i, line in enumerate(lines):
        label_line = line.lower()

        has_summary_labels = (
            "request id" in label_line
            and "patient" in label_line
            and "account" in label_line
            and "service date" in label_line
            and "claim" in label_line
            and "legal entity" in label_line
        )

        if not has_summary_labels or i + 1 >= len(lines):
            continue

        # Sometimes OCR wraps the value row; try one and two lines.
        value_line = lines[i + 1]
        if i + 2 < len(lines) and "provider assigned" not in lines[i + 2].lower():
            value_line_2 = value_line + " " + lines[i + 2]
        else:
            value_line_2 = value_line

        pattern = re.compile(
            rf"(?P<request_id>\d{{5,}})\s+"
            rf"(?P<patient_name>[A-Z][A-Z'\- ]+?)\s+"
            rf"(?P<member_id>[A-Z]\d{{6,}})\s+"
            rf"(?P<dob>{DATE_PATTERN})\s+"
            rf"(?P<account>\d{{5,}})\s+"
            rf"(?P<dos_start>{DATE_PATTERN})\s*[-to]+\s*(?P<dos_end>{DATE_PATTERN})\s+"
            rf"(?P<claim>\d{{8,}})\s+"
            rf"(?P<legal_entity>.+)$",
            flags=re.IGNORECASE,
        )

        match = pattern.search(value_line_2)

        if not match:
            # Return the raw value row for troubleshooting if needed later.
            result["review_findings_raw_value_line"] = value_line
            continue

        patient_name = clean_value(match.group("patient_name"))

        # Conservative correction for a common EasyOCR artifact seen in this file: IJAMES -> JAMES.
        # Only applies when the first token is OCR prefixed with I and there is at least a last name.
        if patient_name:
            parts = patient_name.split()
            if len(parts) >= 2 and parts[0].startswith("I") and len(parts[0]) > 4:
                # Avoid changing likely real names such as ISAAC/IAN by only correcting when the
                # remaining token is still a common all-caps word-like name length.
                corrected_first = parts[0][1:]
                if corrected_first.isalpha() and corrected_first.isupper():
                    patient_name = " ".join([corrected_first] + parts[1:])

        result.update(
            {
                "provider_request_id": clean_value(match.group("request_id")),
                "patient_name": patient_name,
                "member_id": clean_value(match.group("member_id")),
                "patient_date_of_birth": clean_value(match.group("dob")),
                "patient_account_number": clean_value(match.group("account")),
                "service_date_start": clean_value(match.group("dos_start")),
                "service_date_end": clean_value(match.group("dos_end")),
                "claim_number": clean_value(match.group("claim")),
                "provider_name": clean_value(match.group("legal_entity")),
            }
        )

        return result

    return result


def extract_service_dates_fallback(text: str) -> tuple[Optional[str], Optional[str]]:
    value = find_labeled_value_same_or_next_line(
        text,
        ["Service date(s)", "Service dates", "Date(s) of service", "Dates of service", "DOS"],
    )

    if not value:
        return None, None

    dates = re.findall(DATE_PATTERN, value)

    if len(dates) >= 2:
        return dates[0], dates[1]

    if len(dates) == 1:
        return dates[0], dates[0]

    return None, None


def extract_header_fields(case_text: str) -> dict:
    text = normalize_ocr_text(case_text)

    # First try the structured OCR table pattern.
    result = parse_review_findings_summary(text)

    # Then fill remaining fields with simple label-value fallback.
    if not result.get("patient_name"):
        result["patient_name"] = find_labeled_value_same_or_next_line(text, ["Patient name"])

    if not result.get("patient_account_number"):
        result["patient_account_number"] = find_labeled_value_same_or_next_line(
            text,
            ["Provider's patient account number", "Provider patient account number", "Patient account number", "Account number"],
        )

    if not result.get("claim_number"):
        result["claim_number"] = find_labeled_value_same_or_next_line(text, ["Claim number(s)", "Claim number", "Claim ID"])

    if not result.get("provider_name"):
        result["provider_name"] = find_labeled_value_same_or_next_line(
            text,
            ["Legal entity", "Payer", "Health plan", "Insurance company"],
        )

    if not result.get("service_date_start") or not result.get("service_date_end"):
        start, end = extract_service_dates_fallback(text)
        result["service_date_start"] = result.get("service_date_start") or start
        result["service_date_end"] = result.get("service_date_end") or end

    return result


def extract_denial_type_rule_based(text: str) -> Optional[str]:
    q = text.lower()

    if "postpayment" in q or "post-payment" in q or "overpaid" in q or "overpayment" in q:
        if "icd-10-pcs" in q or "code" in q or "not supported" in q:
            return "Post-payment coding denial / overpayment review"
        return "Post-payment medical record review / overpayment review"

    if "drg" in q and ("downgrade" in q or "revised" in q or "recommended" in q or "original codes billed" in q or "new coding assignment" in q):
        return "DRG/coding reassignment"

    if "medical necessity" in q:
        return "Medical necessity denial"

    if "clinical validation" in q:
        return "Clinical validation denial"

    if "authorization" in q or "pre-service" in q:
        return "Authorization / pre-service denial"

    return None


def infer_policy_type(text: str, provider_name: Optional[str]) -> Optional[str]:
    """Infer broad payer/policy type only when the submitted letter supports it."""
    q = text.lower()
    provider = (provider_name or "").lower()

    if "managed medicaid" in q:
        return "Managed Medicaid"

    if "medicare advantage" in q or "medicare part c" in q:
        return "Medicare Advantage"

    if "medicaid" in q:
        return "Medicaid"

    if "medicare" in q:
        return "Medicare"

    # Humana Insurance Company alone does not prove Medicare/Medicaid; default to Commercial
    # only when there is no stronger text evidence above.
    if "humana insurance company" in provider or provider == "humana":
        return "Commercial"

    return None


def extract_before_value_rule_based(text: str) -> Optional[str]:
    """Extract the original/billed non-DRG coding value without letting OCR run the capture too far."""
    patterns = [
        # Keep these deliberately short. Some OCR output changes the closing parenthesis to }
        # or drops it entirely; an unlimited [^)] capture can absorb half the letter.
        r"provider assigned\s+(ICD-10-PCS code\s+[A-Z0-9\.]+(?:\s*\([^\n.;:]{0,180}[\)\}\]])?)",
        r"provider assigned\s+(ICD-10-CM code\s+[A-Z0-9\.]+(?:\s*\([^\n.;:]{0,180}[\)\}\]])?)",
        r"billed\s+((?:MS\s*[- ]?\s*)?DRG\s*#?\s*[O0]*\d{3,5}[^\.\n]{0,120})",
        r"assigned\s+((?:MS\s*[- ]?\s*)?DRG\s*#?\s*[O0]*\d{3,5}[^\.\n]{0,120})",
    ]

    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            value = clean_value(match.group(1))
            if not value:
                continue

            # Stop the capture if it ran into the payer's finding section.
            value = re.split(
                r"\b(?:following review|according to|in this case|the history and physical|review findings)\b",
                value,
                maxsplit=1,
                flags=re.IGNORECASE,
            )[0]
            return clean_value(value)

    return None


def extract_after_value_rule_based(text: str) -> Optional[str]:
    patterns = [
        r"following review,\s+(code\s+[A-Z0-9\.]+\s+is not supported)",
        r"recommended\s+((?:MS\s*[- ]?\s*)?DRG\s*#?\s*[O0]*\d{3,5}[^\.\n]{0,120})",
        r"revised\s+((?:MS\s*[- ]?\s*)?DRG\s*#?\s*[O0]*\d{3,5}[^\.\n]{0,120})",
        r"downcoded\s+to\s+([^\.\n]+)",
    ]

    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return clean_value(match.group(1))

    return None


def normalize_drg_code(code: Optional[str]) -> Optional[str]:
    """
    Normalize DRG codes from OCR.

    Examples handled:
    - 405 -> 405
    - 00438 -> 438
    - O0438 / OO438 -> 438 (OCR letter O used for zero)
    """
    code = clean_value(code)

    if not code:
        return None

    # OCR frequently reads a leading zero as the letter O.
    code = code.replace("O", "0").replace("o", "0")
    digits = re.sub(r"\D", "", code)

    if not digits:
        return None

    if len(digits) > 3 and digits.startswith("0"):
        digits = digits.lstrip("0") or "0"

    # Reject long non-DRG identifiers such as claim/account numbers.
    if len(digits) > 3:
        return None

    # Keep true leading-zero DRGs as three characters if needed, e.g. 005.
    return digits.zfill(3) if len(digits) < 3 else digits

def normalize_drg_value(value: Optional[str]) -> Optional[str]:
    """Normalize DRG values while preserving any short description that follows."""
    value = clean_value(value)

    if not value:
        return None

    value = re.sub(r"\bMS\s*[- ]?\s*DRG\b", "MS-DRG", value, flags=re.IGNORECASE)

    def _replace_drg_code(match: re.Match) -> str:
        prefix = "MS-DRG" if "MS" in match.group(0).upper() else "DRG"
        code = normalize_drg_code(match.group("code"))
        return f"{prefix} {code}" if code else match.group(0)

    value = re.sub(
        r"\b(?:(?:MS\s*[- ]?\s*)?DRG)\s*#?\s*(?P<code>[O0]*\d{3,5})\b",
        _replace_drg_code,
        value,
        flags=re.IGNORECASE,
    )
    value = re.sub(r"\s+", " ", value).strip(" :;,-|[](){}")

    # Do not let a captured DRG value run into the next unrelated sentence/field.
    value = re.split(
        r"\b(?:patient|claim|account|service date|date of service|provider|payer|legal entity|request id)\b\s*:??",
        value,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0].strip(" :;,-|[](){}")

    return value or None


def make_drg_value(code: str, description: Optional[str] = None) -> Optional[str]:
    code = normalize_drg_code(code)

    if not code:
        return None

    description = clean_value(description)

    if description:
        # Keep descriptions short and avoid accidentally absorbing later sentences.
        description = re.split(
            r"\b(?:the\s+new\s+coding\s+assignment|following\s+review|according\s+to|in\s+this\s+case|provider\s+assigned|review\s+findings|claim\s+number|patient\s+name|request\s+id|service\s+date)\b|\n\s*\n",
            description,
            maxsplit=1,
            flags=re.IGNORECASE,
        )[0]
        description = re.sub(r"\b(?:DRG|MS-DRG|DRG Description|Description)\b", "", description, flags=re.IGNORECASE)
        description = re.sub(r"[|_]{2,}", " ", description)
        description = re.sub(r"\s+", " ", description).strip(" :;,-|[](){}")
        description = description[:140].strip(" :;,-|[](){}")
        if description:
            return normalize_drg_value(f"DRG {code} {description}") or f"DRG {code}"

    return f"DRG {code}"


def _normalize_table_text_for_drg(text: str) -> str:
    """Make OCR/table text easier to scan without destroying labels."""
    text = text or ""
    text = text.replace("\r", "\n")
    text = text.replace("|", " ")
    text = re.sub(r"[_=]{2,}", "\n", text)
    text = re.sub(r"(?i)DRG\s*Description", "DRG Description", text)
    text = re.sub(r"(?i)new\s+coding\s+assignment\s*is", "new coding assignment is", text)
    text = re.sub(r"(?i)new\s+coding\s+assignmentis", "new coding assignment is", text)
    text = re.sub(r"(?i)original\s+codes?\s+billed\s+w(?:e|a)re", "original codes billed were", text)
    return text


def _candidate_drg_code_pattern() -> str:
    # Allows OCR leading zero/letter-O variants such as 00438, O0438, OO438.
    return r"(?P<code>[O0]*\d{3,5})"


def _clean_drg_description_tail(tail: str) -> Optional[str]:
    tail = tail or ""
    tail = tail.replace("|", " ")
    tail = re.sub(r"[_=]{2,}", " ", tail)
    tail = re.sub(r"(?i)^\s*(?:DRG\s+Description|Description|DRG)\b\s*", "", tail)
    tail = re.split(
        r"\b(?:the\s+new\s+coding\s+assignment|new\s+coding\s+assignment|following\s+review|according\s+to|in\s+this\s+case|provider\s+assigned|review\s+findings|claim\s+number|patient\s+name|request\s+id|service\s+date|date\s+of\s+birth|legal\s+entity|DRG\s+Table)\b|\n\s*\n",
        tail,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]
    tail = re.sub(r"\s+", " ", tail).strip(" :;,-|[](){}")

    # If another numeric code appears later in the same OCR paragraph, stop before it.
    tail = re.split(r"\s+(?=[O0]*\d{3,5}\s+[A-Z])", tail, maxsplit=1)[0]
    tail = tail[:160].strip(" :;,-|[](){}")

    if not tail or not re.search(r"[A-Za-z]", tail):
        return None

    # Avoid known non-description labels.
    if re.fullmatch(r"(?i)(?:DRG|DRG Description|Description|Table)", tail):
        return None

    return tail


def extract_first_drg_row_from_section(section: str) -> Optional[str]:
    """
    Extract the first DRG row from OCR/table text.

    Handles all of these shapes:
    - DRG   DRG Description\n405   PANCREAS, LIVER AND SHUNT PROCEDURES WITH MCC
    - DRG DRG Description 405 PANCREAS, LIVER AND SHUNT PROCEDURES WITH MCC
    - 405\nPANCREAS, LIVER AND SHUNT PROCEDURES WITH MCC
    - 00438 DIS OF PANCREAS EXC MALIG W MCC
    """
    section = _normalize_table_text_for_drg(section)

    lines = [re.sub(r"\s+", " ", line).strip() for line in section.splitlines()]
    lines = [line for line in lines if line]

    # First pass: line-oriented extraction.
    for i, line in enumerate(lines):
        # Remove table headers even if OCR put them before the row on the same line.
        candidate_line = re.sub(r"(?i)\b(?:DRG\s+Table|DRG\s+Description|Description)\b", " ", line)
        candidate_line = re.sub(r"\s+", " ", candidate_line).strip()

        if not candidate_line or re.fullmatch(r"(?i)(?:DRG|Table)", candidate_line):
            continue

        match = re.match(rf"^(?:DRG\s*)?{_candidate_drg_code_pattern()}\s+(?P<desc>.+)$", candidate_line, flags=re.IGNORECASE)

        if match:
            desc = _clean_drg_description_tail(match.group("desc"))
            if desc and not re.search(r"(?i)\b(?:claim|account|patient|service date|request id|dob|date of birth)\b", desc):
                return make_drg_value(match.group("code"), desc)

        # Sometimes OCR puts the code on one line and the description on the next.
        code_only = re.match(rf"^(?:DRG\s*)?{_candidate_drg_code_pattern()}$", candidate_line, flags=re.IGNORECASE)
        if code_only and i + 1 < len(lines):
            next_line = lines[i + 1]
            if not re.search(r"(?i)\b(?:DRG|description|original|new coding|assignment|claim|account|patient|service date)\b", next_line):
                desc = _clean_drg_description_tail(next_line)
                if desc:
                    return make_drg_value(code_only.group("code"), desc)

    # Second pass: flattened extraction for EasyOCR paragraph mode where the full table is one line.
    flat = re.sub(r"\s+", " ", section).strip()
    flat = re.sub(r"(?i)\b(?:DRG\s+Table|DRG\s+Description|Description)\b", " ", flat)
    flat = re.sub(r"\s+", " ", flat).strip()

    # Look for first 3-5 digit/OCR-O DRG code followed by a text description.
    for match in re.finditer(rf"(?<![\d/]){_candidate_drg_code_pattern()}(?![\d/])\s+(?P<tail>[A-Za-z][A-Za-z0-9,\-/&'(). ]{{4,220}})", flat, flags=re.IGNORECASE):
        code = normalize_drg_code(match.group("code"))
        if not code:
            continue

        desc = _clean_drg_description_tail(match.group("tail"))
        if not desc:
            continue

        if re.search(r"(?i)\b(?:claim|account|patient|service date|request id|dob|date of birth)\b", desc):
            continue

        return make_drg_value(code, desc)

    return None


def _section_between(text: str, start_pattern: str, end_pattern: Optional[str] = None, *, max_chars: int = 2500) -> Optional[str]:
    start_match = re.search(start_pattern, text, flags=re.IGNORECASE | re.DOTALL)
    if not start_match:
        return None

    start = start_match.end()
    end = min(len(text), start + max_chars)

    if end_pattern:
        end_match = re.search(end_pattern, text[start:end], flags=re.IGNORECASE | re.DOTALL)
        if end_match:
            end = start + end_match.start()

    return text[start:end]


def extract_drg_table_before_after_rule_based(text: str) -> tuple[Optional[str], Optional[str]]:
    """
    Handles denial-letter DRG tables with labels like:
    - DRG Table
    - The original codes billed were:
    - The new coding assignment is:

    This intentionally supports OCR where the row is not preserved as a clean table.
    """
    text = _normalize_table_text_for_drg(text)

    original_pattern = r"(?:the\s+)?original\s+codes?\s+billed\s+w(?:e|a)re\s*[:;]?"
    new_pattern = r"(?:the\s+)?new\s+coding\s+assignment\s*(?:is)?\s*[:;]?"

    before = None
    after = None

    # Prefer the DRG Table neighborhood if present, because other parts of the letter may also
    # use the phrase "original codes billed" for ICD-10-PCS/procedure-code tables.
    drg_table_match = re.search(r"(?i)\bDRG\s+Table\b", text)
    drg_text = text[drg_table_match.start(): drg_table_match.start() + 4000] if drg_table_match else text

    before_section = _section_between(
        drg_text,
        original_pattern,
        new_pattern,
        max_chars=2000,
    )
    if before_section:
        before = extract_first_drg_row_from_section(before_section)

    after_section = _section_between(
        drg_text,
        new_pattern,
        r"\b(?:following\s+review|according\s+to|in\s+this\s+case|provider\s+assigned|review\s+findings|appeal|rationale|claim\s+number|patient\s+name)\b",
        max_chars=2000,
    )
    if after_section:
        after = extract_first_drg_row_from_section(after_section)

    # Last-resort single-regex table extraction for paragraph-mode OCR.
    if not before or not after:
        compact = re.sub(r"\s+", " ", drg_text)
        table_match = re.search(
            rf"{original_pattern}.{{0,700}}?{_candidate_drg_code_pattern()}\s+(?P<before_desc>[A-Za-z][A-Za-z0-9,\-/&'(). ]{{4,180}}?)\s+{new_pattern}.{{0,700}}?(?P<after_code>[O0]*\d{{3,5}})\s+(?P<after_desc>[A-Za-z][A-Za-z0-9,\-/&'(). ]{{4,180}})",
            compact,
            flags=re.IGNORECASE,
        )
        if table_match:
            before = before or make_drg_value(table_match.group("code"), _clean_drg_description_tail(table_match.group("before_desc")))
            after = after or make_drg_value(table_match.group("after_code"), _clean_drg_description_tail(table_match.group("after_desc")))

    return before, after

def extract_drg_pair_rule_based(text: str) -> tuple[Optional[str], Optional[str]]:
    """
    Find common before/after DRG phrasing, such as:
    - changed from DRG 291 to DRG 292
    - billed MS-DRG 871 ... recommended MS-DRG 872
    - provider assigned DRG 193 ... revised DRG 194
    """
    pair_patterns = [
        r"(?:changed|revised|downgraded|downcoded)?\s*from\s+(?:MS\s*[- ]?\s*)?DRG\s*#?\s*(?P<before>[O0]*\d{3,5})(?P<before_desc>[^\n.;]{0,120}?)\s+(?:to|into)\s+(?:MS\s*[- ]?\s*)?DRG\s*#?\s*(?P<after>[O0]*\d{3,5})(?P<after_desc>[^\n.;]{0,120})",
        r"(?:billed|submitted|reported|assigned|provider assigned|original(?:ly)? billed|requested)\s+(?:as\s+)?(?:MS\s*[- ]?\s*)?DRG\s*#?\s*(?P<before>[O0]*\d{3,5})(?P<before_desc>[^\n.;]{0,160}?).{0,300}?(?:recommended|revised|changed|downgraded|downcoded|approved)\s+(?:to\s+|as\s+)?(?:MS\s*[- ]?\s*)?DRG\s*#?\s*(?P<after>[O0]*\d{3,5})(?P<after_desc>[^\n.;]{0,120})",
        r"(?:MS\s*[- ]?\s*)?DRG\s*#?\s*(?P<before>[O0]*\d{3,5})(?P<before_desc>[^\n.;]{0,160}?).{0,300}?(?:recommended|revised|changed|downgraded|downcoded)\s+(?:to\s+|as\s+)?(?:MS\s*[- ]?\s*)?DRG\s*#?\s*(?P<after>[O0]*\d{3,5})(?P<after_desc>[^\n.;]{0,120})",
    ]

    for pattern in pair_patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
        if match:
            before = make_drg_value(match.group("before"), match.groupdict().get("before_desc"))
            after = make_drg_value(match.group("after"), match.groupdict().get("after_desc"))

            if before and after and before != after:
                return before, after

    return None, None


def extract_drg_before_rule_based(text: str) -> Optional[str]:
    patterns = [
        r"(?:provider assigned|assigned|billed|submitted|reported|requested|original(?:ly)? billed)\s+(?:as\s+)?((?:MS\s*[- ]?\s*)?DRG\s*#?\s*[O0]*\d{3,5}[^\n.;]{0,120})",
        r"(?:original|billed|submitted|requested)\s+(?:MS\s*[- ]?\s*)?DRG\s*[:#-]?\s*([O0]*\d{3,5}[^\n.;]{0,120})",
    ]

    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            value = match.group(1)
            if re.match(r"[O0]*\d{3,5}", value.strip()):
                value = "DRG " + value
            return normalize_drg_value(value)

    return None


def extract_drg_after_rule_based(text: str) -> Optional[str]:
    patterns = [
        r"(?:recommended|revised|changed|downgraded|downcoded|approved)\s+(?:to\s+|as\s+)?((?:MS\s*[- ]?\s*)?DRG\s*#?\s*[O0]*\d{3,5}[^\n.;]{0,120})",
        r"(?:recommended|revised|approved)\s+(?:MS\s*[- ]?\s*)?DRG\s*[:#-]?\s*([O0]*\d{3,5}[^\n.;]{0,120})",
    ]

    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            value = match.group(1)
            if re.match(r"[O0]*\d{3,5}", value.strip()):
                value = "DRG " + value
            return normalize_drg_value(value)

    return None


def extract_drg_before_after_rule_based(text: str) -> tuple[Optional[str], Optional[str]]:
    before, after = extract_drg_table_before_after_rule_based(text)

    if not before or not after:
        pair_before, pair_after = extract_drg_pair_rule_based(text)
        before = before or pair_before
        after = after or pair_after

    if not before:
        before = extract_drg_before_rule_based(text)

    if not after:
        after = extract_drg_after_rule_based(text)

    return before, after


def extract_drg_code_from_value(value: Optional[str]) -> Optional[str]:
    value = normalize_drg_value(value)

    if not value:
        return None

    match = re.search(r"\b(?:MS-?\s*)?DRG\s*#?\s*(?P<code>[O0]*\d{3,5})\b", value, flags=re.IGNORECASE)
    if not match:
        return None

    return normalize_drg_code(match.group("code"))


def drg_value_supported(value, text: str) -> bool:
    """Validate DRG values by checking that the DRG number appears near DRG/table context in source text."""
    value = normalize_drg_value(value)

    if value is None:
        return True

    code = extract_drg_code_from_value(value)
    if not code:
        return value_appears_in_text(value, text)

    # First accept explicit DRG 438 / MS-DRG 438 text.
    explicit_pattern = rf"\b(?:MS\s*[- ]?\s*)?DRG\s*#?\s*0*{re.escape(code)}\b"
    if re.search(explicit_pattern, text, flags=re.IGNORECASE):
        return True

    # Then accept table rows where the code appears without the word DRG on the same line,
    # as long as nearby context is clearly a DRG table.
    for match in re.finditer(rf"\b0*{re.escape(code)}\b", text):
        window = text[max(0, match.start() - 300): match.end() + 300]
        if re.search(r"(?i)\bDRG\b|DRG\s+Description|original\s+codes?\s+billed|new\s+coding\s+assignment", window):
            return True

    return False

llm_template = """
You classify healthcare denial letters.

Return ONLY valid JSON.
Do not include markdown.
Do not include explanations outside the JSON.

Rules:
- Do not invent patient name, account number, service dates, or claim number.
- Those fields are handled separately by Python and should remain null here.
- Only infer denial_type if the document text supports it.
- before_value should be the original/billed/requested diagnosis, procedure, code, level of care, or general value being denied, if clearly stated.
- after_value should be the payer-recommended/revised/approved diagnosis, procedure, code, level of care, or general replacement value, if clearly stated.
- drg_before_value should be the original/billed/requested MS-DRG or DRG value, if clearly stated. Include the DRG number and short description if present.
- drg_after_value should be the payer-recommended/revised/approved MS-DRG or DRG value, if clearly stated. Include the DRG number and short description if present.
- If this is not a DRG denial/downgrade, set drg_before_value and drg_after_value to null.
- policy_type should be Medicare, Medicaid, Commercial, Medicare Advantage, Managed Medicaid, or null if unclear.
- summary should briefly explain what was denied.
- Use null when unclear.

Return this JSON shape:
{{
  "denial_type": null,
  "before_value": null,
  "after_value": null,
  "drg_before_value": null,
  "drg_after_value": null,
  "policy_type": null,
  "summary": null
}}

Document text:
{case_text}
"""

prompt = ChatPromptTemplate.from_template(llm_template)
llm_chain = prompt | extract_model


def parse_json_response(raw_response: str) -> dict:
    match = re.search(r"\{.*\}", raw_response, re.DOTALL)

    if not match:
        return {}

    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}


def extract_denial_info(case_text: str) -> DenialExtraction:
    if len(case_text.strip()) < 200:
        raise ValueError(
            "Very little text was extracted from the denial file. "
            "The PDF may be scanned/image-based and may need OCR."
        )

    normalized_text = normalize_ocr_text(case_text)
    header_fields = extract_header_fields(normalized_text)

    # Rule-based extraction first. The LLM can fill gaps, but cannot override reliable values.
    rule_denial_type = extract_denial_type_rule_based(normalized_text)
    rule_before_value = extract_before_value_rule_based(normalized_text)
    rule_after_value = extract_after_value_rule_based(normalized_text)
    rule_drg_before_value, rule_drg_after_value = extract_drg_before_after_rule_based(normalized_text)
    rule_policy_type = infer_policy_type(normalized_text, header_fields.get("provider_name"))

    raw_llm_response = llm_chain.invoke({"case_text": normalized_text[:12000]})
    llm_fields = parse_json_response(raw_llm_response)

    # Normalize all LLM-filled values before validation/model creation.
    # This prevents crashes when the local model returns nested JSON for a field
    # that should be a plain string.
    llm_fields = {key: clean_value(value) for key, value in llm_fields.items()}
    llm_fields["drg_before_value"] = normalize_drg_value(llm_fields.get("drg_before_value"))
    llm_fields["drg_after_value"] = normalize_drg_value(llm_fields.get("drg_after_value"))

    result = {
        "patient_name": header_fields.get("patient_name"),
        "patient_account_number": header_fields.get("patient_account_number"),
        "service_date_start": header_fields.get("service_date_start"),
        "service_date_end": header_fields.get("service_date_end"),
        "claim_number": header_fields.get("claim_number"),
        "provider_name": header_fields.get("provider_name"),
        "denial_type": rule_denial_type or llm_fields.get("denial_type"),
        "drg_before_value": rule_drg_before_value or llm_fields.get("drg_before_value"),
        "drg_after_value": rule_drg_after_value or llm_fields.get("drg_after_value"),
        "before_value": rule_before_value or llm_fields.get("before_value") or rule_drg_before_value or llm_fields.get("drg_before_value"),
        "after_value": rule_after_value or llm_fields.get("after_value") or rule_drg_after_value or llm_fields.get("drg_after_value"),
        "policy_type": rule_policy_type or llm_fields.get("policy_type"),
        "summary": llm_fields.get("summary"),
    }

    # Final safety pass: keep every extracted field as a scalar string/null.
    for key in list(result.keys()):
        if key in {"drg_before_value", "drg_after_value"}:
            result[key] = normalize_drg_value(result.get(key))
        else:
            result[key] = clean_value(result.get(key))

    if not result["summary"]:
        before = result.get("before_value")
        after = result.get("after_value")
        denial_type = result.get("denial_type") or "denial/review"

        drg_before = result.get("drg_before_value")
        drg_after = result.get("drg_after_value")

        if drg_before and drg_after:
            result["summary"] = f"The letter describes a {denial_type} involving a DRG change from {drg_before} to {drg_after}."
        elif before:
            result["summary"] = f"The letter describes a {denial_type} involving {before}."
        elif after:
            result["summary"] = f"The letter describes a {denial_type}; payer finding: {after}."
        else:
            result["summary"] = f"The letter describes a {denial_type}."

    # Strict validation for identifiers/dates. These must appear in source text.
    strict_fields = [
        "patient_account_number",
        "service_date_start",
        "service_date_end",
        "claim_number",
    ]

    for field in strict_fields:
        if not value_appears_in_text(result.get(field), normalized_text):
            result[field] = None

    for field in ["drg_before_value", "drg_after_value"]:
        if not drg_value_supported(result.get(field), normalized_text):
            result[field] = None

    if result.get("before_value") and not value_appears_in_text(result.get("before_value"), normalized_text):
        # Keep generic before_value only if it is directly supported or duplicates a validated DRG value.
        if result.get("before_value") != result.get("drg_before_value"):
            result["before_value"] = result.get("drg_before_value")

    if result.get("after_value") and not value_appears_in_text(result.get("after_value"), normalized_text):
        # Keep generic after_value only if it is directly supported or duplicates a validated DRG value.
        if result.get("after_value") != result.get("drg_after_value"):
            result["after_value"] = result.get("drg_after_value")

    # Patient names in OCR may have minor leading OCR noise, so validate loosely.
    patient_name = result.get("patient_name")
    if patient_name:
        name_parts = [part for part in patient_name.split() if len(part) > 1]
        if not all(part.lower() in normalized_text.lower() for part in name_parts[-2:]):
            result["patient_name"] = None

    return DenialExtraction.model_validate(result)
