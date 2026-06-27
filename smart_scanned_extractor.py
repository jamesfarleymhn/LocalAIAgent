from __future__ import annotations

import contextlib
import os
import re
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from case_review import build_case_review
from final_case import build_final_case_json
from json_utils import json_dumps
from privacy import stable_file_id
from progress import Progress
from schemas import Evidence, ExtractedField, LoadedCase, PageText, to_plain_json


DATE_RE = re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b")
DRG_CODE_RE = re.compile(r"(?<![\d/])([O0]*\d{3,5})(?![\d/])")
ICD_PCS_RE = re.compile(r"\b[A-HJ-NP-Z0-9]{7}\b")
MONEY_RE = re.compile(r"(?<!\w)\$\s?\d[\d,]*(?:\.\d{2})?\b")


@dataclass
class OCRToken:
    text: str
    conf: float
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2

    @property
    def height(self) -> float:
        return max(1.0, self.y2 - self.y1)


@dataclass
class OCRLine:
    text: str
    tokens: list[OCRToken]
    x1: float
    y1: float
    x2: float
    y2: float
    conf: float

    @property
    def lower(self) -> str:
        return self.text.lower()


@contextlib.contextmanager
def _suppress_library_output():
    with open(os.devnull, "w", encoding="utf-8") as devnull:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            yield


_easyocr_reader = None


def get_easyocr_reader(progress: Progress | None = None):
    global _easyocr_reader
    if _easyocr_reader is None:
        if progress:
            progress.log("Loading EasyOCR reader for scanned-document extraction...")
        with _suppress_library_output():
            try:
                import easyocr
                _easyocr_reader = easyocr.Reader(["en"], gpu=False, verbose=False)
            except TypeError:
                import easyocr
                _easyocr_reader = easyocr.Reader(["en"], gpu=False)
        if progress:
            progress.log("EasyOCR reader loaded.")
    return _easyocr_reader


def _box_bounds(box: Any) -> tuple[float, float, float, float]:
    xs = [float(p[0]) for p in box]
    ys = [float(p[1]) for p in box]
    return min(xs), min(ys), max(xs), max(ys)


def _clean_text(text: Any) -> str | None:
    if text is None:
        return None
    text = re.sub(r"\s+", " ", str(text)).strip(" \t\r\n:;,-|[]{}")
    if not text:
        return None
    return text


def _normalize_ocr(text: str) -> str:
    fixes = {
        "billed were": "billed were",
        "bllled": "billed",
        "billed were;": "billed were:",
        "Claim number(s};": "Claim number(s):",
        "Request ID (Humana use only}": "Request ID (Humana use only):",
        "Humana:" : "Humana",
        "codlng": "coding",
        "asslgnment": "assignment",
    }
    for bad, good in fixes.items():
        text = text.replace(bad, good)
    return text.replace("–", "-").replace("—", "-")


def render_pdf_page(path: Path, page_number: int, *, zoom: float) -> np.ndarray:
    import fitz
    with fitz.open(str(path)) as doc:
        page = doc[page_number - 1]
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
        return np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)


def count_pdf_pages(path: Path) -> int:
    import fitz
    with fitz.open(str(path)) as doc:
        return len(doc)


def ocr_page_lines(path: Path, page_number: int, *, zoom: float, progress: Progress | None = None) -> tuple[list[OCRLine], str]:
    image = render_pdf_page(path, page_number, zoom=zoom)
    reader = get_easyocr_reader(progress)
    if progress:
        progress.log(f"OCR/layout reading page {page_number}...")
    with _suppress_library_output():
        results = reader.readtext(image, detail=1, paragraph=False)

    tokens: list[OCRToken] = []
    for item in results:
        if len(item) < 2:
            continue
        box, text = item[0], _clean_text(item[1])
        conf = float(item[2]) if len(item) > 2 else 0.5
        if not text:
            continue
        x1, y1, x2, y2 = _box_bounds(box)
        tokens.append(OCRToken(text=_normalize_ocr(text), conf=conf, x1=x1, y1=y1, x2=x2, y2=y2))

    if not tokens:
        return [], ""

    heights = [t.height for t in tokens]
    threshold = max(12.0, statistics.median(heights) * 0.65)
    tokens_sorted = sorted(tokens, key=lambda t: (t.cy, t.x1))
    groups: list[list[OCRToken]] = []
    for tok in tokens_sorted:
        placed = False
        for group in groups:
            group_cy = statistics.mean(t.cy for t in group)
            if abs(tok.cy - group_cy) <= threshold:
                group.append(tok)
                placed = True
                break
        if not placed:
            groups.append([tok])

    lines: list[OCRLine] = []
    for group in groups:
        group = sorted(group, key=lambda t: t.x1)
        text = " ".join(t.text for t in group)
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            continue
        lines.append(OCRLine(text=text, tokens=group, x1=min(t.x1 for t in group), y1=min(t.y1 for t in group), x2=max(t.x2 for t in group), y2=max(t.y2 for t in group), conf=sum(t.conf for t in group)/len(group)))
    lines = sorted(lines, key=lambda line: (line.y1, line.x1))
    page_text = "\n".join(line.text for line in lines)
    return lines, page_text


def _field(name: str, value: Any, *, category: str, source_id: str, source_name: str | None, page_number: int, evidence: str | None, confidence: float, note: str = "Extracted by scanned-document layout parser.") -> ExtractedField | None:
    text = _clean_text(value)
    if not text:
        return None
    return ExtractedField(
        name=name,
        value=text,
        category=category,
        confidence=confidence,
        evidence=Evidence(source_id=source_id, source_name=source_name, page_number=page_number, excerpt=evidence),
        validated=True,
        validation_note=note,
    )


def _add(fields: list[ExtractedField], *args, **kwargs) -> None:
    f = _field(*args, **kwargs)
    if f is not None:
        fields.append(f)


def normalize_drg_code(raw: str | None) -> str | None:
    raw = _clean_text(raw)
    if not raw:
        return None
    raw = raw.replace("O", "0").replace("o", "0")
    digits = re.sub(r"\D", "", raw)
    if not digits:
        return None
    if len(digits) > 3 and digits.startswith("0"):
        digits = digits.lstrip("0") or "0"
    if len(digits) > 3:
        return None
    return digits.zfill(3) if len(digits) < 3 else digits


def _clean_drg_desc(text: str, code_match: re.Match) -> str | None:
    tail = text[code_match.end():]
    tail = re.sub(r"(?i)\bDRG\b|\bDescription\b|\bDRG Description\b", " ", tail)
    tail = re.sub(r"[^A-Za-z0-9,.'/&()\- ]+", " ", tail)
    tail = re.sub(r"\s+", " ", tail).strip(" :-|[]{}")
    if not tail or not re.search(r"[A-Za-z]", tail):
        return None
    # Stop if another row/code appears later.
    tail = re.split(r"\s+(?=[O0]*\d{3,5}\s+[A-Z])", tail, maxsplit=1)[0]
    return tail[:160].strip(" :-|[]{}") or None


def _line_has_header_noise(text: str) -> bool:
    low = text.lower()
    return ("drg description" in low and not DRG_CODE_RE.search(low)) or low.strip() in {"drg", "drg description"}


def extract_drg_from_lines(lines: list[OCRLine], start_idx: int, *, stop_words: list[str], window: int = 14) -> tuple[str | None, str | None, str | None]:
    stop_words_l = [w.lower() for w in stop_words]
    for j in range(start_idx + 1, min(len(lines), start_idx + 1 + window)):
        line = lines[j]
        low = line.lower
        if any(w in low for w in stop_words_l):
            break
        if _line_has_header_noise(line.text):
            continue
        # Combine adjacent short rows because OCR sometimes separates the code and description.
        candidates = [line.text]
        if j + 1 < len(lines):
            candidates.append(line.text + " " + lines[j + 1].text)
        for candidate in candidates:
            # Prefer numeric DRG at beginning or after a DRG header; reject dates/years.
            for m in DRG_CODE_RE.finditer(candidate):
                raw = m.group(1)
                code = normalize_drg_code(raw)
                if not code:
                    continue
                # DRG code should be near the start of the row or followed by a strong description.
                desc = _clean_drg_desc(candidate, m)
                if desc or m.start() < 20:
                    return code, raw, desc
    return None, None, None


def find_drg_tables(lines: list[OCRLine], *, page_number: int, source_id: str, source_name: str | None) -> list[ExtractedField]:
    fields: list[ExtractedField] = []
    original_idx = None
    new_idx = None
    for i, line in enumerate(lines):
        low = line.lower
        if "original" in low and "code" in low and "billed" in low:
            original_idx = i
        if "new" in low and "coding" in low and "assignment" in low:
            new_idx = i

    if original_idx is not None:
        code, raw, desc = extract_drg_from_lines(lines, original_idx, stop_words=["new coding assignment", "following review", "according to"], window=18)
        if code:
            value = f"DRG {code}" + (f" (raw {raw})" if raw and raw != code else "") + (f" - {desc}" if desc else "")
            ev_lines = [lines[original_idx].text]
            ev_lines.extend(line.text for line in lines[original_idx+1: min(len(lines), original_idx+7)])
            _add(fields, "original_drg", value, category="coding", source_id=source_id, source_name=source_name, page_number=page_number, evidence=" | ".join(ev_lines), confidence=0.97, note="Extracted from DRG table following 'original codes billed were'.")

    if new_idx is not None:
        code, raw, desc = extract_drg_from_lines(lines, new_idx, stop_words=["following review", "according to", "review findings", "appeal"], window=18)
        if code:
            value = f"DRG {code}" + (f" (raw {raw})" if raw and raw != code else "") + (f" - {desc}" if desc else "")
            ev_lines = [lines[new_idx].text]
            ev_lines.extend(line.text for line in lines[new_idx+1: min(len(lines), new_idx+7)])
            _add(fields, "updated_drg", value, category="coding", source_id=source_id, source_name=source_name, page_number=page_number, evidence=" | ".join(ev_lines), confidence=0.97, note="Extracted from DRG table following 'new coding assignment is'.")
    return fields


def find_patient_claim_summary(page_text: str, *, page_number: int, source_id: str, source_name: str | None) -> list[ExtractedField]:
    fields: list[ExtractedField] = []
    text = re.sub(r"\s+", " ", page_text)
    if "Review Findings Summary" not in text and "review findings summary" not in text.lower():
        return fields
    dates = DATE_RE.findall(text)
    if len(dates) < 2:
        return fields
    # Try a Humana summary row. It usually has request id, patient, member, DOB, account, DOS, claim, legal entity.
    pattern = re.compile(
        r"(?P<request>\b\d{5,}\b)\s+"
        r"(?P<patient>[A-Z][A-Z'\-]+(?:\s+[A-Z][A-Z'\-]+){1,4})\s+"
        r"(?P<member>[A-Z0-9]{6,})\s+"
        r"(?P<dob>\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\s+"
        r"(?P<account>\d{4,})\s+"
        r"(?P<dos1>\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\s*(?:-|to)?\s*(?P<dos2>\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\s+"
        r"(?P<claim>\d{6,})\s+"
        r"(?P<legal>.+)$",
        flags=re.I,
    )
    # Search after the header if possible.
    start = text.lower().find("review findings summary")
    search_text = text[start:start+2000] if start >= 0 else text
    m = pattern.search(search_text)
    if not m:
        return fields
    evidence = m.group(0)[:350]
    _add(fields, "patient_name", m.group("patient"), category="patient", source_id=source_id, source_name=source_name, page_number=page_number, evidence=evidence, confidence=0.9, note="Parsed from Review Findings Summary value row.")
    _add(fields, "member_id", m.group("member"), category="patient", source_id=source_id, source_name=source_name, page_number=page_number, evidence=evidence, confidence=0.9, note="Parsed from Review Findings Summary value row.")
    _add(fields, "date_of_birth", m.group("dob"), category="patient", source_id=source_id, source_name=source_name, page_number=page_number, evidence=evidence, confidence=0.9, note="Parsed from Review Findings Summary value row.")
    _add(fields, "account_number", m.group("account"), category="patient", source_id=source_id, source_name=source_name, page_number=page_number, evidence=evidence, confidence=0.9, note="Parsed from Review Findings Summary value row.")
    _add(fields, "date_of_service", f"{m.group('dos1')} - {m.group('dos2')}", category="claim", source_id=source_id, source_name=source_name, page_number=page_number, evidence=evidence, confidence=0.9, note="Parsed from Review Findings Summary value row.")
    _add(fields, "claim_number", m.group("claim"), category="claim", source_id=source_id, source_name=source_name, page_number=page_number, evidence=evidence, confidence=0.9, note="Parsed from Review Findings Summary value row.")
    legal = re.split(r"\b(?:provider assigned|drg table|following review)\b", m.group("legal"), flags=re.I)[0]
    _add(fields, "provider_or_legal_entity", legal, category="parties", source_id=source_id, source_name=source_name, page_number=page_number, evidence=evidence, confidence=0.85, note="Parsed from Review Findings Summary value row.")
    return fields


def find_coding_findings(page_text: str, *, page_number: int, source_id: str, source_name: str | None) -> list[ExtractedField]:
    fields: list[ExtractedField] = []
    text = re.sub(r"\s+", " ", page_text)
    low = text.lower()
    # Payer / reviewer. Keep this simple and deterministic.
    if "humana" in low:
        _add(fields, "payer_or_reviewer", "Humana", category="parties", source_id=source_id, source_name=source_name, page_number=page_number, evidence="Humana appears on the page.", confidence=0.86)
    if "overpaid" in low or "overpayment" in low:
        ev = _excerpt_near(text, "overpaid") or _excerpt_near(text, "overpayment")
        _add(fields, "denial_type", "Post-payment coding denial / overpayment review", category="denial", source_id=source_id, source_name=source_name, page_number=page_number, evidence=ev, confidence=0.8)
        _add(fields, "payer_rationale", ev, category="denial", source_id=source_id, source_name=source_name, page_number=page_number, evidence=ev, confidence=0.7)
    # Provider assigned code and not-supported finding.
    for m in re.finditer(r"provider assigned\s+(ICD-10-[A-Z]+\s+code\s+[A-Z0-9.]{3,})", text, flags=re.I):
        _add(fields, "before_non_drg_code", m.group(1), category="coding", source_id=source_id, source_name=source_name, page_number=page_number, evidence=_excerpt_around(text, m.start()), confidence=0.88)
    for m in re.finditer(r"(code\s+[A-Z0-9.]{3,}\s+is\s+not\s+supported[^.]{0,160})", text, flags=re.I):
        _add(fields, "after_non_drg_code_or_finding", m.group(1), category="coding", source_id=source_id, source_name=source_name, page_number=page_number, evidence=_excerpt_around(text, m.start()), confidence=0.9)
        _add(fields, "not_supported_code", m.group(1), category="coding", source_id=source_id, source_name=source_name, page_number=page_number, evidence=_excerpt_around(text, m.start()), confidence=0.9)
    # If OCR splits the phrase, scan for ICD-10-PCS code close to not supported.
    if "not supported" in low:
        for code in ICD_PCS_RE.findall(text):
            if code.lower() not in {"humana", "review"}:
                ev = _excerpt_near(text, code)
                if ev and "not supported" in ev.lower():
                    _add(fields, "not_supported_code", f"code {code} is not supported by documentation", category="coding", source_id=source_id, source_name=source_name, page_number=page_number, evidence=ev, confidence=0.82)
    money = MONEY_RE.search(text)
    if money and ("overpayment" in low or "refund" in low or "amount" in low):
        _add(fields, "amount", money.group(0), category="claim", source_id=source_id, source_name=source_name, page_number=page_number, evidence=_excerpt_around(text, money.start()), confidence=0.75)
    return fields


def _excerpt_around(text: str, index: int, radius: int = 220) -> str:
    return re.sub(r"\s+", " ", text[max(0, index-radius): index+radius]).strip()


def _excerpt_near(text: str, needle: str, radius: int = 220) -> str | None:
    i = text.lower().find(needle.lower())
    return _excerpt_around(text, i, radius) if i >= 0 else None


def _add_manual_summary(final_case: dict[str, Any]) -> None:
    coding = final_case.get("coding_change", {})
    orig = (coding.get("original_drg") or {}).get("value")
    upd = (coding.get("updated_drg") or {}).get("value")
    unsupported = coding.get("unsupported_procedure_or_code_findings") or []
    parts = []
    if orig or upd:
        parts.append(f"The denial involves a DRG reassignment from {orig or 'an unresolved original DRG'} to {upd or 'an unresolved updated DRG'}")
    if unsupported:
        code = unsupported[0].get("code") or "a procedure/code"
        parts.append(f"the payer also states that {code} is not supported by the submitted documentation")
    if parts:
        summary = "; and ".join(parts) + "."
        final_case.setdefault("case_summary", {})["one_sentence_summary"] = summary
        final_case.setdefault("coding_change", {})["plain_english_summary"] = summary


def extract_scanned_denial_pdf(
    case_paths: list[str],
    *,
    include_source_names: bool = False,
    include_page_text: bool = False,
    zoom: float = 2.0,
    max_pages: int | None = None,
    progress: Progress | None = None,
) -> dict[str, Any]:
    """Document-intelligence extraction for scanned denial PDFs.

    This path intentionally avoids sending full pages to a model. It OCRs all pages once,
    preserves line layout, finds important sections automatically, and produces one
    concise resolved case object. It is designed for scanned payer denial letters where
    text-layer extraction and full-page vision calls are unreliable/slow.
    """
    progress = progress or Progress(enabled=True)
    all_fields: list[ExtractedField] = []
    pages: list[PageText] = []
    warnings: list[str] = []
    source_ids: list[str] = []
    raw_page_debug: list[dict[str, Any]] = []

    for file_index, raw_path in enumerate(case_paths, start=1):
        path = Path(raw_path).expanduser().resolve()
        if not path.exists():
            warnings.append(f"File not found: {path}")
            continue
        if path.suffix.lower() != ".pdf":
            warnings.append(f"Scanned extraction currently supports PDFs only. Skipped: {path.name}")
            continue
        source_id = stable_file_id(path)
        source_ids.append(source_id)
        source_name = path.name if include_source_names else None
        try:
            total_pages = count_pdf_pages(path)
        except Exception as exc:
            warnings.append(f"Could not open PDF {path.name}: {type(exc).__name__}: {exc}")
            continue
        page_count = min(total_pages, max_pages) if max_pages else total_pages
        progress.log(f"Scanned extraction file {file_index}/{len(case_paths)}: {path.name}")
        progress.log(f"Reading {page_count}/{total_pages} page(s) with OCR layout detection once. No per-page LLM calls.")
        for page_number in range(1, page_count + 1):
            try:
                lines, page_text = ocr_page_lines(path, page_number, zoom=zoom, progress=progress)
                pages.append(PageText(source_id=source_id, source_name=source_name, page_number=page_number, text=page_text, extraction_method="easyocr_layout"))
                all_fields.extend(find_drg_tables(lines, page_number=page_number, source_id=source_id, source_name=source_name))
                all_fields.extend(find_patient_claim_summary(page_text, page_number=page_number, source_id=source_id, source_name=source_name))
                all_fields.extend(find_coding_findings(page_text, page_number=page_number, source_id=source_id, source_name=source_name))
                raw_page_debug.append({"page_number": page_number, "line_count": len(lines), "page_text_preview": page_text[:1200]})
            except Exception as exc:
                warnings.append(f"Page {page_number} extraction failed for {path.name}: {type(exc).__name__}: {exc}")

    loaded = LoadedCase(document_id=";".join(source_ids) if source_ids else "unknown", pages=pages, warnings=warnings)
    extraction = {
        "core": {},
        "fields_by_category": {},
        "all_fields": [to_plain_json(field) for field in all_fields],
        "scanned_page_debug": raw_page_debug,
    }
    result = {
        "schema_version": "4.0-scanned-document-intelligence",
        "analysis_mode": "scanned-extract",
        "privacy": {
            "phi_in_source_code": False,
            "case_text_handling": "Submitted scanned PDF pages are OCRed at runtime. The case is not ingested into Chroma.",
            "raw_page_text_included": include_page_text,
        },
        "document": {
            "document_id": loaded.document_id,
            "page_count": loaded.page_count,
            "ocr_layout_zoom": zoom,
            "source_names_included": include_source_names,
        },
        "structured_extraction": extraction,
        "summary": {
            "plain_english_summary": None,
            "key_denial_rationale": None,
            "recommended_next_steps": [],
            "missing_or_uncertain_information": [],
        },
        "warnings": warnings,
    }
    result["final_case"] = build_final_case_json(result)
    _add_manual_summary(result["final_case"])
    result["case_review"] = build_case_review(result)
    if include_page_text:
        result["document_pages"] = [to_plain_json(page) for page in loaded.pages]
    return result
