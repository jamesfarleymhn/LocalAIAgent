from __future__ import annotations

from pathlib import Path
import contextlib
import os
import warnings

import docx2txt
import easyocr
import fitz  # PyMuPDF
import numpy as np
import pandas as pd
from pypdf import PdfReader


# Hide noisy library warnings/messages in the normal app experience.
warnings.filterwarnings("ignore", message=".*pin_memory.*")
warnings.filterwarnings("ignore", category=UserWarning, module="torch.*")

_easyocr_reader = None


@contextlib.contextmanager
def _suppress_library_output():
    """Suppress noisy stdout/stderr from EasyOCR/Torch during normal runs."""
    with open(os.devnull, "w", encoding="utf-8") as devnull:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            yield


def get_easyocr_reader():
    """Load EasyOCR once. First run may download OCR model weights."""
    global _easyocr_reader

    if _easyocr_reader is None:
        with _suppress_library_output():
            try:
                _easyocr_reader = easyocr.Reader(["en"], gpu=False, verbose=False)
            except TypeError:
                # Older EasyOCR versions may not support verbose=False.
                _easyocr_reader = easyocr.Reader(["en"], gpu=False)

    return _easyocr_reader


def extract_pdf_text_with_pypdf(file_path: Path) -> str:
    """Extract text from PDFs that already contain a text layer."""
    reader = PdfReader(str(file_path))
    pages: list[str] = []

    for page_index, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""

        if text.strip():
            pages.append(f"\n--- PDF PAGE {page_index} ---\n{text}")

    return "\n".join(pages)


def extract_pdf_text_with_easyocr(file_path: Path, zoom: float = 2.5) -> str:
    """OCR scanned/image-based PDFs using only pip-installable Python packages."""
    reader = get_easyocr_reader()
    pdf = fitz.open(str(file_path))
    pages: list[str] = []

    for page_index in range(len(pdf)):
        page = pdf[page_index]
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)

        image = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
            pix.height,
            pix.width,
            pix.n,
        )

        with _suppress_library_output():
            results = reader.readtext(image, detail=0, paragraph=True)

        page_text = "\n".join(results)

        if page_text.strip():
            pages.append(f"\n--- OCR PDF PAGE {page_index + 1} ---\n{page_text}")

    return "\n".join(pages)


def extract_pdf_text(file_path: Path, *, min_text_length: int = 200) -> str:
    """
    Try normal PDF text extraction first; if that fails, fall back to OCR.
    This function intentionally does not print noisy debug messages.
    """
    normal_text = extract_pdf_text_with_pypdf(file_path)

    if len(normal_text.strip()) >= min_text_length:
        return normal_text

    return extract_pdf_text_with_easyocr(file_path)


def extract_docx_text(file_path: Path) -> str:
    text = docx2txt.process(str(file_path)) or ""
    return f"\n--- WORD DOCUMENT: {file_path.name} ---\n{text}"


def extract_excel_text(file_path: Path) -> str:
    parts: list[str] = []
    excel_file = pd.ExcelFile(file_path)

    for sheet_name in excel_file.sheet_names:
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        df = df.fillna("")

        parts.append(f"\n--- EXCEL FILE: {file_path.name}, SHEET: {sheet_name} ---")

        for row_index, row in df.iterrows():
            row_values: list[str] = []

            for column_name, value in row.items():
                value = str(value).strip()

                if value:
                    row_values.append(f"{column_name}: {value}")

            if row_values:
                parts.append(f"Row {row_index + 2}: " + " | ".join(row_values))

    return "\n".join(parts)


def load_case_file(file_path: str) -> str:
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    suffix = path.suffix.lower()

    if suffix == ".pdf":
        return extract_pdf_text(path)

    if suffix == ".docx":
        return extract_docx_text(path)

    if suffix in [".xlsx", ".xls"]:
        return extract_excel_text(path)

    if suffix in [".txt", ".md"]:
        return path.read_text(encoding="utf-8", errors="ignore")

    raise ValueError(f"Unsupported file type: {suffix}")


def load_case_files(file_paths: list[str]) -> str:
    all_text: list[str] = []

    for file_path in file_paths:
        path = Path(file_path)
        text = load_case_file(file_path)
        all_text.append(f"\n\n===== CASE FILE: {path.name} =====\n{text}")

    return "\n".join(all_text)