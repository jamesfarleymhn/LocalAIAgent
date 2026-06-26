from __future__ import annotations

from pathlib import Path
from typing import Iterable

from config import CONFIG
from privacy import stable_file_id
from schemas import LoadedCase, PageText


SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md", ".xlsx", ".xls"}


def _import_or_raise(package_name: str, install_name: str | None = None):
    try:
        return __import__(package_name)
    except ImportError as exc:
        raise RuntimeError(f"Missing dependency '{install_name or package_name}'. Install requirements.txt.") from exc


def _source_name(path: Path, include_source_names: bool) -> str | None:
    return path.name if include_source_names else None


def _load_pdf_text_pages(path: Path, *, include_source_names: bool) -> list[PageText]:
    pypdf = _import_or_raise("pypdf")
    source_id = stable_file_id(path)
    reader = pypdf.PdfReader(str(path))
    pages: list[PageText] = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        pages.append(
            PageText(
                source_id=source_id,
                source_name=_source_name(path, include_source_names),
                page_number=page_number,
                text=text,
                extraction_method="pypdf_text_layer",
            )
        )

    return pages


def _ocr_pdf_pages(path: Path, page_numbers: Iterable[int], *, include_source_names: bool) -> dict[int, PageText]:
    """OCR selected pages only. Imports OCR packages lazily so normal PDFs do not fail startup."""
    try:
        fitz = _import_or_raise("fitz", "PyMuPDF")
        np = _import_or_raise("numpy")
        easyocr = _import_or_raise("easyocr")
    except RuntimeError as exc:
        warning = str(exc)
        return {
            page_number: PageText(
                source_id=stable_file_id(path),
                source_name=_source_name(path, include_source_names),
                page_number=page_number,
                text="",
                extraction_method="ocr_unavailable",
                warnings=[warning],
            )
            for page_number in page_numbers
        }

    reader = easyocr.Reader(["en"], gpu=False, verbose=False)
    pdf = fitz.open(str(path))
    source_id = stable_file_id(path)
    out: dict[int, PageText] = {}

    for page_number in page_numbers:
        zero_index = page_number - 1
        if zero_index < 0 or zero_index >= len(pdf):
            continue
        page = pdf[zero_index]
        pix = page.get_pixmap(matrix=fitz.Matrix(CONFIG.ocr_zoom, CONFIG.ocr_zoom), alpha=False)
        image = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        ocr_lines = reader.readtext(image, detail=0, paragraph=True)
        text = "\n".join(str(line) for line in ocr_lines)
        out[page_number] = PageText(
            source_id=source_id,
            source_name=_source_name(path, include_source_names),
            page_number=page_number,
            text=text,
            extraction_method="easyocr",
        )

    return out


def load_pdf(path: Path, *, include_source_names: bool = False) -> list[PageText]:
    pages = _load_pdf_text_pages(path, include_source_names=include_source_names)
    low_text_pages = [
        page.page_number
        for page in pages
        if len((page.text or "").strip()) < CONFIG.min_pdf_page_text_chars
    ]

    if low_text_pages:
        ocr_pages = _ocr_pdf_pages(path, low_text_pages, include_source_names=include_source_names)
        for index, page in enumerate(pages):
            replacement = ocr_pages.get(page.page_number)
            if replacement and replacement.text.strip():
                pages[index] = replacement
            elif replacement and replacement.warnings:
                page.warnings.extend(replacement.warnings)

    return pages


def load_docx(path: Path, *, include_source_names: bool = False) -> list[PageText]:
    docx2txt = _import_or_raise("docx2txt")
    text = docx2txt.process(str(path)) or ""
    return [
        PageText(
            source_id=stable_file_id(path),
            source_name=_source_name(path, include_source_names),
            page_number=1,
            text=text,
            extraction_method="docx2txt",
        )
    ]


def load_spreadsheet(path: Path, *, include_source_names: bool = False) -> list[PageText]:
    pd = _import_or_raise("pandas")
    source_id = stable_file_id(path)
    excel = pd.ExcelFile(path)
    pages: list[PageText] = []
    for sheet_index, sheet_name in enumerate(excel.sheet_names, start=1):
        df = pd.read_excel(path, sheet_name=sheet_name).fillna("")
        lines = [f"Sheet: {sheet_name}"]
        for row_index, row in df.iterrows():
            cells = []
            for column_name, value in row.items():
                value = str(value).strip()
                if value:
                    cells.append(f"{column_name}: {value}")
            if cells:
                lines.append(f"Row {row_index + 2}: " + " | ".join(cells))
        pages.append(
            PageText(
                source_id=source_id,
                source_name=_source_name(path, include_source_names),
                page_number=sheet_index,
                text="\n".join(lines),
                extraction_method="pandas_excel",
            )
        )
    return pages


def load_text_file(path: Path, *, include_source_names: bool = False) -> list[PageText]:
    return [
        PageText(
            source_id=stable_file_id(path),
            source_name=_source_name(path, include_source_names),
            page_number=1,
            text=path.read_text(encoding="utf-8", errors="ignore"),
            extraction_method="plain_text",
        )
    ]


def load_case_files(paths: list[str | Path], *, include_source_names: bool = False) -> LoadedCase:
    all_pages: list[PageText] = []
    warnings: list[str] = []
    source_ids: list[str] = []

    for raw_path in paths:
        path = Path(raw_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported file type: {path.suffix}")

        source_ids.append(stable_file_id(path))
        if path.suffix.lower() == ".pdf":
            pages = load_pdf(path, include_source_names=include_source_names)
        elif path.suffix.lower() == ".docx":
            pages = load_docx(path, include_source_names=include_source_names)
        elif path.suffix.lower() in {".xlsx", ".xls"}:
            pages = load_spreadsheet(path, include_source_names=include_source_names)
        else:
            pages = load_text_file(path, include_source_names=include_source_names)

        for page in pages:
            warnings.extend(page.warnings)
        all_pages.extend(pages)

    document_id = "+".join(source_ids) if source_ids else "no-document"
    if not any(page.text.strip() for page in all_pages):
        warnings.append("No readable text was found. For scanned PDFs, install OCR dependencies and retry.")

    return LoadedCase(document_id=document_id, pages=all_pages, warnings=warnings)
