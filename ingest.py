from pathlib import Path
import hashlib
import shutil
import pandas as pd
from pypdf import PdfReader
import docx2txt
from collections import defaultdict

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma


DATA_DIR = Path("knowledge_base")
DB_DIR = Path("./chroma_denials_db")
COLLECTION_NAME = "denials_knowledge_base"

EMBEDDING_MODEL = "bge-m3"

# Rebuild the DB from the current knowledge_base contents each time ingest.py runs.
# This prevents deleted files/folders from remaining searchable in Chroma.
RESET_DB_ON_INGEST = True

# These folders should never be treated as reusable knowledge-base evidence.
# User-submitted cases should be loaded by case_loader.py at runtime, not ingested into RAG.
EXCLUDED_KB_FOLDERS = {
    "denial_letters",
    "submitted_denials",
    "submitted_cases",
    "uploaded_cases",
    "case_files",
    "patient_files",
    "user_input",
}


def get_doc_category(file_path: Path) -> str:
    """
    Use the first folder under knowledge_base as the document category.

    Example:
      knowledge_base/payer_policies/humana_policy.pdf -> payer_policies
      knowledge_base/guides/coding/foo.pdf -> guides
    """
    try:
        relative = file_path.relative_to(DATA_DIR)
        if len(relative.parts) > 1:
            return relative.parts[0]
    except ValueError:
        pass

    return file_path.parent.name


def should_skip_file(file_path: Path) -> bool:
    try:
        relative_parts = [part.lower() for part in file_path.relative_to(DATA_DIR).parts]
    except ValueError:
        relative_parts = [part.lower() for part in file_path.parts]

    return any(part in EXCLUDED_KB_FOLDERS for part in relative_parts)


def get_file_hash(file_path: Path) -> str:
    """Create a stable hash so we can identify the source file."""
    hasher = hashlib.sha256()

    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)

    return hasher.hexdigest()


def base_metadata(file_path: Path, file_type: str, file_hash: str) -> dict:
    return {
        "source": str(file_path.resolve()),
        "file_name": file_path.name,
        "file_type": file_type,
        "file_hash": file_hash,
        "folder": file_path.parent.name,
        "doc_category": get_doc_category(file_path),
    }


def load_pdf(file_path: Path) -> list[Document]:
    docs = []
    file_hash = get_file_hash(file_path)

    reader = PdfReader(str(file_path))

    try:
        page_labels = reader.page_labels
    except Exception:
        page_labels = []

    for page_index, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""

        if not text.strip():
            continue

        page_label = None
        if page_labels and len(page_labels) >= page_index:
            page_label = page_labels[page_index - 1]

        metadata = base_metadata(file_path, "pdf", file_hash)
        metadata.update(
            {
                "page": page_index,
                "page_label": page_label,
            }
        )

        docs.append(
            Document(
                page_content=(
                    f"File: {file_path.name}\n"
                    f"PDF page: {page_index}\n\n"
                    f"{text}"
                ),
                metadata=metadata,
            )
        )

    return docs


def load_docx(file_path: Path) -> list[Document]:
    file_hash = get_file_hash(file_path)

    text = docx2txt.process(str(file_path)) or ""

    if not text.strip():
        return []

    return [
        Document(
            page_content=(
                f"File: {file_path.name}\n\n"
                f"{text}"
            ),
            metadata=base_metadata(file_path, "word", file_hash),
        )
    ]


def load_excel(file_path: Path) -> list[Document]:
    """
    Load Excel files.

    For denial work, this is usually better than treating the spreadsheet
    as one big blob of text. Each row becomes searchable evidence.
    """
    docs = []
    file_hash = get_file_hash(file_path)

    excel_file = pd.ExcelFile(file_path)

    for sheet_name in excel_file.sheet_names:
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        df = df.fillna("")

        for row_index, row in df.iterrows():
            row_text_parts = []

            for column_name, value in row.items():
                value = str(value).strip()
                if value:
                    row_text_parts.append(f"{column_name}: {value}")

            if not row_text_parts:
                continue

            page_content = (
                f"Excel file: {file_path.name}\n"
                f"Sheet: {sheet_name}\n"
                f"Row: {row_index + 2}\n\n"
                + "\n".join(row_text_parts)
            )

            metadata = base_metadata(file_path, "excel", file_hash)
            metadata.update(
                {
                    "sheet_name": sheet_name,
                    "row_number": row_index + 2,
                }
            )

            docs.append(
                Document(
                    page_content=page_content,
                    metadata=metadata,
                )
            )

    return docs


def load_text(file_path: Path) -> list[Document]:
    file_hash = get_file_hash(file_path)

    text = file_path.read_text(encoding="utf-8", errors="ignore")

    if not text.strip():
        return []

    return [
        Document(
            page_content=(
                f"File: {file_path.name}\n\n"
                f"{text}"
            ),
            metadata=base_metadata(file_path, "text", file_hash),
        )
    ]


def load_documents_from_folder(folder: Path) -> list[Document]:
    all_docs = []

    for file_path in folder.rglob("*"):
        if file_path.is_dir():
            continue

        if should_skip_file(file_path):
            print(f"Skipping patient-specific/user-input folder: {file_path}")
            continue

        suffix = file_path.suffix.lower()

        print(f"Loading: {file_path}")

        try:
            if suffix == ".pdf":
                all_docs.extend(load_pdf(file_path))

            elif suffix in [".xlsx", ".xls"]:
                all_docs.extend(load_excel(file_path))

            elif suffix == ".docx":
                all_docs.extend(load_docx(file_path))

            elif suffix in [".txt", ".md"]:
                all_docs.extend(load_text(file_path))

            else:
                print(f"Skipping unsupported file type: {file_path}")

        except Exception as e:
            print(f"ERROR loading {file_path}: {e}")

    return all_docs


def reset_vector_db():
    if RESET_DB_ON_INGEST and DB_DIR.exists():
        print(f"Resetting vector database: {DB_DIR}")
        shutil.rmtree(DB_DIR)


def main():
    reset_vector_db()

    print("Loading documents...")
    raw_docs = load_documents_from_folder(DATA_DIR)

    print(f"Loaded {len(raw_docs)} raw documents/pages/rows.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=100,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = splitter.split_documents(raw_docs)

    chunk_counts_by_source = defaultdict(int)

    for chunk in chunks:
        source = chunk.metadata.get("source", "unknown")
        chunk_counts_by_source[source] += 1

        chunk.metadata["chunk_index"] = chunk_counts_by_source[source]

    print(f"Created {len(chunks)} chunks.")

    ids = [
        f"{chunk.metadata.get('file_hash', 'unknown')}_{chunk.metadata.get('chunk_index', i)}"
        for i, chunk in enumerate(chunks)
    ]

    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=str(DB_DIR),
        embedding_function=embeddings,
    )

    print("Adding chunks to Chroma in batches...")

    BATCH_SIZE = 32

    for start in range(0, len(chunks), BATCH_SIZE):
        end = start + BATCH_SIZE

        batch_chunks = chunks[start:end]
        batch_ids = ids[start:end]

        print(f"Adding chunks {start + 1} to {min(end, len(chunks))} of {len(chunks)}")

        vector_store.add_documents(
            documents=batch_chunks,
            ids=batch_ids,
        )

    print("Done. Vector database rebuilt from current knowledge_base contents.")


if __name__ == "__main__":
    main()
