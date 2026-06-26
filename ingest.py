from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from chunking import chunk_loaded_case
from config import CONFIG
from document_loader import SUPPORTED_EXTENSIONS, load_case_files
from privacy import validate_knowledge_path


def iter_knowledge_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for path in root.rglob("*"):
        if path.is_dir():
            continue
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        validate_knowledge_path(path, root)
        files.append(path)
    return files


def build_documents_for_chroma(files: list[Path]):
    try:
        from langchain_core.documents import Document
    except ImportError as exc:
        raise RuntimeError("Missing LangChain core dependency. Install requirements.txt.") from exc

    docs = []
    for file_path in files:
        loaded = load_case_files([file_path], include_source_names=False)
        for index, chunk in enumerate(chunk_loaded_case(loaded), start=1):
            docs.append(
                Document(
                    page_content=chunk.text,
                    metadata={
                        "source_id": chunk.source_id,
                        "source_name_included": False,
                        "page_numbers": ",".join(str(page) for page in chunk.page_numbers),
                        "chunk_index": index,
                        "doc_category": file_path.relative_to(CONFIG.knowledge_base_dir).parts[0]
                        if len(file_path.relative_to(CONFIG.knowledge_base_dir).parts) > 1
                        else file_path.parent.name,
                        "file_type": file_path.suffix.lower().lstrip("."),
                    },
                )
            )
    return docs


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the local general knowledge-base vector store.")
    parser.add_argument("--knowledge-base", default=str(CONFIG.knowledge_base_dir))
    parser.add_argument("--db", default=str(CONFIG.vector_db_dir))
    parser.add_argument("--reset", action="store_true", help="Delete and rebuild the vector database.")
    args = parser.parse_args()

    kb_root = Path(args.knowledge_base).resolve()
    db_dir = Path(args.db).resolve()
    if not kb_root.exists():
        raise FileNotFoundError(f"Knowledge base folder not found: {kb_root}")

    files = iter_knowledge_files(kb_root)
    if args.reset and db_dir.exists():
        shutil.rmtree(db_dir)

    try:
        from langchain_chroma import Chroma
        from langchain_ollama import OllamaEmbeddings
    except ImportError as exc:
        raise RuntimeError("Missing Chroma/Ollama dependencies. Install requirements.txt.") from exc

    docs = build_documents_for_chroma(files)
    embeddings = OllamaEmbeddings(model=CONFIG.embedding_model)
    vector_store = Chroma(
        collection_name=CONFIG.collection_name,
        persist_directory=str(db_dir),
        embedding_function=embeddings,
    )

    ids = []
    for i, doc in enumerate(docs):
        ids.append(f"{doc.metadata.get('source_id')}:{doc.metadata.get('page_numbers')}:{doc.metadata.get('chunk_index')}:{i}")

    batch_size = 32
    for start in range(0, len(docs), batch_size):
        vector_store.add_documents(docs[start : start + batch_size], ids=ids[start : start + batch_size])

    print(f"Ingested {len(docs)} chunks from {len(files)} general knowledge-base files.")
    print("Patient/case folders were blocked from ingestion by design.")


if __name__ == "__main__":
    main()
