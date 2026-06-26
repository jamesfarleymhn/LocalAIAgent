from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

from chunking import chunk_loaded_case
from config import CONFIG
from document_loader import SUPPORTED_EXTENSIONS, load_case_files
from privacy import classify_knowledge_path, sanitize_text, scan_phi_indicators, stable_file_id


def iter_knowledge_files(root: Path) -> tuple[list[tuple[Path, str, str]], list[dict[str, str]]]:
    """Return ingestible files and skipped files.

    This intentionally skips risky paths instead of crashing the whole ingest job.
    """
    files: list[tuple[Path, str, str]] = []
    skipped: list[dict[str, str]] = []

    for path in sorted(root.rglob("*")):
        if path.is_dir():
            continue
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        decision, reason = classify_knowledge_path(path, root)
        if decision == "block":
            skipped.append({"source_id": stable_file_id(path), "file_type": path.suffix.lower().lstrip("."), "reason": reason})
            continue

        files.append((path, decision, reason))

    return files, skipped


def get_doc_category(file_path: Path, kb_root: Path) -> str:
    """Return a non-identifying broad category for metadata.

    Do not store folder names directly in Chroma metadata because folder/file names can contain PHI.
    """
    try:
        parts = [part.lower() for part in file_path.relative_to(kb_root).parts[:-1]]
    except ValueError:
        parts = [part.lower() for part in file_path.parts[:-1]]
    joined = " ".join(parts)
    if "appeal" in joined:
        return "appeal_support"
    if "case" in joined or "stud" in joined:
        return "case_study_support"
    if "payer" in joined or "policy" in joined:
        return "payer_policy"
    if "coding" in joined or "icd" in joined or "drg" in joined or "cpt" in joined:
        return "coding_guideline"
    if "clinical" in joined or "criteria" in joined:
        return "clinical_guideline"
    if "cdi" in joined:
        return "cdi_guide"
    if "template" in joined or "sample" in joined or "example" in joined:
        return "template_or_example"
    return "reference_material"


def build_documents_for_chroma(
    files: list[tuple[Path, str, str]],
    *,
    kb_root: Path,
    sanitize: bool,
    skip_if_phi_remains: bool,
) -> tuple[list[Any], dict[str, Any]]:
    try:
        from langchain_core.documents import Document
    except ImportError as exc:
        raise RuntimeError("Missing LangChain core dependency. Install requirements.txt.") from exc

    docs: list[Any] = []
    manifest: dict[str, Any] = {
        "files_seen": len(files),
        "files_loaded": 0,
        "chunks_ingested": 0,
        "chunks_skipped_after_scan": 0,
        "raw_phi_indicator_counts": Counter(),
        "redaction_counts": Counter(),
        "remaining_phi_indicator_counts": Counter(),
        "skipped_chunks": [],
        "loaded_sources": [],
    }

    for file_path, path_decision, path_reason in files:
        try:
            loaded = load_case_files([file_path], include_source_names=False)
        except Exception as exc:
            manifest.setdefault("file_errors", []).append(
                {"source_id": stable_file_id(file_path), "file_type": file_path.suffix.lower().lstrip("."), "error": f"{type(exc).__name__}: {exc}"}
            )
            continue

        source_chunks = 0
        for index, chunk in enumerate(chunk_loaded_case(loaded), start=1):
            raw_scan = scan_phi_indicators(chunk.text)
            manifest["raw_phi_indicator_counts"].update(raw_scan.counts)

            page_text = chunk.text
            redaction_counts: dict[str, int] = {}
            if sanitize or path_decision == "sanitize":
                page_text, redaction_counts = sanitize_text(page_text)
                manifest["redaction_counts"].update(redaction_counts)

            remaining_scan = scan_phi_indicators(page_text)
            manifest["remaining_phi_indicator_counts"].update(remaining_scan.counts)

            if skip_if_phi_remains and remaining_scan.has_risk:
                manifest["chunks_skipped_after_scan"] += 1
                manifest["skipped_chunks"].append(
                    {
                        "source_id": chunk.source_id,
                        "chunk_id": chunk.chunk_id,
                        "page_numbers": chunk.page_numbers,
                        "reason": "PHI-like indicators remained after sanitization; chunk not embedded.",
                        "remaining_indicators": remaining_scan.counts,
                    }
                )
                continue

            metadata = {
                "source_id": chunk.source_id,
                "page_numbers": ",".join(str(page) for page in chunk.page_numbers),
                "chunk_index": index,
                "doc_category": get_doc_category(file_path, kb_root),
                "file_type": file_path.suffix.lower().lstrip("."),
                "source_classification": path_decision,
                "sanitized_before_embedding": bool(sanitize or path_decision == "sanitize"),
                "path_policy_reason": path_reason,
                "use_limit": "Reusable general appeal/policy/case-study support only; do not treat as patient-specific facts.",
            }
            docs.append(Document(page_content=page_text, metadata=metadata))
            source_chunks += 1

        if source_chunks:
            manifest["files_loaded"] += 1
            manifest["loaded_sources"].append(
                {
                    "source_id": loaded.document_id,
                    "classification": path_decision,
                    "doc_category": get_doc_category(file_path, kb_root),
                    "chunks_loaded": source_chunks,
                    "sanitized_before_embedding": bool(sanitize or path_decision == "sanitize"),
                }
            )

    manifest["chunks_ingested"] = len(docs)
    # Convert Counters to plain dicts for JSON output.
    for key in ["raw_phi_indicator_counts", "redaction_counts", "remaining_phi_indicator_counts"]:
        manifest[key] = dict(manifest[key])
    return docs, manifest


def write_manifest(db_dir: Path, manifest: dict[str, Any], skipped_files: list[dict[str, str]]) -> None:
    db_dir.mkdir(parents=True, exist_ok=True)
    manifest = dict(manifest)
    manifest["skipped_files"] = skipped_files
    (db_dir / "ingest_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the local reusable knowledge-base vector store.")
    parser.add_argument("--knowledge-base", default=str(CONFIG.knowledge_base_dir))
    parser.add_argument("--db", default=str(CONFIG.vector_db_dir))
    parser.add_argument("--reset", action="store_true", help="Delete and rebuild the vector database.")
    parser.add_argument(
        "--no-sanitize",
        action="store_true",
        help="Do not sanitize text before embedding. Use only for already verified non-PHI public/reference material.",
    )
    parser.add_argument(
        "--allow-remaining-phi-indicators",
        action="store_true",
        help="Embed chunks even if the scanner still sees PHI-like indicators after sanitization. Not recommended.",
    )
    args = parser.parse_args()

    kb_root = Path(args.knowledge_base).expanduser().resolve()
    db_dir = Path(args.db).expanduser().resolve()
    if not kb_root.exists():
        raise FileNotFoundError(f"Knowledge base folder not found: {kb_root}")

    files, skipped_files = iter_knowledge_files(kb_root)
    if args.reset and db_dir.exists():
        shutil.rmtree(db_dir)

    try:
        from langchain_chroma import Chroma
        from langchain_ollama import OllamaEmbeddings
    except ImportError as exc:
        raise RuntimeError("Missing Chroma/Ollama dependencies. Install requirements.txt.") from exc

    docs, manifest = build_documents_for_chroma(
        files,
        kb_root=kb_root,
        sanitize=not args.no_sanitize,
        skip_if_phi_remains=not args.allow_remaining_phi_indicators,
    )

    if not docs:
        write_manifest(db_dir, manifest, skipped_files)
        print("No chunks were ingested.")
        print(f"Skipped files: {len(skipped_files)}")
        print(f"Manifest written to: {db_dir / 'ingest_manifest.json'}")
        return

    embeddings = OllamaEmbeddings(model=CONFIG.embedding_model)
    vector_store = Chroma(
        collection_name=CONFIG.collection_name,
        persist_directory=str(db_dir),
        embedding_function=embeddings,
    )

    ids = [
        f"{doc.metadata.get('source_id')}:{doc.metadata.get('page_numbers')}:{doc.metadata.get('chunk_index')}:{i}"
        for i, doc in enumerate(docs)
    ]

    batch_size = 32
    for start in range(0, len(docs), batch_size):
        vector_store.add_documents(docs[start : start + batch_size], ids=ids[start : start + batch_size])

    write_manifest(db_dir, manifest, skipped_files)
    print(f"Ingested {len(docs)} sanitized/reusable chunks from {manifest['files_loaded']} file(s).")
    print(f"Skipped blocked files: {len(skipped_files)}")
    print(f"Chunks skipped because PHI-like indicators remained: {manifest['chunks_skipped_after_scan']}")
    print(f"Manifest written to: {db_dir / 'ingest_manifest.json'}")


if __name__ == "__main__":
    main()
