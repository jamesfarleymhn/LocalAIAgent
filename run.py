from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def log(msg: str) -> None:
    print(msg, flush=True)


def extract_with_pymupdf4llm(pdf_path: Path) -> str:
    import pymupdf4llm
    return pymupdf4llm.to_markdown(str(pdf_path))


def extract_page_texts(pdf_path: Path) -> list[dict]:
    # Use PyMuPDF page text also, so the fact extractor has page numbers even if Markdown has no page markers.
    import fitz
    doc = fitz.open(str(pdf_path))
    pages = []
    for idx, page in enumerate(doc, start=1):
        text = page.get_text("text") or ""
        pages.append({"page": idx, "text": text})
    return pages


def main() -> int:
    parser = argparse.ArgumentParser(description="Test PDF extraction with PyMuPDF4LLM.")
    parser.add_argument("--pdf", required=True, help="Path to the denial PDF.")
    parser.add_argument("--output-base", default="outputs/pymupdf4llm_result", help="Base output path without extension.")
    parser.add_argument("--question", default="Identify payer, patient, claim, before/after DRG, and coding change.")
    parser.add_argument("--ollama", action="store_true", help="Ask local Ollama to produce a readable review from extracted evidence.")
    parser.add_argument("--model", default="llama3.1:latest")
    parser.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args()

    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        print(f"ERROR: PDF not found: {pdf_path}", file=sys.stderr)
        return 2

    from fact_extractor import build_case_facts, write_outputs

    log("[1/5] Extracting Markdown with PyMuPDF4LLM...")
    markdown = extract_with_pymupdf4llm(pdf_path)

    log("[2/5] Extracting page-level text with PyMuPDF for page references...")
    page_texts = extract_page_texts(pdf_path)

    log("[3/5] Running focused DRG/coding/claim fact extractor...")
    facts = build_case_facts(page_texts)
    facts["parser"] = "pymupdf4llm + pymupdf page text"
    facts["source_pdf"] = str(pdf_path)

    out_base = Path(args.output_base)
    log("[4/5] Writing Markdown, page text, JSON facts, and case review...")
    write_outputs(out_base, markdown, page_texts, facts)

    if args.ollama:
        log("[5/5] Asking Ollama for readable review from extracted evidence...")
        from ollama_review import build_review_prompt, ollama_generate
        prompt = build_review_prompt(facts, args.question)
        try:
            review = ollama_generate(prompt, model=args.model, timeout=args.timeout)
        except Exception as exc:
            review = f"Ollama review failed: {type(exc).__name__}: {exc}"
        out_base.with_suffix(".ollama_review.md").write_text(review, encoding="utf-8")
    else:
        log("[5/5] Skipping Ollama review. Use --ollama to enable it.")

    print("\nDone. Open this first:")
    print(out_base.with_suffix(".case_review.md"))
    print("\nAlso compare:")
    print(out_base.with_suffix(".markdown.md"))
    print(out_base.with_suffix(".facts.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
