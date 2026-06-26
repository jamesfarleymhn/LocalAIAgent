# Local Denial Document RAG Overhaul

This version separates three things that should not be mixed:

1. **Submitted case files** are read at runtime only.
2. **Reusable knowledge-base files** are ingested separately for general policy, coding, CDI, and appeal support.
3. **The final result** is a JSON object that can be passed to later steps.

The Python scripts do not contain patient examples, real identifiers, or embedded PHI. Runtime JSON can contain PHI because it is extracted from the submitted local document; control where you save that output.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
ollama pull llama3.1:latest
ollama pull bge-m3
```

## Analyze a submitted denial document

```bash
python main.py --case "path/to/local/case.pdf" --question "What is being denied and what should we look for?" --output outputs/result.json
```

By default, extracted raw page text is **not** included in the JSON. The document is still analyzed end-to-end by chunking every page and merging the results.

To include extracted page text in the JSON:

```bash
python main.py --case "path/to/local/case.pdf" --question "Summarize this." --include-page-text --output outputs/result.json
```

To run without Ollama for a dependency check or regex-only extraction:

```bash
python main.py --case "path/to/local/case.pdf" --no-llm
```

## Build the general knowledge base

Only put non-PHI policy/guideline/reference documents under `knowledge_base/`. The ingest script blocks folders with names like `submitted_cases`, `denial_letters`, `patient_files`, and similar.

```bash
python ingest.py --knowledge-base knowledge_base --reset
```

Then ask a question with general RAG support:

```bash
python main.py --case "path/to/local/case.pdf" --question "What appeal arguments are supported?" --use-kb
```

## What changed from the old design

- No fixed `case_context[:12000]` or `normalized_text[:12000]` truncation.
- Every readable page is chunked and analyzed.
- OCR is page-level fallback instead of all-or-nothing PDF fallback.
- LangChain/Chroma imports are lazy, so case-only runs do not crash because of optional RAG dependencies.
- Submitted case files are not ingested into the reusable vector database.
- Output is a structured JSON object with core facts, all extracted fields, evidence snippets, warnings, and an optional answer.
