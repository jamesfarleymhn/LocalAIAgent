# PyMuPDF4LLM PDF Extraction Test

This standalone test parses a denial PDF with PyMuPDF4LLM and runs a focused DRG/coding/claim fact extractor.

## Install

```powershell
cd pymupdf4llm_solution
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

## Run without Ollama

```powershell
python run.py --pdf "C:\path\to\denial.pdf" --output-base outputs\pymupdf4llm_test
```

Open first:

```text
outputs\pymupdf4llm_test.case_review.md
```

Other outputs:

```text
outputs\pymupdf4llm_test.markdown.md
outputs\pymupdf4llm_test.pages.txt
outputs\pymupdf4llm_test.facts.json
```

## Run with Ollama review

```powershell
python run.py --pdf "C:\path\to\denial.pdf" --output-base outputs\pymupdf4llm_test --ollama --model llama3.1:latest
```

This creates:

```text
outputs\pymupdf4llm_test.ollama_review.md
```

## What to check

Compare whether this parser captures:

- before/original DRG
- after/updated/recommended DRG
- ICD-10-CM or ICD-10-PCS coding changes
- payer/reviewer
- patient/account/claim/service-date details
- evidence excerpts and page numbers
