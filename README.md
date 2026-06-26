# Local Denial Document RAG Overhaul

This version separates three things that should not be mixed:

1. **Submitted case files** are read at runtime only.
2. **Reusable knowledge-base files** are ingested separately for general policy, coding, CDI, sanitized appeal examples, sanitized case studies, and appeal support.
3. **The final result** is a JSON object that can be passed to later steps.

The Python scripts do not contain patient examples, real identifiers, or embedded PHI. Runtime JSON can contain PHI because it is extracted from the submitted local document; control where you save that output.

## Setup

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
ollama pull llama3.1:latest
ollama pull bge-m3
```

## Analyze a submitted denial document

Submitted patient/case files are analyzed at runtime. They are **not** ingested into Chroma.

```powershell
python main.py --case "path\to\local\case.pdf" --question "What is being denied and what should we look for?" --output outputs\result.json
```

By default, extracted raw page text is **not** included in the JSON. The document is still analyzed end-to-end by chunking every page and merging the results.

To include extracted page text in the JSON:

```powershell
python main.py --case "path\to\local\case.pdf" --question "Summarize this." --include-page-text --output outputs\result.json
```

To run without Ollama for a dependency check or regex-only extraction:

```powershell
python main.py --case "path\to\local\case.pdf" --no-llm
```

## Build the reusable knowledge base

The Chroma DB should contain reusable support material, not raw patient files. This project allows appeal letters and case studies **only as sanitized/de-identified reusable knowledge**.

Recommended folders:

```text
knowledge_base/
  payer_policies/
  coding_guidelines/
  clinical_guidelines/
  cdi_guides/
  appeal_templates/
  appeal_letters_deidentified/
  case_studies_deidentified/
```

Do **not** use these for reusable ingestion:

```text
knowledge_base/submitted_cases/
knowledge_base/patient_files/
knowledge_base/uploaded_cases/
knowledge_base/user_input/
```

Run ingest:

```powershell
python ingest.py --knowledge-base knowledge_base --reset
```

What ingest does:

- Skips live/submitted patient-case folders instead of crashing.
- Sanitizes text before embedding it into Chroma by default.
- Allows appeal letters and case studies after sanitization.
- Writes `chroma_denials_db\ingest_manifest.json` showing loaded files, skipped files, redaction counts, and chunks skipped because PHI-like indicators remained.

If you are ingesting only already-public/reference material and want to bypass sanitizer:

```powershell
python ingest.py --knowledge-base knowledge_base --reset --no-sanitize
```

Use that only for verified non-PHI material.

## Ask for appeal arguments or a draft starter

```powershell
python main.py --case "path\to\local\case.pdf" --question "What are the strongest appeal arguments and draft a starter appeal letter?" --use-kb --output outputs\appeal_support.json
```

The answer uses:

- submitted case facts from the runtime case document only;
- sanitized/de-identified Chroma evidence for reusable appeal themes, argument patterns, policy support, and template language.

## What changed from the old design

- No fixed `case_context[:12000]` or `normalized_text[:12000]` truncation.
- Every readable page is chunked and analyzed.
- OCR is page-level fallback instead of all-or-nothing PDF fallback.
- LangChain/Chroma imports are lazy, so case-only runs do not crash because of optional RAG dependencies.
- Submitted case files are not ingested into the reusable vector database.
- Output is a structured JSON object with core facts, all extracted fields, evidence snippets, warnings, and an optional answer.
- Chroma ingestion now supports sanitized/de-identified appeal examples and case studies for appeal-letter drafting support.

## v2.1 fix: prompt rendering and model-first extraction

This version fixes the crash below:

```text
KeyError: '\n  "plain_english_summary"'
```

Cause: Python `str.format()` was being used on prompt templates that contained literal JSON examples. The JSON braces were interpreted as replacement fields. Prompts are now rendered with `prompting.render_prompt()`, which only replaces exact placeholders like `{extraction_json}` and leaves JSON examples alone.

This version also moves the submitted-case workflow closer to the intended design:

1. The local model examines each document chunk first and extracts structured JSON.
2. Regex extraction runs afterward as fallback/validation support, not as the main understanding layer.
3. Model-extracted fields are checked against the same source chunk, and each field includes validation metadata.
4. OCR/Torch `pin_memory` CPU-only warnings are suppressed in `document_loader.py` before EasyOCR/Torch is loaded.

The source code still contains no embedded PHI examples. Runtime outputs may contain PHI because they are extracted from the local submitted case file, so control where JSON outputs are saved.

## v2.2 fix: Ollama timeout handling

This version fixes runs failing with a raw Python traceback like:

```text
TimeoutError: timed out
```

The local Ollama client now defaults to a 600-second timeout per model call instead of 180 seconds, catches timeout errors, and falls back to deterministic extraction/partial answers instead of crashing the whole run.

You can override the timeout from the command line:

```powershell
python main.py --case "path\to\case.pdf" --question "summarize the denial letter" --ollama-timeout 900
```

Or by environment variable:

```powershell
$env:OLLAMA_TIMEOUT_SECONDS="900"
python main.py --case "path\to\case.pdf" --question "summarize the denial letter"
```

For a quick extraction-only check without calling Ollama:

```powershell
python main.py --case "path\to\case.pdf" --question "summarize the denial letter" --no-llm
```

## v2.3 speed fix: automatic fast/full/appeal modes

This version fixes the root cause of slow simple summaries. The app no longer runs the expensive full model-first chunk workflow for every question.

### Modes

```text
auto   = default. Chooses fast/full/appeal based on the question.
fast   = fastest path for summaries/basic extraction. Regex scans all pages, then one compact local-model call summarizes selected pages.
full   = deeper document Q&A. Runs model extraction over every chunk.
appeal = full extraction plus sanitized Chroma knowledge-base support for appeal arguments/drafts.
```

For your summary command, `auto` will choose `fast`:

```powershell
python main.py --case "C:\Users\jf062324\Documents\CDI_Denials\Denial_Letters\Example Humana Denial Letter Coding Barnes.pdf" --question "summarize the denial letter"
```

Equivalent explicit fast command:

```powershell
python main.py --case "C:\Users\jf062324\Documents\CDI_Denials\Denial_Letters\Example Humana Denial Letter Coding Barnes.pdf" --question "summarize the denial letter" --mode fast
```

Fastest test for text-layer PDFs, skipping OCR entirely:

```powershell
python main.py --case "C:\Users\jf062324\Documents\CDI_Denials\Denial_Letters\Example Humana Denial Letter Coding Barnes.pdf" --question "summarize the denial letter" --mode fast --ocr-mode never
```

Use OCR only when the PDF is scanned/image-based:

```powershell
python main.py --case "path\to\scanned.pdf" --question "summarize the denial letter" --mode fast --ocr-mode always
```

Deep full-document analysis:

```powershell
python main.py --case "path\to\case.pdf" --question "answer this specific question about the document" --mode full
```

Appeal arguments and starter draft using sanitized/de-identified Chroma examples:

```powershell
python main.py --case "path\to\case.pdf" --question "What are the strongest appeal arguments and draft a starter appeal letter?" --mode appeal --use-kb --output outputs\appeal_support.json
```

### Optional speed settings

Reduce how much text is sent to the model in fast mode:

```powershell
python main.py --case "path\to\case.pdf" --question "summarize the denial letter" --mode fast --fast-max-pages 5 --fast-max-chars 16000
```

Try a smaller/faster local Ollama model without editing code:

```powershell
python main.py --case "path\to\case.pdf" --question "summarize the denial letter" --mode fast --model qwen2.5:7b
```

You must pull the model first:

```powershell
ollama pull qwen2.5:7b
```
