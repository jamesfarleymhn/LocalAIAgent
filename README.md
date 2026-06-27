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

## v2.4 progress/status output

Long PDF/OCR/local-model runs now print progress messages to stderr by default. This makes it easier to see whether the app is loading the PDF, OCRing pages, calling Ollama, retrieving RAG evidence, or writing output.

Example:

```powershell
python main.py --case "C:\Users\jf062324\Documents\CDI_Denials\Denial_Letters\Example Humana Denial Letter Coding Barnes.pdf" --question "summarize the denial letter" --mode fast --ocr-mode never --output outputs\summary_test.json
```

You should see messages like:

```text
[14:22:11 +0.0s] [1/7] Starting denial analysis...
[14:22:11 +0.0s] [2/7] Loading and extracting document text...
[14:22:13 +2.4s] Reading PDF page 1/4 with text extraction...
[14:22:17 +6.1s] Sending compact fast summary/extraction prompt to Ollama...
[14:23:02 +51.3s] Ollama fast summary/extraction returned.
[14:23:02 +51.4s] [7/7] Analysis complete.
```

To hide progress messages:

```powershell
python main.py --case "path\to\case.pdf" --question "summarize the denial letter" --quiet
```

Progress messages are written to stderr so JSON output remains clean when printing to stdout.

## v2.5 human-readable case review output

This version adds a top-level `case_review` section and writes a Markdown report by default when `--output` is used.

The goal is to make it easy to verify whether the tool got the important facts correct before looking at the machine JSON.

The report focuses on:

- payer / reviewer
- payee / provider / facility
- patient identifiers found in the submitted document
- claim number and service dates
- denial type, decision, and rationale
- before and after DRG values
- non-DRG coding changes, including diagnosis/procedure code findings
- evidence excerpts and page numbers for manual verification

Example:

```powershell
python main.py --case "C:\Users\jf062324\Documents\CDI_Denials\Denial_Letters\Example Humana Denial Letter Coding Barnes.pdf" --question "summarize the denial letter" --mode fast --output outputs\summary_test.json
```

That writes two files:

```text
outputs\summary_test.json
outputs\summary_test.case_review.md
```

Open the `.case_review.md` first. The JSON is for downstream tools and debugging.

To write only the human-readable report:

```powershell
python main.py --case "file.pdf" --question "summarize the denial letter" --mode fast --output outputs\case_review.md --output-format report
```

To write only JSON:

```powershell
python main.py --case "file.pdf" --question "summarize the denial letter" --mode fast --output outputs\summary_test.json --output-format json
```

PHI note: the submitted case is still read at runtime only. The case review report may contain PHI because it reflects the submitted denial document, so store it only in an approved secure location.

## v2.6 note: safer case-review fact extraction

This version adds stricter validation before values appear in the human-readable `.case_review.md` report. It rejects OCR/table-header text such as `Patient name: Member ID: DOB: Account number:` instead of showing it as an extracted patient or claim value. It also adds a targeted parser for payer review-summary grids when OCR preserves a value row. If a value row cannot be confidently parsed, the report now leaves the fact as `Not found / needs manual review` rather than displaying header-label garbage.

## Version 3.0 concise final case output

This version changes the default JSON output. The file passed to `--output` now contains a concise resolved `final_case` object by default, rather than every raw extraction candidate.

Use this for normal workflow testing:

```powershell
python main.py --case "C:\path\to\denial.pdf" --question "identify the original DRG, updated DRG, coding change, payer, payee, patient, and claim details, and summarize the denial document" --mode fast --ocr-mode always --output outputs\drg_check.final.json
```

It writes:

- `outputs\drg_check.final.json` — concise workflow-friendly JSON
- `outputs\drg_check.final.case_review.md` — human-readable fact review

If you need the internal extraction candidates for troubleshooting, add:

```powershell
--debug-output outputs\drg_check.debug.json
```

Or write the full old-style JSON to `--output` with:

```powershell
--json-detail full
```

The concise `final_case` object resolves multiple page candidates into one best value per fact. It includes:

- `case_summary`
- `parties`
- `patient`
- `claim`
- `denial`
- `coding_change`
- `confidence`

The `coding_change` section now has dedicated fields for original DRG, updated DRG, provider assigned/billed non-DRG codes, payer non-DRG findings, and unsupported procedure/code findings. The model prompt was also changed so it should not attempt to fill patient or claim fields on every page.

## v3.1 Vision fact-check mode for scanned PDFs

Scanned denial letters often contain DRG tables that are easy for a human to read but difficult for plain OCR. The `vision-fact-check` mode renders PDF pages as images and sends those images to a local Ollama vision-capable model. This lets the model read the table layout directly instead of relying only on damaged OCR text.

Install/pull a local vision model in Ollama. Example:

```powershell
ollama pull qwen2.5vl:7b
```

Then run the scanned PDF through the vision path:

```powershell
python main.py --case "C:\Users\jf062324\Documents\CDI_Denials\Denial_Letters\Example Humana Denial Letter Coding Barnes.pdf" --question "identify the original DRG, updated DRG, coding change, payer, payee, patient, and claim details" --mode vision-fact-check --vision-model qwen2.5vl:7b --output outputs\vision_drg.final.json
```

If you know the DRG table is on a specific page, target only that page to make it faster:

```powershell
python main.py --case "C:\Users\jf062324\Documents\CDI_Denials\Denial_Letters\Example Humana Denial Letter Coding Barnes.pdf" --question "identify the DRG change" --mode vision-fact-check --vision-model qwen2.5vl:7b --vision-pages 3 --output outputs\vision_page3.final.json
```

You can pass page ranges:

```powershell
--vision-pages 1,3-5
```

Useful tuning options:

```powershell
--vision-zoom 2.0        # default; increase to 2.5 if small table text is missed
--vision-max-pages 12    # default number of pages when --vision-pages is not set
--ollama-timeout 900     # increase if the local vision model is slow
```

Privacy behavior: submitted PDFs are rendered at runtime and sent only to the configured local Ollama endpoint. They are not ingested into Chroma. The generated `.final.json` and `.case_review.md` may contain PHI, so save them only in approved locations.

---

## v4.0 scanned-document intelligence mode

This version adds a new default-style extraction path for scanned denial PDFs:

```powershell
python main.py --case "C:\path\to\denial.pdf" --question "identify the original DRG, updated DRG, coding change, payer, payee, patient, and claim details" --mode scanned-extract --output outputs\case.final.json
```

`scanned-extract` is designed for scanned PDFs where the text layer is missing or unreliable. It does **not** send every full page to a vision model and it does **not** call Ollama for every page. Instead it:

1. Renders each PDF page once.
2. Runs OCR with bounding boxes/layout information once.
3. Reconstructs page lines in reading order.
4. Automatically looks for important denial sections such as:
   - `Review Findings Summary`
   - `DRG Table`
   - `The original codes billed were`
   - `The new coding assignment is`
   - `ICD-10-PCS code ... is not supported`
   - `overpayment` / `overpaid`
5. Resolves one concise final case object.
6. Writes a readable `.case_review.md` next to the JSON.

This mode is the recommended starting point for scanned Humana-style denial letters because it avoids the long per-page local vision model calls.

### Recommended command

```powershell
python main.py --case "C:\Users\jf062324\Documents\CDI_Denials\Denial_Letters\Example Humana Denial Letter Coding Barnes.pdf" --question "identify the original DRG, updated DRG, coding change, payer, payee, patient, and claim details, and summarize the denial document" --mode scanned-extract --output outputs\drg_scanned.final.json
```

Open this first:

```text
outputs\drg_scanned.final.case_review.md
```

### Useful options

```powershell
--scanned-zoom 2.5       # better OCR, slower
--scanned-max-pages 8    # limit pages during testing
--include-page-text      # include OCR text in debug/full JSON only
--debug-output outputs\debug.json
```

### Why this mode exists

Full-page local vision models can be extremely slow on scanned 8-page denial PDFs. `scanned-extract` is the pragmatic middle ground: it uses OCR/layout to automatically locate important facts and tables, then produces the concise resolved JSON without requiring the user to manually specify page numbers.
