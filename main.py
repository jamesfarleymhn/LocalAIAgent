from __future__ import annotations

import argparse
import os
from pathlib import Path

from denial_extractor import (
    answer_question_fast,
    answer_question_from_case,
    choose_analysis_mode,
    extract_case_to_json,
    extract_case_to_json_fast,
)
from document_loader import load_case_files
from json_utils import json_dumps
from progress import Progress
from case_review import render_case_review_markdown


def make_llm_kwargs(args: argparse.Namespace) -> dict:
    return {"llm_timeout_seconds": args.ollama_timeout}


def analyze_once(args: argparse.Namespace) -> dict:
    progress = getattr(args, "progress", None) or Progress(enabled=not getattr(args, "quiet", False))
    selected_mode = choose_analysis_mode(args.question, args.mode, use_kb=args.use_kb)
    progress.step(1, 7, "Starting denial analysis...")
    progress.log(f"Mode selected: {selected_mode}")
    progress.log(f"OCR mode: {args.ocr_mode}")
    progress.log(f"Ollama model: {os.getenv('OLLAMA_MODEL', args.model or 'default from config')}")
    progress.log(f"Ollama timeout: {args.ollama_timeout or 'default from config/env'} seconds")

    progress.step(2, 7, "Loading and extracting document text...")
    loaded = load_case_files(
        args.case,
        include_source_names=args.include_source_names,
        ocr_mode=args.ocr_mode,
        progress=progress,
    )

    use_llm = not args.no_llm

    if selected_mode == "fast":
        progress.step(3, 7, "Running fast extraction path...")
        result = extract_case_to_json_fast(
            loaded,
            question=args.question,
            use_llm=use_llm,
            include_page_text=args.include_page_text,
            include_source_names=args.include_source_names,
            llm_timeout_seconds=args.ollama_timeout,
            max_fast_pages=args.fast_max_pages,
            max_fast_chars=args.fast_max_chars,
            progress=progress,
        )
        if args.question:
            progress.step(5, 7, "Answering question in fast mode...")
            result["answer"] = answer_question_fast(
                result,
                loaded,
                args.question,
                use_llm=use_llm,
                llm_timeout_seconds=args.ollama_timeout,
                progress=progress,
            )
        progress.step(6, 7, "Preparing output JSON...")
        progress.step(7, 7, "Analysis complete.")
        return result

    # Full and appeal modes still run the full model-first extraction workflow.
    progress.step(3, 7, "Running full model-first extraction path...")
    result = extract_case_to_json(
        loaded,
        use_llm=use_llm,
        include_page_text=args.include_page_text,
        include_source_names=args.include_source_names,
        llm_timeout_seconds=args.ollama_timeout,
        progress=progress,
    )
    result["analysis_mode"] = selected_mode

    if args.question:
        progress.step(5, 7, "Answering user question...")
        result["answer"] = answer_question_from_case(
            result,
            loaded,
            args.question,
            use_llm=use_llm,
            use_kb=(args.use_kb or selected_mode == "appeal"),
            llm_timeout_seconds=args.ollama_timeout,
            progress=progress,
        )

    progress.step(6, 7, "Preparing output JSON...")
    progress.step(7, 7, "Analysis complete.")
    return result


def _default_report_path(json_path: Path) -> Path:
    return json_path.with_name(f"{json_path.stem}.case_review.md")


def write_output(
    result: dict,
    output_path: str | None,
    *,
    output_format: str = "both",
    report_output_path: str | None = None,
) -> None:
    """Write machine JSON plus a human-readable case review.

    The JSON is still available for downstream tools, but the markdown report is
    meant for humans who want to verify whether the extraction got the facts right.
    """
    review = result.get("case_review") or {}
    report = render_case_review_markdown(review) if review else "No case_review section was generated."
    rendered_json = json_dumps(result, indent=2)

    if output_path:
        path = Path(output_path).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)

        if output_format in {"json", "both"}:
            path.write_text(rendered_json, encoding="utf-8")
            print(f"Wrote JSON output to: {path}")

        if output_format in {"report", "both"}:
            report_path = Path(report_output_path).expanduser().resolve() if report_output_path else _default_report_path(path)
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(report, encoding="utf-8")
            print(f"Wrote human-readable case review to: {report_path}")
        return

    if output_format == "json":
        print(rendered_json)
    else:
        print(report)


def interactive(args: argparse.Namespace) -> None:
    while True:
        raw_case = input("Enter case file path(s), comma-separated, or q to quit: ").strip()
        if raw_case.lower() == "q":
            return
        question = input("Question about the submitted document: ").strip()
        case_paths = [item.strip().strip('"') for item in raw_case.split(",") if item.strip()]
        loop_args = argparse.Namespace(**vars(args))
        loop_args.case = case_paths
        loop_args.question = question
        loop_args.output = None
        result = analyze_once(loop_args)
        print(render_case_review_markdown(result.get("case_review", {})))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Local denial document analyzer. Reads submitted cases at runtime and returns structured JSON."
    )
    parser.add_argument("--case", nargs="+", help="Path(s) to submitted local case document(s).")
    parser.add_argument("--question", help="Question to answer about the submitted document.")
    parser.add_argument("--output", help="Optional path to write JSON output. In default both mode, a .case_review.md report is also written next to it.")
    parser.add_argument("--report-output", help="Optional path for the human-readable Markdown case review report.")
    parser.add_argument("--output-format", choices=["json", "report", "both"], default="both", help="json writes machine output, report writes a human-readable case review, both writes both. Default: both.")
    parser.add_argument("--use-kb", action="store_true", help="Retrieve general policy/guideline support from local RAG DB.")
    parser.add_argument("--no-llm", action="store_true", help="Run deterministic extraction without local Ollama.")
    parser.add_argument("--include-page-text", action="store_true", help="Include extracted page text in JSON output.")
    parser.add_argument("--include-source-names", action="store_true", help="Include original file names in JSON output.")
    parser.add_argument("--interactive", action="store_true", help="Run an interactive prompt loop.")
    parser.add_argument("--ollama-timeout", type=int, default=None, help="Seconds to wait for each local Ollama response. Default comes from OLLAMA_TIMEOUT_SECONDS or 600.")
    parser.add_argument(
        "--mode",
        choices=["auto", "fast", "full", "appeal"],
        default="auto",
        help="auto chooses fast for simple summaries, full for deeper document Q&A, and appeal for appeal/policy questions.",
    )
    parser.add_argument(
        "--ocr-mode",
        choices=["auto", "never", "always"],
        default="auto",
        help="auto OCRs only low-text PDF pages; never is fastest for text-layer PDFs; always is for scanned PDFs.",
    )
    parser.add_argument("--fast-max-pages", type=int, default=8, help="Maximum pages sent to the local model in fast mode.")
    parser.add_argument("--fast-max-chars", type=int, default=24000, help="Maximum characters sent to the local model in fast mode.")
    parser.add_argument("--model", help="Override the Ollama generation model for this run, e.g. qwen2.5:7b or llama3.1:8b.")
    parser.add_argument("--quiet", action="store_true", help="Hide progress/status messages.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.model:
        os.environ["OLLAMA_MODEL"] = args.model

    if args.interactive:
        interactive(args)
        return

    if not args.case:
        parser.error("--case is required unless --interactive is used.")

    result = analyze_once(args)
    write_output(result, args.output, output_format=args.output_format, report_output_path=args.report_output)


if __name__ == "__main__":
    main()
