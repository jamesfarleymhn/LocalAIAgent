from __future__ import annotations

import argparse
from pathlib import Path

from denial_extractor import answer_question_from_case, extract_case_to_json
from document_loader import load_case_files
from json_utils import json_dumps


def analyze_once(args: argparse.Namespace) -> dict:
    loaded = load_case_files(args.case, include_source_names=args.include_source_names)
    result = extract_case_to_json(
        loaded,
        use_llm=not args.no_llm,
        include_page_text=args.include_page_text,
        include_source_names=args.include_source_names,
    )

    if args.question:
        result["answer"] = answer_question_from_case(
            result,
            loaded,
            args.question,
            use_llm=not args.no_llm,
            use_kb=args.use_kb,
        )

    return result


def write_output(result: dict, output_path: str | None) -> None:
    rendered = json_dumps(result, indent=2)
    if output_path:
        path = Path(output_path).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered, encoding="utf-8")
        print(f"Wrote JSON output to: {path}")
    else:
        print(rendered)


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
        print(json_dumps(result, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Local denial document analyzer. Reads submitted cases at runtime and returns structured JSON."
    )
    parser.add_argument("--case", nargs="+", help="Path(s) to submitted local case document(s).")
    parser.add_argument("--question", help="Question to answer about the submitted document.")
    parser.add_argument("--output", help="Optional path to write JSON output.")
    parser.add_argument("--use-kb", action="store_true", help="Retrieve general policy/guideline support from local RAG DB.")
    parser.add_argument("--no-llm", action="store_true", help="Run regex-only extraction without local Ollama.")
    parser.add_argument("--include-page-text", action="store_true", help="Include extracted page text in JSON output.")
    parser.add_argument("--include-source-names", action="store_true", help="Include original file names in JSON output.")
    parser.add_argument("--interactive", action="store_true", help="Run an interactive prompt loop.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.interactive:
        interactive(args)
        return

    if not args.case:
        parser.error("--case is required unless --interactive is used.")

    result = analyze_once(args)
    write_output(result, args.output)


if __name__ == "__main__":
    main()
