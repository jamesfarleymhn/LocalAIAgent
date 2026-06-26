from __future__ import annotations

import json
import logging
import re
import warnings

from case_loader import load_case_files
from denial_extractor import extract_denial_info
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from vector import retrieve_supporting_knowledge


# Keep terminal output focused on user-facing results.
warnings.filterwarnings("ignore")
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)

SHOW_DEBUG = False

model = OllamaLLM(
    model="llama3.1:latest",
    temperature=0.1,
    verbose=False,
)


# Questions like these should be answered from the submitted case only.
# They should not retrieve example appeal letters, denial letters, or policies.
CASE_ONLY_TERMS = [
    "summarize",
    "summary",
    "what is the denial",
    "what is this denial",
    "denial type",
    "patient",
    "account",
    "claim",
    "dos",
    "date of service",
    "service date",
    "extract",
    "structured json",
    "what does the letter say",
    "what is the letter about",
]

# Questions like these may benefit from general supporting guidance.
# Even then, patient-specific facts must still come only from the submitted case.
KNOWLEDGE_SUPPORT_TERMS = [
    "appeal",
    "argument",
    "arguments",
    "guideline",
    "policy",
    "criteria",
    "coding guideline",
    "medical necessity criteria",
    "clinical criteria",
    "support",
    "supporting documentation",
    "recommend",
    "draft",
    "write",
    "respond",
    "strategy",
]


CASE_ANALYSIS_TEMPLATE = """
You are analyzing one healthcare denial letter submitted by the user.

SOURCE RULE:
Use ONLY the submitted case document text and the structured denial JSON.
Do NOT use prior examples, knowledge-base files, outside facts, or memory.

Return ONLY valid JSON. Do not include markdown.

Important rules:
- Do not discuss whether patient name, account number, service dates, claim number, payer, before_value, after_value, drg_before_value, or drg_after_value are missing. Python renders those fields exactly from the structured extraction.
- Do not contradict the verified extracted case facts.
- Use the document text to summarize the denial rationale.
- Keep wording concise and factual.

Return this JSON shape:
{{
  "about": "plain English summary of what the submitted letter is about",
  "key_rationale": "main payer/reviewer rationale from the submitted letter",
  "missing_uncertainty": "only non-demographic missing information or uncertainty, or 'No additional uncertainty identified from the submitted file.'"
}}

Structured denial JSON:
{denial_json}

Verified extracted case facts:
{extracted_case_facts}

Submitted case document text:
{case_context}
"""

KB_CONSIDERATIONS_TEMPLATE = """
You are helping with general appeal/coding/guideline support for one submitted healthcare denial letter.

SOURCE PRIORITY:
1. Patient-specific facts come ONLY from the structured denial JSON and submitted case document text.
2. Knowledge-base evidence is for general policy, coding, CDI, or appeal-support context only.
3. Never use knowledge-base evidence to fill patient name, account number, claim number, service dates, payer, before_value, after_value, drg_before_value, drg_after_value, or any case-specific denial facts.

Return ONLY valid JSON. Do not include markdown.

Important rules:
- Do not contradict the verified extracted case facts.
- Do not discuss whether patient/account/claim/DRG fields are missing; Python renders those fields exactly.
- If using evidence, identify the file/page/sheet/row from the evidence block.
- Keep this section focused on the user's question.

Return this JSON shape:
{{
  "policy_guideline_appeal_considerations": "general support considerations relevant to the user's question, with evidence citations when available",
  "missing_uncertainty": "only non-demographic missing information or uncertainty, or 'No additional uncertainty identified from the submitted file.'"
}}

Structured denial JSON:
{denial_json}

Verified extracted case facts:
{extracted_case_facts}

Submitted case document text:
{case_context}

General knowledge-base evidence:
{knowledge_context}

User question:
{question}
"""

case_analysis_prompt = ChatPromptTemplate.from_template(CASE_ANALYSIS_TEMPLATE)
case_analysis_chain = case_analysis_prompt | model

kb_considerations_prompt = ChatPromptTemplate.from_template(KB_CONSIDERATIONS_TEMPLATE)
kb_considerations_chain = kb_considerations_prompt | model


def parse_json_response(raw_response: str) -> dict:
    """Best-effort parser for local LLM JSON output."""
    if not raw_response:
        return {}

    match = re.search(r"\{.*\}", raw_response, re.DOTALL)
    if not match:
        return {}

    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}


def should_use_knowledge_base(question: str, *, has_case_file: bool) -> bool:
    """
    Decide whether to retrieve supporting knowledge-base evidence.

    Important safety behavior:
    - If the user supplied a denial letter and asks for summary/extraction/facts,
      answer from the case only.
    - Retrieve knowledge-base evidence only when the user asks for appeal support,
      policy/guideline help, strategy, or drafting.
    """
    q = question.lower()

    if has_case_file and any(term in q for term in CASE_ONLY_TERMS):
        return False

    return any(term in q for term in KNOWLEDGE_SUPPORT_TERMS)


def make_citation(doc):
    metadata = doc.metadata

    file_name = metadata.get("file_name", "Unknown file")
    file_type = metadata.get("file_type", "Unknown type")
    page = metadata.get("page")
    sheet = metadata.get("sheet_name")
    row = metadata.get("row_number")

    parts = [file_name, f"Type: {file_type}"]

    if file_type == "pdf" and page is not None:
        parts.append(f"PDF page: {page}")

    if file_type == "excel":
        if sheet:
            parts.append(f"Sheet: {sheet}")
        if row:
            parts.append(f"Row: {row}")

    return " | ".join(parts)


def format_docs(docs):
    formatted = []

    for i, doc in enumerate(docs, start=1):
        formatted.append(
            f"EVIDENCE BLOCK {i}\n"
            f"CITATION: {make_citation(doc)}\n"
            f"GENERAL KNOWLEDGE ONLY - DO NOT COPY PATIENT OR CLAIM FACTS FROM THIS BLOCK.\n\n"
            f"{doc.page_content}"
        )

    return "\n\n---\n\n".join(formatted)


def print_retrieved_evidence(docs):
    if not SHOW_DEBUG:
        return

    print("\nRetrieved general knowledge-base evidence:")

    for i, doc in enumerate(docs, start=1):
        citation = make_citation(doc)
        snippet = doc.page_content.replace("\n", " ")
        snippet = re.sub(r"\s+", " ", snippet)
        snippet = snippet[:700]

        print(f"\n{i}. {citation}")
        print(f"   {snippet}...")


def dedupe_docs(docs):
    seen = set()
    unique_docs = []

    for doc in docs:
        metadata = doc.metadata
        key = (
            metadata.get("source"),
            metadata.get("page"),
            metadata.get("sheet_name"),
            metadata.get("row_number"),
            metadata.get("chunk_index"),
            doc.page_content[:250],
        )

        if key in seen:
            continue

        seen.add(key)
        unique_docs.append(doc)

    return unique_docs


def parse_file_paths(file_input: str) -> list[str]:
    return [x.strip().strip('"') for x in file_input.split(",") if x.strip()]


def value_or_not_found(value) -> str:
    """Return a display-safe value for extracted facts."""
    if value is None:
        return "Not found in submitted file"

    value = str(value).strip()
    return value if value else "Not found in submitted file"


def format_extracted_case_facts(denial_info) -> str:
    """
    Deterministically render case-specific facts from the validated extraction object.

    This prevents the answer LLM from re-deciding whether patient/account/claim
    fields exist after Python already extracted them from the submitted file.
    """
    if denial_info is None:
        return "No submitted case file was loaded."

    data = denial_info.model_dump()

    fact_rows = [
        ("Patient name", data.get("patient_name")),
        ("Patient account number", data.get("patient_account_number")),
        ("Service date start", data.get("service_date_start")),
        ("Service date end", data.get("service_date_end")),
        ("Claim number", data.get("claim_number")),
        ("Payer / reviewer", data.get("provider_name")),
        ("Denial type", data.get("denial_type")),
        ("Before value", data.get("before_value")),
        ("After value", data.get("after_value")),
        ("DRG before value", data.get("drg_before_value")),
        ("DRG after value", data.get("drg_after_value")),
        ("Policy type", data.get("policy_type")),
        ("Summary", data.get("summary")),
    ]

    return "\n".join(
        f"- {label}: {value_or_not_found(value)}"
        for label, value in fact_rows
    )


def get_fact(denial_info, field_name: str) -> str:
    if denial_info is None:
        return "Not found in submitted file"
    return value_or_not_found(getattr(denial_info, field_name, None))


def clean_llm_section(value, default: str) -> str:
    if value is None:
        return default
    value = str(value).strip()
    if not value:
        return default
    return value


def build_patient_claim_section(denial_info) -> str:
    """This section is intentionally not generated by the LLM."""
    lines = [
        f"- Patient name: {get_fact(denial_info, 'patient_name')}",
        f"- Patient account number: {get_fact(denial_info, 'patient_account_number')}",
        f"- Service date start: {get_fact(denial_info, 'service_date_start')}",
        f"- Service date end: {get_fact(denial_info, 'service_date_end')}",
        f"- Claim number: {get_fact(denial_info, 'claim_number')}",
        f"- Payer / reviewer: {get_fact(denial_info, 'provider_name')}",
        f"- Before value: {get_fact(denial_info, 'before_value')}",
        f"- After value: {get_fact(denial_info, 'after_value')}",
        f"- DRG before value: {get_fact(denial_info, 'drg_before_value')}",
        f"- DRG after value: {get_fact(denial_info, 'drg_after_value')}",
    ]
    return "\n".join(lines)


def build_case_only_answer(denial_info, analysis: dict) -> str:
    """
    Build the final answer with Python-rendered fact sections.

    The model may summarize rationale, but it never writes the patient/account/claim/DRG section.
    That prevents contradictions like 'patient name not provided' when the extractor found it.
    """
    about = clean_llm_section(
        analysis.get("about"),
        get_fact(denial_info, "summary"),
    )
    rationale = clean_llm_section(
        analysis.get("key_rationale"),
        "The key rationale was not clearly summarized by the model; review the submitted letter text and structured extraction.",
    )
    missing = clean_llm_section(
        analysis.get("missing_uncertainty"),
        "No additional uncertainty identified from the submitted file.",
    )

    return (
        "1. What the submitted denial letter is about\n"
        f"{about}\n\n"
        "2. Denial category/type\n"
        f"{get_fact(denial_info, 'denial_type')}\n\n"
        "3. Relevant patient/account/claim information from the submitted file, including DRG before/after values if present\n"
        f"{build_patient_claim_section(denial_info)}\n\n"
        "4. Key denial rationale from the submitted file\n"
        f"{rationale}\n\n"
        "5. Missing information or uncertainty\n"
        f"{missing}"
    )


def build_knowledge_answer(denial_info, case_analysis: dict, kb_analysis: dict) -> str:
    about = clean_llm_section(
        case_analysis.get("about"),
        get_fact(denial_info, "summary"),
    )
    considerations = clean_llm_section(
        kb_analysis.get("policy_guideline_appeal_considerations"),
        "No specific policy/guideline/appeal considerations were identified from the retrieved knowledge base.",
    )
    missing = clean_llm_section(
        kb_analysis.get("missing_uncertainty") or case_analysis.get("missing_uncertainty"),
        "No additional uncertainty identified from the submitted file.",
    )

    return (
        "1. What the submitted denial letter is about\n"
        f"{about}\n\n"
        "2. Denial category/type\n"
        f"{get_fact(denial_info, 'denial_type')}\n\n"
        "3. Relevant patient/account/claim information from the submitted file, including DRG before/after values if present\n"
        f"{build_patient_claim_section(denial_info)}\n\n"
        "4. Relevant policy/guideline/appeal considerations\n"
        f"{considerations}\n\n"
        "5. Missing information or uncertainty\n"
        f"{missing}"
    )


def run_case_analysis(denial_json: str, extracted_case_facts: str, case_context: str) -> dict:
    raw = case_analysis_chain.invoke(
        {
            "denial_json": denial_json,
            "extracted_case_facts": extracted_case_facts,
            "case_context": case_context[:12000],
        }
    )
    return parse_json_response(raw)


while True:
    print("\n\n-------------------------------")
    file_input = input("Enter denial letter file path(s), comma-separated, or press Enter to skip: ")
    question = input("Ask your question (q to quit): ")

    if question.lower() == "q":
        break

    case_context = ""
    denial_info = None
    has_case_file = bool(file_input.strip())

    if has_case_file:
        file_paths = parse_file_paths(file_input)
        case_context = load_case_files(file_paths)

        if len(case_context.strip()) < 200:
            print("Could not extract enough text from this file. The PDF may need OCR or may be unreadable.")
            continue

        denial_info = extract_denial_info(case_context)

        print("\nStructured JSON:")
        print(denial_info.model_dump_json(indent=2))

    denial_json = denial_info.model_dump_json(indent=2) if denial_info else "{}"
    extracted_case_facts = format_extracted_case_facts(denial_info)

    if denial_info:
        print("\nVerified Extracted Case Facts:")
        print(extracted_case_facts)

    case_analysis = run_case_analysis(denial_json, extracted_case_facts, case_context) if has_case_file else {}
    use_kb = should_use_knowledge_base(question, has_case_file=has_case_file)

    if use_kb:
        # Keep retrieval general. Do not send patient/account/claim details into retrieval.
        # This avoids pulling similar prior denial letters based on patient-specific facts.
        retrieval_question = question

        if denial_info:
            safe_fields = {
                "denial_type": denial_info.denial_type,
                "policy_type": denial_info.policy_type,
                "provider_name": denial_info.provider_name,
                "drg_before_value": getattr(denial_info, "drg_before_value", None),
                "drg_after_value": getattr(denial_info, "drg_after_value", None),
                "summary": denial_info.summary,
            }
            retrieval_question += "\n\nGeneral denial context, not patient identifiers:\n" + str(safe_fields)

        docs = retrieve_supporting_knowledge(retrieval_question)
        docs = dedupe_docs(docs)
        print_retrieved_evidence(docs)
        knowledge_context = format_docs(docs)

        raw_kb = kb_considerations_chain.invoke(
            {
                "denial_json": denial_json,
                "extracted_case_facts": extracted_case_facts,
                "case_context": case_context[:12000],
                "knowledge_context": knowledge_context,
                "question": question,
            }
        )
        kb_analysis = parse_json_response(raw_kb)
        result = build_knowledge_answer(denial_info, case_analysis, kb_analysis)
    else:
        result = build_case_only_answer(denial_info, case_analysis)

    print("\nAnswer:")
    print(result)
