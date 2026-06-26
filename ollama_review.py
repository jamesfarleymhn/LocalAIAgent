from __future__ import annotations

import json
import urllib.request
import urllib.error
from typing import Any


def ollama_generate(prompt: str, model: str = "llama3.1:latest", timeout: int = 600) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.0},
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        "http://localhost:11434/api/generate",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    return body.get("response", "")


def build_review_prompt(facts: dict[str, Any], question: str) -> str:
    compact = {
        "summary_for_human": facts.get("summary_for_human", {}),
        "diagnosis_code_candidates": facts.get("diagnosis_code_candidates", [])[:20],
        "procedure_code_candidates": facts.get("procedure_code_candidates", [])[:20],
        "top_evidence": facts.get("evidence", [])[:20],
        "candidate_sections": facts.get("candidate_sections", [])[:5],
    }
    return f"""
You are reviewing extracted facts from one healthcare denial PDF.
Use ONLY the extracted facts and evidence below. Do not invent missing values.

User question: {question}

Return a readable case review with these headings:
1. Bottom-line summary
2. Payer / payee / patient / claim facts
3. DRG and coding change
4. Payer rationale
5. What needs manual verification

Extracted facts and evidence JSON:
{json.dumps(compact, indent=2)}
""".strip()
