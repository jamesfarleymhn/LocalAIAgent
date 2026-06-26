from __future__ import annotations

from typing import Any


def render_prompt(template: str, **values: Any) -> str:
    """Render prompt placeholders without treating JSON braces as format fields.

    Python str.format() breaks when prompt templates contain literal JSON examples:
    {"plain_english_summary": null}

    This function only replaces exact placeholders such as {question} or
    {extraction_json}. All other braces remain untouched.
    """
    rendered = template
    for key, value in values.items():
        rendered = rendered.replace("{" + key + "}", str(value))
    return rendered
