from __future__ import annotations

import json
import socket
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from config import CONFIG
from json_utils import extract_json_object


@dataclass
class LocalLLM:
    """Small local Ollama client using the standard library.

    This avoids tying the case-only workflow to LangChain import changes. No data
    is sent anywhere except the local Ollama endpoint configured here.
    """

    model: str = CONFIG.generation_model
    base_url: str = CONFIG.ollama_url
    timeout_seconds: int = CONFIG.ollama_timeout_seconds

    def generate_text(self, prompt: str, *, temperature: float = 0.0) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature},
        }
        request = urllib.request.Request(
            f"{self.base_url.rstrip('/')}/api/generate",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                body = json.loads(response.read().decode("utf-8"))
        except (TimeoutError, socket.timeout) as exc:
            raise RuntimeError(
                f"Local Ollama timed out after {self.timeout_seconds} seconds while generating a response. "
                "Try a smaller/faster model, increase --ollama-timeout, or run with --no-llm for extraction-only fallback."
            ) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(
                "Could not reach local Ollama. Start Ollama and pull the configured model, "
                "or run with --no-llm for regex-only extraction."
            ) from exc

        return str(body.get("response", ""))

    def generate_json(self, prompt: str, *, temperature: float = 0.0) -> dict[str, Any]:
        return extract_json_object(self.generate_text(prompt, temperature=temperature))
