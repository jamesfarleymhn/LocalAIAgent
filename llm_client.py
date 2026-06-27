from __future__ import annotations

import json
import os
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

    model: str | None = None
    base_url: str = CONFIG.ollama_url
    timeout_seconds: int = CONFIG.ollama_timeout_seconds

    def __post_init__(self) -> None:
        if not self.model:
            self.model = os.getenv("OLLAMA_MODEL", CONFIG.generation_model)

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


    def generate_text_with_images(self, prompt: str, image_base64_list: list[str], *, temperature: float = 0.0) -> str:
        """Generate a response from a local Ollama vision-capable model.

        image_base64_list must contain base64-encoded image bytes without a data URI prefix.
        This still sends data only to the configured local Ollama endpoint.
        """
        payload = {
            "model": self.model,
            "prompt": prompt,
            "images": image_base64_list,
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
                f"Local Ollama vision model timed out after {self.timeout_seconds} seconds. "
                "Try fewer --vision-max-pages, a smaller image zoom, a faster vision model, or increase --ollama-timeout."
            ) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(
                "Could not reach local Ollama. Start Ollama and pull the configured vision model."
            ) from exc

        return str(body.get("response", ""))

    def generate_json_with_images(self, prompt: str, image_base64_list: list[str], *, temperature: float = 0.0) -> dict[str, Any]:
        return extract_json_object(self.generate_text_with_images(prompt, image_base64_list, temperature=temperature))

    def generate_json(self, prompt: str, *, temperature: float = 0.0) -> dict[str, Any]:
        return extract_json_object(self.generate_text(prompt, temperature=temperature))
