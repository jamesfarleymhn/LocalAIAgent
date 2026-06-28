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


    def _post_ollama(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        request = urllib.request.Request(
            f"{self.base_url.rstrip('/')}{path}",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                raw = response.read().decode("utf-8", errors="replace")
                return json.loads(raw) if raw else {}
        except (TimeoutError, socket.timeout) as exc:
            raise RuntimeError(
                f"Local Ollama vision model timed out after {self.timeout_seconds} seconds. "
                "Try fewer pages, a smaller image zoom, a faster vision model, or increase --ollama-timeout."
            ) from exc
        except urllib.error.HTTPError as exc:
            detail = ""
            try:
                detail = exc.read().decode("utf-8", errors="replace")[:1000]
            except Exception:
                detail = str(exc)
            raise RuntimeError(f"Ollama HTTP error {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(
                "Could not reach local Ollama. Start Ollama and pull the configured vision model."
            ) from exc

    def generate_text_with_images(self, prompt: str, image_base64_list: list[str], *, temperature: float = 0.0) -> str:
        """Generate from a local Ollama vision-capable model.

        Uses Ollama /api/chat first because current Qwen-VL style models are
        more reliable with image messages there. Falls back to /api/generate
        for older LLaVA-style models. image_base64_list must contain base64
        image bytes without a data URI prefix.
        """
        chat_payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                    "images": image_base64_list,
                }
            ],
            "stream": False,
            "options": {"temperature": temperature},
        }
        try:
            body = self._post_ollama("/api/chat", chat_payload)
            message = body.get("message") or {}
            content = message.get("content")
            if content:
                return str(content)
            # If chat succeeded but returned no content, fall through to generate.
        except RuntimeError as chat_exc:
            # Some older Ollama installs/models only support /api/generate with images.
            # Keep the exception and try generate before failing.
            last_error = chat_exc
        else:
            last_error = None

        generate_payload = {
            "model": self.model,
            "prompt": prompt,
            "images": image_base64_list,
            "stream": False,
            "options": {"temperature": temperature},
        }
        try:
            body = self._post_ollama("/api/generate", generate_payload)
            return str(body.get("response", ""))
        except RuntimeError as generate_exc:
            if last_error is not None:
                raise RuntimeError(f"Ollama vision failed with /api/chat and /api/generate. chat={last_error}; generate={generate_exc}") from generate_exc
            raise

    def generate_json_with_images(self, prompt: str, image_base64_list: list[str], *, temperature: float = 0.0) -> dict[str, Any]:
        raw = self.generate_text_with_images(prompt, image_base64_list, temperature=temperature)
        parsed = extract_json_object(raw)
        if not parsed:
            # Preserve a short raw response for caller diagnostics.
            return {"_parse_failed": True, "_raw_response_preview": raw[:2000]}
        return parsed

    def generate_json(self, prompt: str, *, temperature: float = 0.0) -> dict[str, Any]:
        return extract_json_object(self.generate_text(prompt, temperature=temperature))
