from __future__ import annotations

import sys
from dataclasses import dataclass, field
from datetime import datetime
from time import perf_counter


@dataclass
class Progress:
    """Small stderr progress logger for long local OCR/LLM/RAG runs."""

    enabled: bool = True
    start_time: float = field(default_factory=perf_counter)

    def log(self, message: str) -> None:
        if not self.enabled:
            return
        now = datetime.now().strftime("%H:%M:%S")
        elapsed = perf_counter() - self.start_time
        print(f"[{now} +{elapsed:0.1f}s] {message}", file=sys.stderr, flush=True)

    def step(self, number: int, total: int, message: str) -> None:
        self.log(f"[{number}/{total}] {message}")
