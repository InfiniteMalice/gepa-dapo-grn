"""Simple logging utilities for metrics."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional


@dataclass
class MetricsLogger:
    """Log metrics to stdout and optionally to a JSONL file."""

    jsonl_path: Optional[Path] = None
    prefix: str = ""
    buffer: Dict[str, float] = field(default_factory=dict)

    def log(self, metrics: Dict[str, float]) -> None:
        self.buffer.update(metrics)
        timestamp = datetime.utcnow().isoformat()
        payload = {"timestamp": timestamp, **self.buffer}
        prefix = f"{self.prefix} " if self.prefix else ""
        print(prefix + json.dumps(payload, sort_keys=True))
        if self.jsonl_path:
            self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
            with self.jsonl_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload) + "\n")

    def reset(self) -> None:
        self.buffer.clear()
