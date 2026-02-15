"""Simple logging utilities for metrics."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, Optional

from gepa_dapo_grn.gepa_interfaces import GEPAFeedback


def summarize_feedback(feedbacks: Iterable[GEPAFeedback]) -> Dict[str, float]:
    """Summarize full feedback payloads for logging pipelines."""

    metrics: Dict[str, float] = {}
    feedback_list = list(feedbacks)
    count = max(1, len(feedback_list))

    abstained_total = 0.0
    for feedback in feedback_list:
        abstained_total += float(feedback.abstained)
        for key, value in feedback.rewards.items():
            metric_key = f"feedback/rewards/{key}"
            metrics[metric_key] = metrics.get(metric_key, 0.0) + float(value)
        for key, value in feedback.tags.items():
            metric_key = f"feedback/tags/{key}"
            metrics[metric_key] = metrics.get(metric_key, 0.0) + float(value)
        for key, value in feedback.verifier.items():
            metric_key = f"feedback/verifier/{key}"
            metrics[metric_key] = metrics.get(metric_key, 0.0) + float(value)

    for key in list(metrics.keys()):
        metrics[key] /= count
    metrics["feedback/abstained"] = abstained_total / count

    if feedback_list:
        depths = [
            float(feedback.meta.get("composition_depth", 0.0))
            for feedback in feedback_list
            if "composition_depth" in feedback.meta
        ]
        if depths:
            metrics["composition_depth"] = sum(depths) / len(depths)

    if "feedback/verifier/verifier_pass" in metrics:
        metrics["verifier_pass_rate_ema"] = metrics["feedback/verifier/verifier_pass"]
    if "feedback/verifier/verifier_coverage" in metrics:
        metrics["coverage"] = metrics["feedback/verifier/verifier_coverage"]

    return metrics


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
