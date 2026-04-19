"""Verifier protocol and result helpers for verifier-first training."""

from __future__ import annotations

import math
from dataclasses import field
from typing import Any, Dict, Optional, Protocol

from gepa_dapo_grn._compat import dataclass


@dataclass(slots=True)
class VerifierResult:
    """Generic verifier outputs for teacherless or verifier-first workflows."""

    passed: Optional[bool] = None
    score: Optional[float] = None
    confidence: Optional[float] = None
    coverage: float = 1.0
    diagnostics: Dict[str, float] = field(default_factory=dict)

    def as_tags(self, success_key: str = "verifier_success") -> Dict[str, float]:
        """Flatten verifier outputs into numeric tags."""

        final_key = str(success_key).strip()
        if not final_key:
            final_key = "verifier_success"
        tags: Dict[str, float] = {}
        for key, value in self.diagnostics.items():
            key_str = str(key).strip()
            if not key_str or key_str == "None":
                continue
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(numeric):
                tags[key_str] = numeric
        if self.passed is not None:
            try:
                passed_value = float(self.passed)
            except (TypeError, ValueError):
                passed_value = None
            if passed_value is not None and math.isfinite(passed_value):
                tags["verifier_pass"] = passed_value
                tags[final_key] = passed_value
                tags["verifier_success"] = tags[final_key]
        if self.score is not None:
            try:
                score_value = float(self.score)
            except (TypeError, ValueError):
                score_value = None
            if score_value is not None and math.isfinite(score_value):
                tags["verifier_score"] = score_value
                if self.passed is None:
                    tags[final_key] = score_value
                    tags["verifier_success"] = tags[final_key]
        if self.confidence is not None:
            try:
                confidence_value = float(self.confidence)
            except (TypeError, ValueError):
                confidence_value = None
            if confidence_value is not None and math.isfinite(confidence_value):
                tags["verifier_confidence"] = confidence_value
        if self.coverage is not None:
            try:
                coverage_value = float(self.coverage)
            except (TypeError, ValueError):
                coverage_value = None
            if coverage_value is not None and math.isfinite(coverage_value):
                tags["verifier_coverage"] = coverage_value
        return tags


class Verifier(Protocol):
    """Protocol for verifier integrations."""

    def verify(self, sample: Any) -> VerifierResult:
        """Return verifier output for a sample."""
