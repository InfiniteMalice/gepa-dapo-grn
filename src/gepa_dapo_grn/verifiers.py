"""Verifier protocol and result helpers for verifier-first training."""

from __future__ import annotations

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

        tags = dict(self.diagnostics)
        if self.passed is not None:
            tags["verifier_pass"] = float(self.passed)
            tags[success_key] = float(self.passed)
            tags["verifier_success"] = tags[success_key]
        if self.score is not None:
            tags["verifier_score"] = float(self.score)
            if self.passed is None:
                tags[success_key] = float(self.score)
                tags["verifier_success"] = tags[success_key]
        if self.confidence is not None:
            tags["verifier_confidence"] = float(self.confidence)
        tags["verifier_coverage"] = float(self.coverage)
        return tags


class Verifier(Protocol):
    """Protocol for verifier integrations."""

    def verify(self, sample: Any) -> VerifierResult:
        """Return verifier output for a sample."""
