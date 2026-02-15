"""Interfaces for GEPA-style feedback and verifier-driven hooks."""

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
    coverage: Optional[float] = None
    diagnostics: Dict[str, float] = field(default_factory=dict)

    def as_tags(self) -> Dict[str, float]:
        """Flatten verifier output into numeric tags."""

        tags = dict(self.diagnostics)
        if self.passed is not None:
            tags["verifier_pass"] = float(self.passed)
        if self.score is not None:
            tags["verifier_score"] = float(self.score)
        if self.confidence is not None:
            tags["verifier_confidence"] = float(self.confidence)
        if self.coverage is not None:
            tags["verifier_coverage"] = float(self.coverage)
        return tags


class Verifier(Protocol):
    """Protocol for verifier integrations."""

    def verify(self, sample: Any) -> VerifierResult:
        """Return verifier output for a sample."""


@dataclass(slots=True)
class GEPAFeedback:
    """Structured feedback for GEPA-style training signals.

    Args:
        rewards: Vector-valued reward dimensions.
        tags: Auxiliary numeric signals (e.g., calibration error).
        verifier: Optional flattened verifier signals.
        meta: Metadata such as task or prompt identifiers.
        abstained: Whether the model abstained on the task.
    """

    rewards: Dict[str, float] = field(default_factory=dict)
    tags: Dict[str, float] = field(default_factory=dict)
    verifier: Dict[str, float] = field(default_factory=dict)
    meta: Dict[str, str] = field(default_factory=dict)
    abstained: bool = False

    def to_dict(self) -> Dict[str, object]:
        """Return a JSON-serializable dictionary representation."""

        return {
            "rewards": dict(self.rewards),
            "tags": dict(self.tags),
            "verifier": dict(self.verifier),
            "meta": dict(self.meta),
            "abstained": self.abstained,
        }
