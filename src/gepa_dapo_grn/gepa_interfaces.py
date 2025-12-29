"""Interfaces for GEPA-style feedback."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict


@dataclass(slots=True)
class GEPAFeedback:
    """Structured feedback for GEPA-style training signals.

    Args:
        rewards: Vector-valued reward dimensions.
        tags: Auxiliary numeric signals (e.g., calibration error).
        meta: Metadata such as task or prompt identifiers.
        abstained: Whether the model abstained on the task.
    """

    rewards: Dict[str, float] = field(default_factory=dict)
    tags: Dict[str, float] = field(default_factory=dict)
    meta: Dict[str, str] = field(default_factory=dict)
    abstained: bool = False

    def to_dict(self) -> Dict[str, object]:
        """Return a JSON-serializable dictionary representation."""

        return {
            "rewards": dict(self.rewards),
            "tags": dict(self.tags),
            "meta": dict(self.meta),
            "abstained": self.abstained,
        }
