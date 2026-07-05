"""Active-GRPO-style imitate/reinforce scheduling."""

from __future__ import annotations

from enum import Enum
from typing import Optional


class TrainingMode(str, Enum):
    """Training mode selected for one prompt."""

    IMITATE = "imitate"
    REINFORCE = "reinforce"
    MIXED = "mixed"


class ActiveGRPOScheduler:
    """Choose whether to imitate a reference or reinforce a better policy candidate."""

    def __init__(
        self,
        margin: float = 0.02,
        mixed_band: float = 0.01,
        min_reference_score: Optional[float] = None,
    ) -> None:
        self.margin = float(margin)
        self.mixed_band = float(mixed_band)
        if self.margin < 0.0:
            raise ValueError("margin must be non-negative")
        if self.mixed_band < 0.0:
            raise ValueError("mixed_band must be non-negative")
        if self.mixed_band > self.margin:
            raise ValueError("mixed_band must be less than or equal to margin")
        self.min_reference_score = min_reference_score

    def choose_mode(
        self,
        reference_score: float,
        best_policy_score: Optional[float],
    ) -> TrainingMode:
        if best_policy_score is None:
            return TrainingMode.IMITATE
        if self.min_reference_score is not None and reference_score < self.min_reference_score:
            return TrainingMode.REINFORCE
        score_delta = best_policy_score - reference_score
        if score_delta > self.margin:
            return TrainingMode.REINFORCE
        if abs(score_delta) <= self.mixed_band:
            return TrainingMode.MIXED
        return TrainingMode.IMITATE

    def imitation_weight(
        self,
        reference_score: float,
        best_policy_score: Optional[float],
    ) -> float:
        mode = self.choose_mode(reference_score, best_policy_score)
        if mode == TrainingMode.IMITATE:
            return 1.0
        if mode == TrainingMode.REINFORCE:
            return 0.0
        return 0.5

    def reinforcement_weight(
        self,
        reference_score: float,
        best_policy_score: Optional[float],
    ) -> float:
        return 1.0 - self.imitation_weight(reference_score, best_policy_score)
