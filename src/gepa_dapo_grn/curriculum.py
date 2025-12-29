"""Curriculum tracking utilities for GEPA-style feedback."""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional

from gepa_dapo_grn.ema_utils import update_ema
from gepa_dapo_grn.gepa_interfaces import GEPAFeedback


@dataclass(slots=True)
class TaskStats:
    """EMA statistics for a single task."""

    reward_ema: Dict[str, float] = field(default_factory=dict)
    tag_ema: Dict[str, float] = field(default_factory=dict)
    abstention_ema: float = 0.0
    count: int = 0


class CurriculumTracker:
    """Track per-task EMAs to drive sampling decisions."""

    def __init__(
        self,
        decay: float = 0.9,
        reward_weights: Optional[Dict[str, float]] = None,
        tag_weights: Optional[Dict[str, float]] = None,
        abstention_weight: float = 0.5,
        weight_fn: Optional[Callable[[TaskStats], float]] = None,
    ) -> None:
        if not 0.0 < decay < 1.0:
            raise ValueError("decay must be between 0 and 1")
        self.decay = decay
        self.reward_weights = reward_weights or {}
        self.tag_weights = tag_weights or {}
        self.abstention_weight = abstention_weight
        self.weight_fn = weight_fn
        self.tasks: Dict[str, TaskStats] = {}

    def update(self, task_id: str, feedback: GEPAFeedback) -> TaskStats:
        """Update EMA statistics for a task based on new feedback."""

        stats = self.tasks.setdefault(task_id, TaskStats())
        for key, value in feedback.rewards.items():
            current = stats.reward_ema.get(key, float(value))
            stats.reward_ema[key] = update_ema(current, float(value), self.decay)
        for key, value in feedback.tags.items():
            current = stats.tag_ema.get(key, float(value))
            stats.tag_ema[key] = update_ema(current, float(value), self.decay)
        stats.abstention_ema = update_ema(
            stats.abstention_ema,
            float(feedback.abstained),
            self.decay,
        )
        stats.count += 1
        return stats

    def _weighted_score(self, stats: TaskStats) -> float:
        score = 0.0
        if self.reward_weights:
            for key, weight in self.reward_weights.items():
                score += weight * stats.reward_ema.get(key, 0.0)
        else:
            for value in stats.reward_ema.values():
                score += value

        if self.tag_weights:
            for key, weight in self.tag_weights.items():
                score += weight * stats.tag_ema.get(key, 0.0)

        score -= self.abstention_weight * stats.abstention_ema
        return score

    def sample_weight(self, task_id: str) -> float:
        """Compute a sampling weight for a task based on EMA heuristics."""

        stats = self.tasks.get(task_id)
        if stats is None:
            return 1.0
        if self.weight_fn is not None:
            return max(0.0, float(self.weight_fn(stats)))
        score = self._weighted_score(stats)
        if not math.isfinite(score):
            return math.exp(700.0) if score > 0.0 else 0.0
        max_score = min(700.0, math.log(sys.float_info.max))
        return math.exp(min(score, max_score))

    def describe_task(self, task_id: str) -> Dict[str, float]:
        """Return a summary of EMA statistics for a task."""

        stats = self.tasks.get(task_id)
        if stats is None:
            return {}
        summary = {
            **{f"reward_ema/{key}": value for key, value in stats.reward_ema.items()},
            **{f"tag_ema/{key}": value for key, value in stats.tag_ema.items()},
            "abstention_ema": stats.abstention_ema,
            "count": float(stats.count),
        }
        return summary
