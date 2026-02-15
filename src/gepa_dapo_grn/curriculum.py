"""Curriculum tracking and composition utilities for GEPA-style feedback."""

from __future__ import annotations

import math
import random
import sys
from dataclasses import field
from typing import Any, Callable, Dict, List, Optional, Protocol, Sequence

from gepa_dapo_grn._compat import dataclass
from gepa_dapo_grn._ema_helpers import _update_ema
from gepa_dapo_grn.gepa_interfaces import GEPAFeedback


class TaskComposer(Protocol):
    """Protocol for composing task items into harder variants."""

    def compose(self, items: List[Any], depth: int) -> Any:
        """Compose a list of items at a given composition depth."""


@dataclass(slots=True)
class SimpleTextComposer:
    """Simple text composer that concatenates prompts with separators."""

    separator: str = "\n\n---\n\n"

    def compose(self, items: List[Any], depth: int) -> str:
        chunk = [str(item) for item in items]
        return f"[depth={depth}] " + self.separator.join(chunk)


@dataclass(slots=True)
class TaskStats:
    """EMA statistics for a single task."""

    reward_ema: Dict[str, float] = field(default_factory=dict)
    tag_ema: Dict[str, float] = field(default_factory=dict)
    verifier_ema: Dict[str, float] = field(default_factory=dict)
    abstention_ema: float = 0.0
    verifier_pass_rate_ema: float = 0.0
    coverage_ema: float = 0.0
    difficulty_ema: float = 0.5
    composition_depth: int = 0
    saturated: bool = False
    count: int = 0


class CurriculumTracker:
    """Track per-task EMAs to drive saturation-aware sampling."""

    def __init__(
        self,
        decay: float = 0.9,
        reward_weights: Optional[Dict[str, float]] = None,
        tag_weights: Optional[Dict[str, float]] = None,
        abstention_weight: float = 0.5,
        weight_fn: Optional[Callable[[TaskStats], float]] = None,
        saturation_pass_rate: float = 0.95,
        saturation_reward_threshold: float = 0.95,
        min_samples_for_saturation: int = 10,
        depth_step: int = 1,
        max_depth: int = 4,
    ) -> None:
        if not 0.0 < decay < 1.0:
            raise ValueError("decay must be strictly between 0 and 1 (exclusive)")
        self.decay = decay
        self.reward_weights = reward_weights or {}
        self.tag_weights = tag_weights or {}
        self.abstention_weight = abstention_weight
        self.weight_fn = weight_fn
        self.saturation_pass_rate = saturation_pass_rate
        self.saturation_reward_threshold = saturation_reward_threshold
        self.min_samples_for_saturation = min_samples_for_saturation
        self.depth_step = depth_step
        self.max_depth = max_depth
        self.tasks: Dict[str, TaskStats] = {}

    def _mean_reward(self, feedback: GEPAFeedback) -> float:
        if not feedback.rewards:
            return 0.0
        return sum(float(v) for v in feedback.rewards.values()) / len(feedback.rewards)

    def _verifier_pass(self, feedback: GEPAFeedback) -> float:
        if "verifier_pass" in feedback.verifier:
            return float(feedback.verifier["verifier_pass"])
        if "verifier_pass" in feedback.tags:
            return float(feedback.tags["verifier_pass"])
        return 0.0

    def _coverage(self, feedback: GEPAFeedback) -> float:
        if "verifier_coverage" in feedback.verifier:
            return float(feedback.verifier["verifier_coverage"])
        if "coverage" in feedback.verifier:
            return float(feedback.verifier["coverage"])
        return 1.0

    def update(self, task_id: str, feedback: GEPAFeedback) -> TaskStats:
        """Update EMA statistics for a task based on new feedback."""

        stats = self.tasks.setdefault(task_id, TaskStats())
        for key, value in feedback.rewards.items():
            current = stats.reward_ema.get(key, float(value))
            stats.reward_ema[key] = _update_ema(current, float(value), self.decay)
        for key, value in feedback.tags.items():
            current = stats.tag_ema.get(key, float(value))
            stats.tag_ema[key] = _update_ema(current, float(value), self.decay)
        for key, value in feedback.verifier.items():
            current = stats.verifier_ema.get(key, float(value))
            stats.verifier_ema[key] = _update_ema(current, float(value), self.decay)

        mean_reward = self._mean_reward(feedback)
        stats.difficulty_ema = _update_ema(stats.difficulty_ema, 1.0 - mean_reward, self.decay)
        stats.verifier_pass_rate_ema = _update_ema(
            stats.verifier_pass_rate_ema,
            self._verifier_pass(feedback),
            self.decay,
        )
        stats.coverage_ema = _update_ema(stats.coverage_ema, self._coverage(feedback), self.decay)
        stats.abstention_ema = _update_ema(
            stats.abstention_ema, float(feedback.abstained), self.decay
        )
        stats.count += 1

        reward_saturated = mean_reward >= self.saturation_reward_threshold
        verifier_saturated = stats.verifier_pass_rate_ema >= self.saturation_pass_rate
        enough_samples = stats.count >= self.min_samples_for_saturation
        stats.saturated = enough_samples and (reward_saturated or verifier_saturated)
        if stats.saturated:
            stats.composition_depth = min(self.max_depth, stats.composition_depth + self.depth_step)

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
        score += stats.difficulty_ema
        if stats.saturated:
            score -= 1.0
        score += 0.25 * stats.composition_depth
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

    def choose_task(self, task_ids: Sequence[str]) -> str:
        """Sample a task id according to dynamic curriculum weights."""

        weights = [self.sample_weight(task_id) for task_id in task_ids]
        return random.choices(list(task_ids), weights=weights, k=1)[0]

    def current_depth(self, task_id: str) -> int:
        """Return current composition depth for task."""

        stats = self.tasks.get(task_id)
        if stats is None:
            return 0
        return stats.composition_depth

    def describe_task(self, task_id: str) -> Dict[str, float]:
        """Return a summary of EMA statistics for a task."""

        stats = self.tasks.get(task_id)
        if stats is None:
            return {}
        return {
            **{f"reward_ema/{key}": value for key, value in stats.reward_ema.items()},
            **{f"tag_ema/{key}": value for key, value in stats.tag_ema.items()},
            **{f"verifier_ema/{key}": value for key, value in stats.verifier_ema.items()},
            "abstention_ema": stats.abstention_ema,
            "verifier_pass_rate_ema": stats.verifier_pass_rate_ema,
            "coverage": stats.coverage_ema,
            "difficulty_ema": stats.difficulty_ema,
            "composition_depth": float(stats.composition_depth),
            "saturated": float(stats.saturated),
            "count": float(stats.count),
        }
