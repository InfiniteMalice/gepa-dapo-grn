"""Curriculum tracking utilities for reward-driven sampling."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List

import torch


@dataclass
class CurriculumState:
    """Holds EMA statistics for reward dimensions."""

    ema: Dict[str, float] = field(default_factory=dict)
    count: int = 0


class CurriculumTracker:
    """Track exponential moving averages of rewards to drive sampling decisions."""

    def __init__(self, decay: float = 0.9) -> None:
        if not 0.0 < decay < 1.0:
            raise ValueError("decay must be between 0 and 1")
        self.decay = decay
        self.state = CurriculumState()

    def update(self, reward_vectors: Iterable[Dict[str, float]]) -> Dict[str, float]:
        """Update EMA statistics with new reward vectors."""

        for vector in reward_vectors:
            for key, value in vector.items():
                previous = self.state.ema.get(key, float(value))
                updated = self.decay * previous + (1.0 - self.decay) * float(value)
                self.state.ema[key] = updated
            self.state.count += 1
        return dict(self.state.ema)

    def sampling_weights(self, keys: List[str]) -> torch.Tensor:
        """Compute sampling weights based on inverse EMA values.

        Higher weights correspond to lower EMA rewards, encouraging focus on hard cases.
        """

        weights = torch.tensor(
            [1.0 / (self.state.ema.get(key, 1.0) + 1e-6) for key in keys],
            dtype=torch.float32,
        )
        return weights / weights.sum()

    def describe(self) -> Dict[str, float]:
        """Return a dictionary of EMA statistics."""

        return dict(self.state.ema)
