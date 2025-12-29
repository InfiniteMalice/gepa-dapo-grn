"""Shared utilities for EMA updates."""

from __future__ import annotations


def update_ema(current: float, value: float, decay: float) -> float:
    """Return the exponential moving average update."""

    if not 0.0 <= decay <= 1.0:
        raise ValueError(f"decay must be in [0, 1], got {decay}")
    return decay * current + (1.0 - decay) * value
