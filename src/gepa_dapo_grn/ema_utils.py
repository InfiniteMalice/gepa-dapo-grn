"""Shared utilities for EMA updates."""

from __future__ import annotations


def update_ema(current: float, value: float, decay: float) -> float:
    """Return the exponential moving average update."""

    return decay * current + (1.0 - decay) * value
