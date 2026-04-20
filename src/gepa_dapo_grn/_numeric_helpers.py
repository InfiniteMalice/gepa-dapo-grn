"""Numeric helper utilities shared across training components."""

from __future__ import annotations

import math
from typing import Optional


def finite_or_none(value: object) -> Optional[float]:
    """Return finite float(value) or None when conversion fails/non-finite."""

    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(parsed):
        return None
    return parsed
