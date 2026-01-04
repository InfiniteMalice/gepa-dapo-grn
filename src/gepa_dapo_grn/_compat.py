"""Compatibility helpers for multiple Python versions."""

from __future__ import annotations

from dataclasses import dataclass as _dataclass

try:
    _dataclass(slots=True)  # type: ignore[call-arg]
except TypeError:  # pragma: no cover - Python < 3.10
    _SUPPORTS_SLOTS = False
else:
    _SUPPORTS_SLOTS = True


def dataclass(*args, **kwargs):
    """Return a dataclass decorator that ignores slots on older Python versions."""

    if not _SUPPORTS_SLOTS and "slots" in kwargs:
        kwargs = dict(kwargs)
        kwargs.pop("slots", None)
    return _dataclass(*args, **kwargs)
