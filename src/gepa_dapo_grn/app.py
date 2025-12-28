"""Application utilities for the gepa-dapo-grn project."""

from __future__ import annotations

DEFAULT_VERSION = "0.1.0"


def get_version(version: str | None = None) -> str:
    """Return the configured version for the application.

    Args:
        version: Optional version override.

    Returns:
        The version string for the application.
    """

    return version or DEFAULT_VERSION
