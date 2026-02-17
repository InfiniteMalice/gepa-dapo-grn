"""Internal helpers for release/build packaging workflows."""

from __future__ import annotations

from pathlib import Path


def find_single_wheel(dist_dir: Path, package_prefix: str, version: str) -> Path:
    """Return exactly one wheel path for ``package_prefix`` and ``version``.

    Raises:
        FileNotFoundError: If no matching wheel exists.
    """
    pattern = f"{package_prefix}-{version}-*.whl"
    matches = list(dist_dir.glob(pattern))
    if not matches:
        raise FileNotFoundError(
            f"No wheel found matching {pattern!r} in {dist_dir}. Build first with python -m build."
        )
    return max(matches, key=lambda path: path.stat().st_mtime)
