"""Internal helpers for release/build packaging workflows."""

from __future__ import annotations

from pathlib import Path


def find_single_wheel(dist_dir: Path, package_prefix: str, version: str) -> Path:
    """Return exactly one wheel path for ``package_prefix`` and ``version``.

    Raises:
        FileNotFoundError: If no matching wheel exists.
        FileExistsError: If multiple matching wheels exist.
    """
    pattern = f"{package_prefix}-{version}-*.whl"
    matches = list(dist_dir.glob(pattern))
    if not matches:
        raise FileNotFoundError(
            f"No wheel found matching {pattern!r} in {dist_dir}. "
            "Build first with python -m build."
        )
    if len(matches) > 1:
        ambiguous_paths = ", ".join(str(path) for path in sorted(matches))
        raise FileExistsError(
            f"Multiple wheels matched {pattern!r} in {dist_dir}: {ambiguous_paths}. "
            "Clean dist/ or remove duplicate wheels before installing."
        )

    return matches[0]
