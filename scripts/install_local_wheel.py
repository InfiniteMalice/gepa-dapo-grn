#!/usr/bin/env python3
"""Install exactly one local wheel for the project version from pyproject.toml."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))


def _load_project_version(pyproject_path: Path) -> str:
    try:
        import tomllib as toml
    except ImportError:  # pragma: no cover
        try:
            import tomli as toml
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "Failed to parse "
                f"{pyproject_path}: tomllib is unavailable and tomli is not installed. "
                "Install tomli (for example: `pip install tomli`) or install dev extras."
            ) from exc

    pyproject_data = toml.loads(pyproject_path.read_text(encoding="utf-8"))

    project_table = pyproject_data.get("project")
    if not isinstance(project_table, dict):
        raise RuntimeError(
            f"Failed to parse {pyproject_path}: pyproject.toml missing 'project.version'."
        )

    project_version = project_table.get("version")
    if not isinstance(project_version, str) or not project_version.strip():
        raise RuntimeError(
            f"Failed to parse {pyproject_path}: pyproject.toml missing 'project.version'."
        )

    return project_version


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Install a single wheel matching the current pyproject version to avoid pip resolving "
            "multiple local versions at once."
        )
    )
    parser.add_argument("--dist-dir", default="dist", help="Wheel directory (default: dist)")
    parser.add_argument(
        "--prune-other-versions",
        action="store_true",
        help="Delete wheel files for other versions in dist before install.",
    )
    parser.add_argument(
        "--remove-version",
        action="append",
        default=[],
        help=(
            "Delete wheel(s) for a specific version before install. "
            "Use repeatedly, e.g. --remove-version 0.1.0"
        ),
    )
    parser.add_argument(
        "--pip-arg",
        action="append",
        default=[],
        help="Additional argument to pass through to pip install (repeatable).",
    )
    return parser.parse_args()


def _safe_delete_wheel(wheel_path: Path) -> None:
    try:
        wheel_path.unlink()
    except (OSError, PermissionError) as exc:
        print(
            f"Warning: failed to remove old wheel {wheel_path}: {exc}",
            file=sys.stderr,
        )


def main() -> int:
    args = _parse_args()
    repo_root = REPO_ROOT
    dist_dir = repo_root / args.dist_dir
    pyproject_path = repo_root / "pyproject.toml"

    project_version = _load_project_version(pyproject_path)
    package_prefix = "gepa_dapo_grn"

    if dist_dir.exists() and args.remove_version:
        for version in args.remove_version:
            for wheel_path in dist_dir.glob(f"{package_prefix}-{version}-*.whl"):
                _safe_delete_wheel(wheel_path)

    if args.prune_other_versions and dist_dir.exists():
        current_prefix = f"{package_prefix}-{project_version}-"
        for wheel_path in dist_dir.glob(f"{package_prefix}-*.whl"):
            if wheel_path.name.startswith(current_prefix):
                continue
            _safe_delete_wheel(wheel_path)

    from gepa_dapo_grn._packaging import find_single_wheel

    try:
        wheel_path = find_single_wheel(
            dist_dir=dist_dir,
            package_prefix=package_prefix,
            version=project_version,
        )
    except (FileNotFoundError, FileExistsError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    install_cmd = [sys.executable, "-m", "pip", "install", str(wheel_path), *args.pip_arg]
    print("Installing wheel:", wheel_path)
    print("Running:", " ".join(install_cmd))
    return subprocess.call(install_cmd)


if __name__ == "__main__":
    raise SystemExit(main())
