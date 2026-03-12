#!/usr/bin/env python3
"""Install exactly one local wheel for the project version from pyproject.toml."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

DEPRECATED_VERSIONS: tuple[str, ...] = ("0.1.0",)


def _load_toml_module() -> Any:
    try:
        import tomllib as toml
    except ImportError:  # pragma: no cover
        try:
            import tomli as toml
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "Failed to parse pyproject.toml: tomllib is unavailable and "
                "tomli is not installed. Install tomli (for example: "
                "`pip install tomli`) or install dev extras."
            ) from exc
    return toml


def _load_pyproject_data(pyproject_path: Path) -> dict[str, Any]:
    toml = _load_toml_module()
    data = toml.loads(pyproject_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise RuntimeError(f"Failed to parse {pyproject_path}: invalid TOML document root.")
    return data


def _load_project_version(pyproject_path: Path) -> str:
    pyproject_data = _load_pyproject_data(pyproject_path)
    project_table = pyproject_data.get("project")
    if isinstance(project_table, dict):
        project_version = project_table.get("version")
        if isinstance(project_version, str) and project_version.strip():
            return project_version

    tool_table = pyproject_data.get("tool")
    if isinstance(tool_table, dict):
        poetry_table = tool_table.get("poetry")
        if isinstance(poetry_table, dict):
            poetry_version = poetry_table.get("version")
            if isinstance(poetry_version, str) and poetry_version.strip():
                return poetry_version

    raise RuntimeError(
        "Failed to parse "
        f"{pyproject_path}: pyproject.toml missing or invalid 'project.version' value."
    )


def _load_project_name(pyproject_path: Path) -> str:
    pyproject_data = _load_pyproject_data(pyproject_path)

    project_table = pyproject_data.get("project")
    if isinstance(project_table, dict):
        project_name = project_table.get("name")
        if isinstance(project_name, str) and project_name.strip():
            return project_name

    tool_table = pyproject_data.get("tool")
    if isinstance(tool_table, dict):
        poetry_table = tool_table.get("poetry")
        if isinstance(poetry_table, dict):
            poetry_name = poetry_table.get("name")
            if isinstance(poetry_name, str) and poetry_name.strip():
                return poetry_name

    raise RuntimeError(
        "Failed to parse "
        f"{pyproject_path}: missing or invalid 'project.name' (or tool.poetry.name)."
    )


def _wheel_prefix_for_project_name(project_name: str) -> str:
    return project_name.replace("-", "_")


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
    except OSError as exc:
        print(
            f"Warning: failed to remove old wheel {wheel_path}: {exc}",
            file=sys.stderr,
        )


def _versions_to_remove(user_requested_versions: list[str]) -> list[str]:
    """Return ordered unique versions to remove from dist before installation."""
    versions = [*DEPRECATED_VERSIONS, *user_requested_versions]
    deduped_versions: list[str] = []
    for version in versions:
        if version not in deduped_versions:
            deduped_versions.append(version)
    return deduped_versions


def main() -> int:
    args = _parse_args()
    repo_root = REPO_ROOT
    dist_dir = repo_root / args.dist_dir
    pyproject_path = repo_root / "pyproject.toml"

    project_version = _load_project_version(pyproject_path)
    project_name = _load_project_name(pyproject_path)
    package_prefix = _wheel_prefix_for_project_name(project_name)

    versions_to_remove = _versions_to_remove(args.remove_version)
    if dist_dir.exists() and versions_to_remove:
        for version in versions_to_remove:
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
    except (FileNotFoundError, FileExistsError, RuntimeError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    install_cmd = [sys.executable, "-m", "pip", "install", str(wheel_path), *args.pip_arg]
    print("Installing wheel:", wheel_path)
    print("Running:", shlex.join(install_cmd))
    return subprocess.call(install_cmd)


if __name__ == "__main__":
    raise SystemExit(main())
