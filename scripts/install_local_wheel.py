#!/usr/bin/env python3
"""Install exactly one local wheel for the project version from pyproject.toml."""

from __future__ import annotations

import argparse
import ast
import re
import shlex
import subprocess
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import Any, Callable

SCRIPT_REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = SCRIPT_REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

# Local wheel 0.1.0 is deprecated because mixed 0.1.0/0.2.x installs
# caused resolver conflicts in local/CI packaging flows (see changelog notes).
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


def _load_project_version(
    pyproject_path: Path,
    pyproject_data: dict[str, Any] | None = None,
) -> str:
    data = pyproject_data if pyproject_data is not None else _load_pyproject_data(pyproject_path)
    project_table = data.get("project")
    if isinstance(project_table, dict):
        project_version = project_table.get("version")
        if isinstance(project_version, str) and project_version.strip():
            return project_version
        dynamic_fields = project_table.get("dynamic")
        if isinstance(dynamic_fields, list) and "version" in dynamic_fields:
            tool_table = data.get("tool")
            if isinstance(tool_table, dict):
                setuptools_table = tool_table.get("setuptools")
                if isinstance(setuptools_table, dict):
                    dynamic_table = setuptools_table.get("dynamic")
                    if isinstance(dynamic_table, dict):
                        version_table = dynamic_table.get("version")
                        if isinstance(version_table, dict):
                            attr_path = version_table.get("attr")
                            if isinstance(attr_path, str) and attr_path.strip():
                                module_path, sep, attr_name = attr_path.rpartition(".")
                                if sep and module_path and attr_name:
                                    try:
                                        attr_value = _extract_string_attr_from_module_source(
                                            module_path=module_path,
                                            attr_name=attr_name,
                                        )
                                    except Exception as exc:
                                        raise RuntimeError(
                                            "Failed to resolve dynamic project version from "
                                            f"attr_path={attr_path!r} "
                                            f"(module_path={module_path!r}, "
                                            f"attr_name={attr_name!r}): {exc}"
                                        ) from exc
                                    if isinstance(attr_value, str) and attr_value.strip():
                                        return attr_value

    tool_table = data.get("tool")
    if isinstance(tool_table, dict):
        poetry_table = tool_table.get("poetry")
        if isinstance(poetry_table, dict):
            poetry_version = poetry_table.get("version")
            if isinstance(poetry_version, str) and poetry_version.strip():
                return poetry_version

    raise RuntimeError(
        "Failed to parse "
        f"{pyproject_path}: pyproject.toml missing or invalid project version metadata."
    )


def _load_project_name(
    pyproject_path: Path,
    pyproject_data: dict[str, Any] | None = None,
) -> str:
    data = pyproject_data if pyproject_data is not None else _load_pyproject_data(pyproject_path)

    project_table = data.get("project")
    if isinstance(project_table, dict):
        project_name = project_table.get("name")
        if isinstance(project_name, str) and project_name.strip():
            return project_name

    tool_table = data.get("tool")
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


def _extract_string_attr_from_module_source(module_path: str, attr_name: str) -> str:
    module_parts = module_path.split(".")
    if not module_parts or any(not part or not part.isidentifier() for part in module_parts):
        raise ValueError(f"Invalid module path for dynamic version resolution: {module_path!r}")
    relative_module_path = Path(*module_parts)
    candidates = [
        SRC_PATH / f"{relative_module_path}.py",
        SRC_PATH / relative_module_path / "__init__.py",
    ]
    if relative_module_path.name != "_version":
        candidates.append(SRC_PATH / relative_module_path / "_version.py")

    existing_candidates = [path for path in candidates if path.exists()]
    if not existing_candidates:
        raise FileNotFoundError(f"Could not locate module source for {module_path!r}")

    for module_file in existing_candidates:
        parsed = ast.parse(module_file.read_text(encoding="utf-8"), filename=str(module_file))
        for node in parsed.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == attr_name:
                        if isinstance(node.value, ast.Constant) and isinstance(
                            node.value.value, str
                        ):
                            return node.value.value
                        raise RuntimeError(
                            f"Attribute {attr_name!r} in {module_file} is not a string constant."
                        )
            if isinstance(node, ast.AnnAssign):
                if isinstance(node.target, ast.Name) and node.target.id == attr_name:
                    value = node.value
                    if isinstance(value, ast.Constant) and isinstance(value.value, str):
                        return value.value
                    raise RuntimeError(
                        f"Attribute {attr_name!r} in {module_file} is not a string constant."
                    )

    raise AttributeError(f"Attribute {attr_name!r} not found in any of {existing_candidates}")


def _wheel_prefix_for_project_name(project_name: str) -> str:
    """Normalize project name into a wheel filename prefix."""
    normalized = re.sub(r"[^0-9A-Za-z]+", "_", project_name.lower())
    normalized = re.sub(r"_+", "_", normalized)
    normalized = normalized.strip("_")
    if not normalized:
        raise RuntimeError(f"Invalid project name for wheel prefix: {project_name!r}")
    return normalized


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


def _load_find_single_wheel() -> Callable[..., Any]:
    # Keep this dynamic import to avoid importing package runtime dependencies
    # (e.g. torch via __init__) before wheel installation in smoke checks.
    packaging_path = SCRIPT_REPO_ROOT / "src" / "gepa_dapo_grn" / "_packaging.py"
    spec = spec_from_file_location("gepa_dapo_grn._packaging", packaging_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load packaging helpers from {packaging_path}")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.find_single_wheel


def _versions_to_remove(user_requested_versions: list[str]) -> list[str]:
    """Return ordered unique versions to remove from dist before installation."""
    versions = [*DEPRECATED_VERSIONS, *user_requested_versions]
    return list(dict.fromkeys(versions))


def main() -> int:
    args = _parse_args()
    dist_dir = SCRIPT_REPO_ROOT / args.dist_dir
    pyproject_path = SCRIPT_REPO_ROOT / "pyproject.toml"

    try:
        pyproject_data = _load_pyproject_data(pyproject_path)
        project_version = _load_project_version(pyproject_path, pyproject_data)
        project_name = _load_project_name(pyproject_path, pyproject_data)
        package_prefix = _wheel_prefix_for_project_name(project_name)
        find_single_wheel = _load_find_single_wheel()
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1

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

    try:
        wheel_path = find_single_wheel(
            dist_dir=dist_dir,
            package_prefix=package_prefix,
            version=project_version,
        )
    except (FileNotFoundError, FileExistsError, RuntimeError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    install_cmd = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--force-reinstall",
        str(wheel_path),
        *args.pip_arg,
    ]
    print("Installing wheel:", wheel_path)
    print("Running:", shlex.join(install_cmd))
    return subprocess.call(install_cmd)


if __name__ == "__main__":
    raise SystemExit(main())
