"""Release/packaging metadata consistency checks."""

from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib


def _load_pyproject() -> dict:
    resolved_path = Path(__file__).resolve()
    pyproject_path: Path | None = None
    for parent in resolved_path.parents:
        candidate = parent / "pyproject.toml"
        if candidate.exists():
            pyproject_path = candidate
            break

    if pyproject_path is None:
        raise RuntimeError(f"Unable to find pyproject.toml from {resolved_path}")

    return tomllib.loads(pyproject_path.read_text(encoding="utf-8"))


def _load_runtime_version() -> str:
    resolved_path = Path(__file__).resolve()
    version_path: Path | None = None
    for parent in resolved_path.parents:
        candidate = parent / "src" / "gepa_dapo_grn" / "_version.py"
        if candidate.exists():
            version_path = candidate
            break

    if version_path is None:
        raise RuntimeError(f"Unable to find _version.py from {resolved_path}")

    spec = spec_from_file_location("gepa_dapo_grn._version", version_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load version module from {version_path}")

    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.__version__


def test_project_version_matches_runtime_version() -> None:
    data = _load_pyproject()
    assert data["project"]["version"] == _load_runtime_version()


def test_dependency_constraints_remain_release_safe() -> None:
    # Intentional snapshot guard: update these hardcoded values whenever release
    # dependency constraints are intentionally changed in pyproject.toml.
    data = _load_pyproject()
    project = data["project"]

    assert project["requires-python"] == ">=3.9"
    assert project["dependencies"] == ["torch>=2.0,<3.0"]

    optional_dependencies = project["optional-dependencies"]
    assert optional_dependencies["hf"] == ["transformers>=4.40,<5.0"]
    assert optional_dependencies["dev"] == [
        "pytest",
        "ruff",
        "black",
        'tomli>=1.1.0; python_version < "3.11"',
    ]


def test_build_system_constraints_remain_twine_compatible() -> None:
    data = _load_pyproject()
    build_system = data["build-system"]

    assert build_system["build-backend"] == "setuptools.build_meta"
    assert build_system["requires"] == ["setuptools>=68,<77", "wheel"]
