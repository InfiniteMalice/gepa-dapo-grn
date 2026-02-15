"""Release/packaging metadata consistency checks."""

from __future__ import annotations

from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib

from gepa_dapo_grn import __version__


def _load_pyproject() -> dict:
    pyproject_path = Path(__file__).resolve().parents[1] / "pyproject.toml"
    return tomllib.loads(pyproject_path.read_text(encoding="utf-8"))


def test_project_version_matches_runtime_version() -> None:
    data = _load_pyproject()
    assert data["project"]["version"] == __version__


def test_dependency_constraints_remain_release_safe() -> None:
    data = _load_pyproject()
    project = data["project"]

    assert project["requires-python"] == ">=3.9"
    assert project["dependencies"] == ["torch>=2.0,<3.0"]

    optional_dependencies = project["optional-dependencies"]
    assert optional_dependencies["hf"] == ["transformers>=4.40,<5.0"]
