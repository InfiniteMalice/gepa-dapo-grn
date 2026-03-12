from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace


def _load_installer_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "install_local_wheel.py"
    spec = spec_from_file_location("install_local_wheel", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load installer module from {script_path}")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fake_find_single_wheel(dist_dir: Path, package_prefix: str, version: str) -> Path:
    pattern = f"{package_prefix}-{version}-*.whl"
    matches = list(dist_dir.glob(pattern))
    if not matches:
        raise FileNotFoundError(
            f"No wheel found for package {package_prefix!r} version {version!r} in {dist_dir}."
        )
    if len(matches) > 1:
        raise FileExistsError(
            "Multiple wheels found for package "
            f"{package_prefix!r} version {version!r} in {dist_dir}."
        )
    return matches[0]


def _write_project_layout(tmp_path: Path, version: str = "0.2.1") -> None:
    (tmp_path / "pyproject.toml").write_text(
        "\n".join(
            [
                "[project]",
                'name = "gepa-dapo-grn"',
                f'version = "{version}"',
                "",
            ]
        ),
        encoding="utf-8",
    )
    (tmp_path / "dist").mkdir()


def test_versions_to_remove_includes_deprecated_default() -> None:
    module = _load_installer_module()
    assert module._versions_to_remove([]) == ["0.1.0"]


def test_main_happy_path_uses_project_name_prefix_and_invokes_pip(
    monkeypatch,
    tmp_path: Path,
) -> None:
    module = _load_installer_module()
    _write_project_layout(tmp_path)
    (tmp_path / "dist" / "gepa_dapo_grn-0.2.1-py3-none-any.whl").write_text("new", encoding="utf-8")
    (tmp_path / "dist" / "gepa_dapo_grn-0.1.0-py3-none-any.whl").write_text("old", encoding="utf-8")

    monkeypatch.setattr(module, "SCRIPT_REPO_ROOT", tmp_path)
    monkeypatch.setattr(
        module,
        "_parse_args",
        lambda: SimpleNamespace(
            dist_dir="dist", prune_other_versions=True, remove_version=[], pip_arg=[]
        ),
    )

    calls: list[list[str]] = []

    def _fake_call(cmd: list[str]) -> int:
        calls.append(cmd)
        return 0

    monkeypatch.setattr(module.subprocess, "call", _fake_call)
    monkeypatch.setattr(module, "_load_find_single_wheel", lambda: _fake_find_single_wheel)

    exit_code = module.main()

    assert exit_code == 0
    assert calls
    assert calls[0][0:4] == [module.sys.executable, "-m", "pip", "install"]
    assert "--force-reinstall" in calls[0]
    target_wheel = Path(calls[0][-1])
    assert target_wheel.name == "gepa_dapo_grn-0.2.1-py3-none-any.whl"
    assert target_wheel.parent.name == "dist"
    assert not (tmp_path / "dist" / "gepa_dapo_grn-0.1.0-py3-none-any.whl").exists()


def test_main_returns_subprocess_failure_code(monkeypatch, tmp_path: Path) -> None:
    module = _load_installer_module()
    _write_project_layout(tmp_path)
    (tmp_path / "dist" / "gepa_dapo_grn-0.2.1-py3-none-any.whl").write_text("new", encoding="utf-8")

    monkeypatch.setattr(module, "SCRIPT_REPO_ROOT", tmp_path)
    monkeypatch.setattr(
        module,
        "_parse_args",
        lambda: SimpleNamespace(
            dist_dir="dist", prune_other_versions=False, remove_version=[], pip_arg=[]
        ),
    )
    monkeypatch.setattr(module.subprocess, "call", lambda _cmd: 9)
    monkeypatch.setattr(module, "_load_find_single_wheel", lambda: _fake_find_single_wheel)

    assert module.main() == 9


def test_main_reports_missing_wheel_and_returns_1(monkeypatch, tmp_path: Path, capsys) -> None:
    module = _load_installer_module()
    _write_project_layout(tmp_path)

    monkeypatch.setattr(module, "SCRIPT_REPO_ROOT", tmp_path)
    monkeypatch.setattr(
        module,
        "_parse_args",
        lambda: SimpleNamespace(
            dist_dir="dist", prune_other_versions=False, remove_version=[], pip_arg=[]
        ),
    )
    monkeypatch.setattr(module, "_load_find_single_wheel", lambda: _fake_find_single_wheel)

    assert module.main() == 1
    captured = capsys.readouterr()
    assert "No wheel found for package 'gepa_dapo_grn' version '0.2.1'" in captured.err


def test_load_project_version_supports_poetry_fallback(tmp_path: Path) -> None:
    module = _load_installer_module()
    pyproject_path = tmp_path / "pyproject.toml"
    pyproject_path.write_text(
        "\n".join(
            [
                "[tool.poetry]",
                'name = "gepa-dapo-grn"',
                'version = "9.9.9"',
                "",
            ]
        ),
        encoding="utf-8",
    )

    assert module._load_project_version(pyproject_path) == "9.9.9"


def test_main_logs_shell_quoted_command(monkeypatch, tmp_path: Path, capsys) -> None:
    module = _load_installer_module()
    _write_project_layout(tmp_path)
    (tmp_path / "dist" / "gepa_dapo_grn-0.2.1-py3-none-any.whl").write_text("new", encoding="utf-8")

    monkeypatch.setattr(module, "SCRIPT_REPO_ROOT", tmp_path)
    monkeypatch.setattr(
        module,
        "_parse_args",
        lambda: SimpleNamespace(
            dist_dir="dist",
            prune_other_versions=False,
            remove_version=[],
            pip_arg=["--find-links", "path with spaces"],
        ),
    )
    monkeypatch.setattr(module.subprocess, "call", lambda _cmd: 0)
    monkeypatch.setattr(module, "_load_find_single_wheel", lambda: _fake_find_single_wheel)

    assert module.main() == 0
    captured = capsys.readouterr()
    assert "'path with spaces'" in captured.out
