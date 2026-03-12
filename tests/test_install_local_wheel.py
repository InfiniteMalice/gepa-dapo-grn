from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


def _load_installer_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "install_local_wheel.py"
    spec = spec_from_file_location("install_local_wheel", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load installer module from {script_path}")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_versions_to_remove_includes_deprecated_default() -> None:
    module = _load_installer_module()
    assert module._versions_to_remove([]) == ["0.1.0"]


def test_versions_to_remove_deduplicates_while_preserving_order() -> None:
    module = _load_installer_module()
    assert module._versions_to_remove(["0.1.0", "0.2.0", "0.2.0"]) == ["0.1.0", "0.2.0"]
