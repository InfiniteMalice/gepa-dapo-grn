from __future__ import annotations

import tarfile
import zipfile
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


def _load_validator_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "validate_dist_metadata.py"
    spec = spec_from_file_location("validate_dist_metadata", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load validator module from {script_path}")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_wheel_with_metadata(tmp_path: Path, metadata_text: str, wheel_name: str) -> Path:
    wheel_path = tmp_path / wheel_name
    with zipfile.ZipFile(wheel_path, "w") as wheel:
        wheel.writestr("pkg-0.0.0.dist-info/METADATA", metadata_text)
    return wheel_path


def _write_sdist_with_pkg_info(tmp_path: Path, pkg_info_text: str, sdist_name: str) -> Path:
    sdist_path = tmp_path / sdist_name
    pkg_root = sdist_name.removesuffix(".tar.gz")
    pkg_info_path = tmp_path / "PKG-INFO"
    pkg_info_path.write_text(pkg_info_text, encoding="utf-8")

    with tarfile.open(sdist_path, "w:gz") as sdist:
        sdist.add(pkg_info_path, arcname=f"{pkg_root}/PKG-INFO")

    return sdist_path


def test_validate_artifact_accepts_wheel_with_required_fields(tmp_path: Path) -> None:
    module = _load_validator_module()
    wheel = _write_wheel_with_metadata(
        tmp_path,
        "Metadata-Version: 2.1\nName: gepa-dapo-grn\nVersion: 0.2.1\n",
        "pkg-0.0.0-py3-none-any.whl",
    )

    assert module._validate_artifact(wheel) == []


def test_validate_artifact_rejects_missing_fields_in_wheel(tmp_path: Path) -> None:
    module = _load_validator_module()
    wheel = _write_wheel_with_metadata(
        tmp_path,
        "Metadata-Version: 2.1\n",
        "pkg-0.0.0-py3-none-any.whl",
    )

    errors = module._validate_artifact(wheel)

    assert len(errors) == 1
    assert "missing required metadata fields: Name, Version" in errors[0]


def test_validate_artifact_rejects_missing_fields_in_sdist(tmp_path: Path) -> None:
    module = _load_validator_module()
    sdist = _write_sdist_with_pkg_info(
        tmp_path,
        "Metadata-Version: 2.1\n",
        "broken-0.0.0.tar.gz",
    )

    errors = module._validate_artifact(sdist)

    assert len(errors) == 1
    assert "missing required metadata fields: Name, Version" in errors[0]


def test_validate_artifact_accepts_sdist_with_required_fields(tmp_path: Path) -> None:
    module = _load_validator_module()
    sdist = _write_sdist_with_pkg_info(
        tmp_path,
        "Metadata-Version: 2.1\nName: gepa-dapo-grn\nVersion: 0.2.1\n",
        "gepa_dapo_grn-0.2.1.tar.gz",
    )

    assert module._validate_artifact(sdist) == []


def test_read_sdist_metadata_uses_root_pkg_info_when_nested_exists(tmp_path: Path) -> None:
    module = _load_validator_module()
    sdist_name = "gepa_dapo_grn-0.2.1.tar.gz"
    sdist_path = tmp_path / sdist_name
    pkg_root = sdist_name.removesuffix(".tar.gz")

    root_pkg_info = tmp_path / "root-PKG-INFO"
    root_pkg_info.write_text(
        "Metadata-Version: 2.1\nName: gepa-dapo-grn\nVersion: 0.2.1\n",
        encoding="utf-8",
    )
    nested_pkg_info = tmp_path / "nested-PKG-INFO"
    nested_pkg_info.write_text("Metadata-Version: 2.1\n", encoding="utf-8")

    with tarfile.open(sdist_path, "w:gz") as sdist:
        sdist.add(root_pkg_info, arcname=f"{pkg_root}/PKG-INFO")
        sdist.add(nested_pkg_info, arcname=f"{pkg_root}/src/gepa_dapo_grn.egg-info/PKG-INFO")

    assert module._validate_artifact(sdist_path) == []


def test_read_wheel_metadata_prefers_expected_dist_info_path(tmp_path: Path) -> None:
    module = _load_validator_module()
    wheel_path = tmp_path / "gepa_dapo_grn-0.2.1-py3-none-any.whl"

    with zipfile.ZipFile(wheel_path, "w") as wheel:
        wheel.writestr(
            "other-1.0.0.dist-info/METADATA",
            "Metadata-Version: 2.1\n",
        )
        wheel.writestr(
            "gepa_dapo_grn-0.2.1.dist-info/METADATA",
            "Metadata-Version: 2.1\nName: gepa-dapo-grn\nVersion: 0.2.1\n",
        )

    assert module._validate_artifact(wheel_path) == []
