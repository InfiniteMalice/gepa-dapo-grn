#!/usr/bin/env python3
"""Validate core metadata fields in built distribution artifacts."""

from __future__ import annotations

import argparse
import tarfile
import zipfile
from email.parser import Parser
from pathlib import Path

REQUIRED_FIELDS = ("Metadata-Version", "Name", "Version")


def _read_wheel_metadata(artifact_path: Path) -> str:
    with zipfile.ZipFile(artifact_path) as wheel:
        metadata_candidates = [
            member_name
            for member_name in wheel.namelist()
            if member_name.endswith(".dist-info/METADATA")
        ]
        if len(metadata_candidates) != 1:
            raise RuntimeError(
                f"{artifact_path}: expected exactly one METADATA file, "
                f"found {len(metadata_candidates)}"
            )
        metadata_bytes = wheel.read(metadata_candidates[0])
    return metadata_bytes.decode("utf-8")


def _read_sdist_metadata(artifact_path: Path) -> str:
    with tarfile.open(artifact_path, "r:gz") as sdist:
        pkg_info_candidates = [
            member for member in sdist.getmembers() if member.name.endswith("/PKG-INFO")
        ]
        if len(pkg_info_candidates) != 1:
            raise RuntimeError(
                f"{artifact_path}: expected exactly one PKG-INFO file, "
                f"found {len(pkg_info_candidates)}"
            )
        pkg_info_file = sdist.extractfile(pkg_info_candidates[0])
        if pkg_info_file is None:
            raise RuntimeError(f"{artifact_path}: PKG-INFO is not readable")
        metadata_bytes = pkg_info_file.read()
    return metadata_bytes.decode("utf-8")


def _load_metadata(artifact_path: Path) -> dict[str, str]:
    if artifact_path.suffix == ".whl":
        raw_metadata = _read_wheel_metadata(artifact_path)
    elif artifact_path.suffixes[-2:] == [".tar", ".gz"]:
        raw_metadata = _read_sdist_metadata(artifact_path)
    else:
        raise RuntimeError(f"{artifact_path}: unsupported artifact type")

    parsed_metadata = Parser().parsestr(raw_metadata)
    return {field_name: parsed_metadata.get(field_name, "") for field_name in REQUIRED_FIELDS}


def _validate_artifact(artifact_path: Path) -> list[str]:
    metadata = _load_metadata(artifact_path)
    missing_fields = [field_name for field_name, value in metadata.items() if not value.strip()]
    if not missing_fields:
        return []
    return [f"{artifact_path}: missing required metadata fields: {', '.join(missing_fields)}"]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate Name/Version metadata in wheel and sdist artifacts before twine upload."
        )
    )
    parser.add_argument("artifacts", nargs="+", help="Paths to .whl and .tar.gz files")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    errors: list[str] = []

    for artifact in args.artifacts:
        artifact_path = Path(artifact)
        if not artifact_path.exists():
            errors.append(f"{artifact_path}: file does not exist")
            continue

        try:
            artifact_errors = _validate_artifact(artifact_path)
        except Exception as exc:  # pragma: no cover - defensive command-line guard
            errors.append(f"{artifact_path}: {exc}")
            continue

        if artifact_errors:
            errors.extend(artifact_errors)
        else:
            print(f"OK: {artifact_path}")

    if errors:
        for error in errors:
            print(error)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
