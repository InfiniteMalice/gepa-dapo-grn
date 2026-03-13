#!/usr/bin/env python3
"""Validate core metadata fields in built distribution artifacts."""

from __future__ import annotations

import argparse
import sys
import tarfile
import tempfile
import zipfile
from email.parser import Parser
from functools import lru_cache
from pathlib import Path

from packaging.version import InvalidVersion, Version

REQUIRED_FIELDS = ("Metadata-Version", "Name", "Version")


def _normalized_distribution_name(name: str) -> str:
    return "-".join(part for part in _name_split(name.lower()) if part)


def _name_split(name: str) -> list[str]:
    split_chars = "-_."
    parts: list[str] = []
    current = []
    for char in name:
        if char in split_chars:
            if current:
                parts.append("".join(current))
                current = []
            continue
        current.append(char)
    if current:
        parts.append("".join(current))
    return parts


def _expected_identity_from_artifact_path(artifact_path: Path) -> tuple[str, str]:
    if artifact_path.suffix == ".whl":
        wheel_stem = artifact_path.name.removesuffix(".whl")
        parts = wheel_stem.split("-")
        if len(parts) < 5:
            raise RuntimeError(f"{artifact_path}: invalid wheel filename")
        return parts[0], parts[1]

    if artifact_path.suffixes[-2:] == [".tar", ".gz"]:
        archive_stem = artifact_path.name.removesuffix(".tar.gz")
    elif artifact_path.suffix == ".zip":
        archive_stem = artifact_path.name.removesuffix(".zip")
    else:
        raise RuntimeError(f"{artifact_path}: unsupported artifact type")

    split_points = [index for index, char in enumerate(archive_stem) if char == "-"]
    if not split_points:
        raise RuntimeError(f"{artifact_path}: invalid source distribution filename")

    for split_index in reversed(split_points):
        expected_name = archive_stem[:split_index]
        expected_version = archive_stem[split_index + 1 :]
        if not expected_name or not expected_version:
            continue
        try:
            Version(expected_version)
        except InvalidVersion:
            continue
        return expected_name, expected_version

    raise RuntimeError(f"{artifact_path}: invalid source distribution filename")


def _expected_wheel_metadata_member(artifact_path: Path) -> str:
    wheel_stem = artifact_path.name.removesuffix(".whl")
    parts = wheel_stem.split("-")
    if len(parts) < 5:
        raise RuntimeError(f"{artifact_path}: invalid wheel filename")
    distribution = parts[0]
    version = parts[1]
    return f"{distribution}-{version}.dist-info/METADATA"


def _read_wheel_metadata(artifact_path: Path) -> str:
    expected_member = _expected_wheel_metadata_member(artifact_path)
    with zipfile.ZipFile(artifact_path) as wheel:
        all_members = set(wheel.namelist())
        if expected_member in all_members:
            metadata_bytes = wheel.read(expected_member)
            return metadata_bytes.decode("utf-8")

        metadata_candidates = [
            member_name
            for member_name in wheel.namelist()
            if member_name.endswith(".dist-info/METADATA")
        ]
        if len(metadata_candidates) != 1:
            raise RuntimeError(
                f"{artifact_path}: expected METADATA at {expected_member!r} "
                "or exactly one fallback .dist-info/METADATA file; "
                f"found {len(metadata_candidates)} fallback candidates"
            )
        metadata_bytes = wheel.read(metadata_candidates[0])
    return metadata_bytes.decode("utf-8")


def _read_sdist_metadata(artifact_path: Path) -> str:
    if artifact_path.suffixes[-2:] == [".tar", ".gz"]:
        with tarfile.open(artifact_path, "r:gz") as sdist:
            root_pkg_info_candidates = [
                member
                for member in sdist.getmembers()
                if member.name.endswith("/PKG-INFO") and member.name.count("/") == 1
            ]

            if len(root_pkg_info_candidates) != 1:
                raise RuntimeError(
                    f"{artifact_path}: expected exactly one root PKG-INFO file, "
                    f"found {len(root_pkg_info_candidates)}"
                )

            pkg_info_file = sdist.extractfile(root_pkg_info_candidates[0])
            if pkg_info_file is None:
                raise RuntimeError(f"{artifact_path}: PKG-INFO is not readable")
            metadata_bytes = pkg_info_file.read()

        return metadata_bytes.decode("utf-8")

    if artifact_path.suffix == ".zip":
        with zipfile.ZipFile(artifact_path) as sdist:
            root_pkg_info_candidates = [
                member_name
                for member_name in sdist.namelist()
                if member_name.endswith("/PKG-INFO") and member_name.count("/") == 1
            ]

            if len(root_pkg_info_candidates) != 1:
                raise RuntimeError(
                    f"{artifact_path}: expected exactly one root PKG-INFO file, "
                    f"found {len(root_pkg_info_candidates)}"
                )

            metadata_bytes = sdist.read(root_pkg_info_candidates[0])

        return metadata_bytes.decode("utf-8")

    raise RuntimeError(f"{artifact_path}: unsupported source distribution type")


def _load_metadata(artifact_path: Path) -> dict[str, str]:
    if artifact_path.suffix == ".whl":
        raw_metadata = _read_wheel_metadata(artifact_path)
    elif artifact_path.suffixes[-2:] == [".tar", ".gz"] or artifact_path.suffix == ".zip":
        raw_metadata = _read_sdist_metadata(artifact_path)
    else:
        raise RuntimeError(f"{artifact_path}: unsupported artifact type")

    parsed_metadata = Parser().parsestr(raw_metadata)
    return {field_name: parsed_metadata.get(field_name, "") for field_name in REQUIRED_FIELDS}


def _validate_artifact(artifact_path: Path) -> list[str]:
    metadata = _load_metadata(artifact_path)
    missing_fields = [field_name for field_name, value in metadata.items() if not value.strip()]
    if missing_fields:
        return [f"{artifact_path}: missing required metadata fields: {', '.join(missing_fields)}"]

    expected_name, expected_version = _expected_identity_from_artifact_path(artifact_path)
    errors: list[str] = []

    actual_name = metadata["Name"]
    if _normalized_distribution_name(expected_name) != _normalized_distribution_name(actual_name):
        errors.append(
            f"{artifact_path}: Name mismatch; expected {expected_name!r} from filename, "
            f"found metadata Name {actual_name!r}."
        )

    actual_version = metadata["Version"]
    if expected_version != actual_version:
        errors.append(
            f"{artifact_path}: Version mismatch; expected {expected_version!r} from filename, "
            f"found metadata Version {actual_version!r}."
        )

    metadata_version = metadata["Metadata-Version"]
    max_supported = _max_supported_pkginfo_metadata_version()
    if max_supported is not None:
        metadata_version_value = _metadata_version_tuple(metadata_version)
        max_supported_value = _metadata_version_tuple(max_supported)
        if metadata_version_value > max_supported_value:
            errors.append(
                f"{artifact_path}: Metadata-Version {metadata_version} exceeds locally supported "
                f"pkginfo/twine maximum {max_supported}. Upgrade twine/pkginfo before upload."
            )

    return errors


def _metadata_version_tuple(metadata_version: str) -> tuple[int, ...]:
    try:
        return tuple(int(part) for part in metadata_version.split("."))
    except ValueError as exc:
        raise RuntimeError(f"Invalid Metadata-Version value: {metadata_version!r}") from exc


@lru_cache(maxsize=1)
def _max_supported_pkginfo_metadata_version() -> str | None:
    """Return the max metadata version that local pkginfo can parse, if detectable.

    pkginfo does not expose a dedicated public constant for supported metadata versions.
    As a documented fallback, we probe support via its public ``pkginfo.Wheel`` API by
    parsing synthetic wheel files across known metadata versions.
    """
    try:
        import pkginfo
    except ImportError:
        return None

    if not hasattr(pkginfo, "Wheel"):
        return None

    candidate_versions = ["2.5", "2.4", "2.3", "2.2", "2.1", "2.0", "1.2", "1.1", "1.0"]
    supported_versions: list[str] = []

    for candidate_version in candidate_versions:
        if _pkginfo_supports_metadata_version(pkginfo, candidate_version):
            supported_versions.append(candidate_version)

    if not supported_versions:
        return None

    return max(supported_versions, key=_metadata_version_tuple)


def _pkginfo_supports_metadata_version(pkginfo_module: object, metadata_version: str) -> bool:
    wheel_name = "probe_pkg-0.0.0-py3-none-any.whl"
    metadata_member = "probe_pkg-0.0.0.dist-info/METADATA"
    metadata_text = "\n".join(
        [
            f"Metadata-Version: {metadata_version}",
            "Name: probe-pkg",
            "Version: 0.0.0",
            "",
        ]
    )

    with tempfile.TemporaryDirectory() as temp_dir:
        wheel_path = Path(temp_dir) / wheel_name
        with zipfile.ZipFile(wheel_path, "w") as wheel:
            wheel.writestr(metadata_member, metadata_text)

        try:
            parsed = pkginfo_module.Wheel(str(wheel_path))
        except Exception:
            return False

        name = getattr(parsed, "name", None)
        version = getattr(parsed, "version", None)

    return bool(name and version)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate Name, Version, and Metadata-Version in wheel and sdist artifacts "
            "before twine upload."
        )
    )
    parser.add_argument("artifacts", nargs="+", help="Paths to .whl, .tar.gz, and .zip files")
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
            print(error, file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
