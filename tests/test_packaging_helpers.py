from pathlib import Path

import pytest

from gepa_dapo_grn._packaging import find_single_wheel


def test_find_single_wheel_returns_matching_version(tmp_path: Path) -> None:
    (tmp_path / "gepa_dapo_grn-0.1.0-py3-none-any.whl").write_text("old", encoding="utf-8")
    expected = tmp_path / "gepa_dapo_grn-0.2.1-py3-none-any.whl"
    expected.write_text("new", encoding="utf-8")

    actual = find_single_wheel(tmp_path, "gepa_dapo_grn", "0.2.1")
    assert actual == expected


def test_find_single_wheel_raises_for_missing_version(tmp_path: Path) -> None:
    (tmp_path / "gepa_dapo_grn-0.1.0-py3-none-any.whl").write_text("old", encoding="utf-8")

    with pytest.raises(FileNotFoundError):
        find_single_wheel(tmp_path, "gepa_dapo_grn", "0.2.1")
