import pytest

from gepa_dapo_grn.active_grpo import ActiveReference, ActiveReferenceStore
from gepa_dapo_grn.active_grpo import references as references_module


def test_active_reference_store_set_get_update_and_promote(tmp_path) -> None:
    store = ActiveReferenceStore()
    store.set(ActiveReference(prompt_id="p1", reference_output="ref", reference_score=0.6))

    assert store.get("p1").reference_output == "ref"  # type: ignore[union-attr]

    store.update_candidate("p1", "weak", 0.5)
    assert store.get("p1").best_policy_output == "weak"  # type: ignore[union-attr]

    store.update_candidate("p1", "weaker", 0.4)
    assert store.get("p1").best_policy_output == "weak"  # type: ignore[union-attr]

    store.update_candidate("p1", "strong", 0.8)
    assert store.get("p1").best_policy_output == "strong"  # type: ignore[union-attr]

    store.promote("p1", "strong", 0.8, metadata={"verified": True})
    assert store.get("p1").reference_output == "strong"  # type: ignore[union-attr]
    assert store.get("p1").update_count == 1  # type: ignore[union-attr]

    path = tmp_path / "refs.json"
    store.save_json(str(path))
    loaded = ActiveReferenceStore.load_json(str(path))

    assert loaded.to_dict() == store.to_dict()


def test_active_reference_store_save_json_replaces_atomically(tmp_path, monkeypatch) -> None:
    store = ActiveReferenceStore()
    store.set(ActiveReference(prompt_id="p1", reference_output="ref", reference_score=0.6))
    path = tmp_path / "refs.json"
    replace_calls = []
    original_replace = references_module.os.replace

    def tracked_replace(source, destination):
        replace_calls.append((source, destination))
        assert path.exists() is False
        original_replace(source, destination)

    monkeypatch.setattr(references_module.os, "replace", tracked_replace)

    store.save_json(str(path))
    loaded = ActiveReferenceStore.load_json(str(path))

    assert len(replace_calls) == 1
    assert loaded.to_dict() == store.to_dict()


def test_active_reference_store_save_json_creates_parent_directories(tmp_path) -> None:
    store = ActiveReferenceStore()
    store.set(ActiveReference(prompt_id="p1", reference_output="ref", reference_score=0.6))
    path = tmp_path / "nested" / "refs.json"

    store.save_json(str(path))
    loaded = ActiveReferenceStore.load_json(str(path))

    assert loaded.to_dict() == store.to_dict()


def test_active_reference_store_rejects_mismatched_payload_key() -> None:
    payload = {
        "outer": {
            "prompt_id": "inner",
            "reference_output": "ref",
            "reference_score": 0.6,
        }
    }

    with pytest.raises(ValueError, match="does not match payload prompt_id"):
        ActiveReferenceStore.from_dict(payload)
