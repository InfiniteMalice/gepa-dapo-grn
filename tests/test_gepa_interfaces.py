from gepa_dapo_grn.gepa_interfaces import GEPAFeedback


def test_gepa_feedback_defaults_and_dict() -> None:
    feedback = GEPAFeedback()
    assert feedback.rewards == {}
    assert feedback.tags == {}
    assert feedback.meta == {}
    assert feedback.abstained is False

    payload = feedback.to_dict()
    assert payload["rewards"] == {}
    assert payload["tags"] == {}
    assert payload["meta"] == {}
    assert payload["abstained"] is False


def test_gepa_feedback_with_data() -> None:
    feedback = GEPAFeedback(
        rewards={"accuracy": 0.9, "fluency": 0.8},
        tags={"calibration_error": 0.05},
        meta={"task_id": "task_123"},
        abstained=True,
    )
    payload = feedback.to_dict()
    assert payload["rewards"] == {"accuracy": 0.9, "fluency": 0.8}
    assert payload["tags"] == {"calibration_error": 0.05}
    assert payload["meta"] == {"task_id": "task_123"}
    assert payload["abstained"] is True
