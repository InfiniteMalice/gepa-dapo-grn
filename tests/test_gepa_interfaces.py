from gepa_dapo_grn.gepa_interfaces import GEPAFeedback, VerifierResult


def test_gepa_feedback_defaults_and_dict() -> None:
    feedback = GEPAFeedback()
    assert feedback.rewards == {}
    assert feedback.tags == {}
    assert feedback.verifier == {}
    assert feedback.meta == {}
    assert feedback.abstained is False

    payload = feedback.to_dict()
    assert payload["rewards"] == {}
    assert payload["tags"] == {}
    assert payload["verifier"] == {}
    assert payload["meta"] == {}
    assert payload["abstained"] is False


def test_gepa_feedback_with_data() -> None:
    feedback = GEPAFeedback(
        rewards={"accuracy": 0.9, "fluency": 0.8},
        tags={"calibration_error": 0.05},
        verifier={"verifier_pass": 1.0, "verifier_coverage": 0.8},
        meta={"task_id": "task_123"},
        abstained=True,
    )
    payload = feedback.to_dict()
    assert payload["rewards"] == {"accuracy": 0.9, "fluency": 0.8}
    assert payload["tags"] == {"calibration_error": 0.05}
    assert payload["verifier"] == {"verifier_pass": 1.0, "verifier_coverage": 0.8}
    assert payload["meta"] == {"task_id": "task_123"}
    assert payload["abstained"] is True


def test_verifier_result_as_tags() -> None:
    result = VerifierResult(
        passed=True,
        score=0.9,
        confidence=0.8,
        coverage=0.7,
        diagnostics={"calibration_error": 0.1},
    )
    tags = result.as_tags()
    assert tags["verifier_pass"] == 1.0
    assert tags["verifier_score"] == 0.9
    assert tags["verifier_confidence"] == 0.8
    assert tags["verifier_coverage"] == 0.7
    assert tags["calibration_error"] == 0.1
