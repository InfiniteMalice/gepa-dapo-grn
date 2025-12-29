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
