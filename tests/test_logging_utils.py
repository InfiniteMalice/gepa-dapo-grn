from gepa_dapo_grn.gepa_interfaces import GEPAFeedback
from gepa_dapo_grn.logging_utils import feedback_records


def test_feedback_records_preserves_full_feedback_payload() -> None:
    records = feedback_records(
        [
            GEPAFeedback(
                rewards={"r": 1.0},
                tags={"verifier_success": 1.0},
                verifier={"verifier_coverage": 1.0},
                meta={"task_id": "a"},
                abstained=False,
            )
        ],
        backend=" MaxRL ",
    )
    assert len(records) == 1
    record = records[0]
    assert record["backend"] == "maxrl"
    assert record["rewards"]["r"] == 1.0
    assert record["tags"]["verifier_success"] == 1.0
    assert record["verifier"]["verifier_coverage"] == 1.0
    assert record["meta"]["task_id"] == "a"
    assert record["abstained"] is False


def test_feedback_records_empty_list_returns_empty() -> None:
    assert feedback_records([], backend="maxrl") == []


def test_feedback_records_multiple_and_abstained_preserved() -> None:
    feedbacks = [
        GEPAFeedback(rewards={"r": 1.0}, abstained=False),
        GEPAFeedback(rewards={"r": 0.0}, abstained=True),
    ]
    records = feedback_records(feedbacks, backend=" DAPO ")

    assert len(records) == 2
    assert records[0]["backend"] == "dapo"
    assert records[0]["abstained"] is False
    assert records[1]["backend"] == "dapo"
    assert records[1]["abstained"] is True
