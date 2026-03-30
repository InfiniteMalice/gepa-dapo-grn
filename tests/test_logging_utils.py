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
        backend="maxrl",
    )
    assert len(records) == 1
    record = records[0]
    assert record["backend"] == "maxrl"
    assert record["rewards"]["r"] == 1.0
    assert record["tags"]["verifier_success"] == 1.0
    assert record["verifier"]["verifier_coverage"] == 1.0
    assert record["meta"]["task_id"] == "a"
    assert record["abstained"] is False
