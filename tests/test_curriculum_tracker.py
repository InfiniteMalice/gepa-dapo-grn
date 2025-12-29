from gepa_dapo_grn.curriculum import CurriculumTracker
from gepa_dapo_grn.gepa_interfaces import GEPAFeedback


def test_curriculum_tracker_updates_and_weight() -> None:
    tracker = CurriculumTracker(
        decay=0.5,
        reward_weights={"truth": 1.0},
        tag_weights={"deception": -1.0},
        abstention_weight=0.2,
    )
    feedback = GEPAFeedback(
        rewards={"truth": 1.0},
        tags={"deception": 0.2},
        meta={"task_id": "task-a"},
        abstained=False,
    )
    tracker.update("task-a", feedback)
    stats = tracker.describe_task("task-a")
    assert stats["reward_ema/truth"] > 0.0
    assert stats["tag_ema/deception"] > 0.0
    weight = tracker.sample_weight("task-a")
    assert weight > 0.0
