import pytest

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


def test_curriculum_tracker_decay_and_weight_override() -> None:
    tracker = CurriculumTracker(decay=0.5)
    feedback = GEPAFeedback(rewards={"truth": 1.0}, meta={"task_id": "task-a"})
    tracker.update("task-a", feedback)
    tracker.update("task-a", GEPAFeedback(rewards={"truth": 0.0}, meta={"task_id": "task-a"}))
    stats = tracker.describe_task("task-a")
    assert 0.0 < stats["reward_ema/truth"] < 1.0

    custom_tracker = CurriculumTracker(decay=0.5, weight_fn=lambda _: 2.5)
    custom_tracker.update("task-a", feedback)
    assert custom_tracker.sample_weight("task-a") == 2.5
    assert custom_tracker.sample_weight("unknown") == 1.0


def test_curriculum_tracker_decay_bounds() -> None:
    for decay in (0.0, 1.0, -0.1, 1.1):
        with pytest.raises(ValueError):
            CurriculumTracker(decay=decay)
