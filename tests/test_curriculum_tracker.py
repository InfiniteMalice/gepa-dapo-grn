import pytest

from gepa_dapo_grn.curriculum import CurriculumTracker, SimpleTextComposer
from gepa_dapo_grn.gepa_interfaces import GEPAFeedback


def test_curriculum_tracker_updates_and_weight() -> None:
    tracker = CurriculumTracker(
        decay=0.5,
        reward_weights={"truth": 1.0},
        tag_weights={"risk_score": -1.0},
        abstention_weight=0.2,
    )
    feedback = GEPAFeedback(
        rewards={"truth": 1.0},
        tags={"risk_score": 0.2},
        verifier={"verifier_pass": 1.0},
        meta={"task_id": "task-a"},
        abstained=False,
    )
    tracker.update("task-a", feedback)
    stats = tracker.describe_task("task-a")
    assert stats["reward_ema/truth"] > 0.0
    assert stats["tag_ema/risk_score"] > 0.0
    assert stats["verifier_pass_rate_ema"] >= 0.0
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


def test_curriculum_tracker_saturation_increases_composition_depth() -> None:
    tracker = CurriculumTracker(
        decay=0.5,
        saturation_pass_rate=0.8,
        min_samples_for_saturation=2,
        max_depth=3,
    )
    feedback = GEPAFeedback(rewards={"truth": 0.99}, verifier={"verifier_pass": 1.0})
    tracker.update("task-a", feedback)
    stats = tracker.update("task-a", feedback)
    assert stats.saturated is True
    assert stats.composition_depth >= 1


def test_simple_text_composer() -> None:
    composer = SimpleTextComposer(separator=" | ")
    output = composer.compose(["a", "b"], depth=2)
    assert output.startswith("[depth=2]")
    assert "a | b" in output


def test_curriculum_tracker_decay_bounds() -> None:
    for decay in (0.0, 1.0, -0.1, 1.1):
        with pytest.raises(ValueError):
            CurriculumTracker(decay=decay)
