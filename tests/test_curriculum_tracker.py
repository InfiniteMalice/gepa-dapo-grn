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


def test_curriculum_prefers_tag_verifier_pass_over_legacy_verifier_field() -> None:
    tracker = CurriculumTracker(decay=0.5)
    feedback = GEPAFeedback(tags={"verifier_pass": 1.0}, verifier={"verifier_pass": 0.0})
    stats = tracker.update("task-a", feedback)
    assert stats.verifier_pass_rate_ema == 0.5


def test_curriculum_accepts_legacy_verifier_success_field() -> None:
    tracker = CurriculumTracker(decay=0.5)
    feedback = GEPAFeedback(verifier={"verifier_success": 1.0})
    stats = tracker.update("task-a", feedback)
    assert stats.verifier_pass_rate_ema == 0.5


def test_curriculum_accepts_verifier_success_in_tags() -> None:
    tracker = CurriculumTracker(decay=0.5)
    feedback = GEPAFeedback(tags={"verifier_success": 1.0})
    stats = tracker.update("task-a", feedback)
    assert stats.verifier_pass_rate_ema == 0.5


def test_curriculum_verifier_precedence_legacy_success_over_verifier_pass() -> None:
    tracker = CurriculumTracker(decay=0.5)
    feedback = GEPAFeedback(verifier={"verifier_pass": 0.0, "verifier_success": 1.0})
    stats = tracker.update("task-a", feedback)
    assert stats.verifier_pass_rate_ema == 0.5


def test_curriculum_coverage_prefers_tags_over_verifier_fields() -> None:
    tracker = CurriculumTracker(decay=0.5)
    feedback = GEPAFeedback(
        tags={"verifier_coverage": 0.9, "coverage": 0.8},
        verifier={"verifier_coverage": 0.4, "coverage": 0.2},
    )
    stats = tracker.update("task-a", feedback)
    assert stats.coverage_ema == 0.45


def test_curriculum_coverage_uses_verifier_when_tags_absent() -> None:
    tracker = CurriculumTracker(decay=0.5)
    feedback = GEPAFeedback(verifier={"verifier_coverage": 0.6, "coverage": 0.3})
    stats = tracker.update("task-a", feedback)
    assert stats.coverage_ema == 0.3


def test_curriculum_coverage_uses_legacy_fallback_and_default() -> None:
    tracker = CurriculumTracker(decay=0.5)
    legacy_feedback = GEPAFeedback(verifier={"coverage": 0.2})
    stats = tracker.update("task-a", legacy_feedback)
    assert stats.coverage_ema == 0.1

    default_feedback = GEPAFeedback()
    stats = tracker.update("task-a", default_feedback)
    assert stats.coverage_ema == 0.55


def test_curriculum_verifier_and_coverage_ignore_invalid_values() -> None:
    tracker = CurriculumTracker(decay=0.5)
    invalid_feedback = GEPAFeedback(
        tags={"verifier_success": "bad", "verifier_coverage": float("nan")}
    )
    stats = tracker.update("task-a", invalid_feedback)
    assert stats.verifier_pass_rate_ema == 0.0
    assert stats.coverage_ema == 0.5


def test_curriculum_continues_fallback_after_invalid_preferred_keys() -> None:
    tracker = CurriculumTracker(decay=0.5)
    feedback = GEPAFeedback(
        tags={"verifier_success": "bad", "verifier_coverage": "bad"},
        verifier={"verifier_pass": 1.0, "coverage": 0.4},
    )
    stats = tracker.update("task-a", feedback)
    assert stats.verifier_pass_rate_ema == 0.5
    assert stats.coverage_ema == 0.2


def test_curriculum_ignores_invalid_reward_and_difficulty_values() -> None:
    tracker = CurriculumTracker(decay=0.5)
    baseline_stats = tracker.update("task-a", GEPAFeedback())
    baseline_difficulty = baseline_stats.difficulty_ema
    stats = tracker.update(
        "task-a",
        GEPAFeedback(
            rewards={"reward": float("nan"), "difficulty": float("nan")},
            tags={"reward": float("nan"), "difficulty": float("nan")},
        ),
    )
    assert stats.difficulty_ema == baseline_difficulty


def test_simple_text_composer() -> None:
    composer = SimpleTextComposer(separator=" | ")
    output = composer.compose(["a", "b"], depth=2)
    assert output.startswith("[depth=2]")
    assert "a | b" in output


def test_curriculum_tracker_decay_bounds() -> None:
    for decay in (0.0, 1.0, -0.1, 1.1):
        with pytest.raises(ValueError):
            CurriculumTracker(decay=decay)
