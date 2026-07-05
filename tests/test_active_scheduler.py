import pytest

from gepa_dapo_grn.active_grpo import ActiveGRPOScheduler, TrainingMode


def test_active_scheduler_modes() -> None:
    scheduler = ActiveGRPOScheduler(margin=0.02, mixed_band=0.01)

    assert scheduler.choose_mode(0.8, None) == TrainingMode.IMITATE
    assert scheduler.choose_mode(0.8, 0.7) == TrainingMode.IMITATE
    assert scheduler.choose_mode(0.8, 0.805) == TrainingMode.MIXED
    assert scheduler.choose_mode(0.8, 0.83) == TrainingMode.REINFORCE


def test_active_scheduler_min_reference_score_threshold() -> None:
    scheduler = ActiveGRPOScheduler(margin=0.02, mixed_band=0.01, min_reference_score=0.75)

    assert scheduler.choose_mode(0.7, 0.7) == TrainingMode.REINFORCE
    assert scheduler.imitation_weight(0.7, 0.7) == 0.0
    assert scheduler.reinforcement_weight(0.7, 0.7) == 1.0


def test_active_scheduler_rejects_invalid_band_configuration() -> None:
    with pytest.raises(ValueError, match="mixed_band"):
        ActiveGRPOScheduler(margin=0.01, mixed_band=0.02)


def test_active_scheduler_weights_are_bounded_and_sum_to_one() -> None:
    scheduler = ActiveGRPOScheduler(margin=0.02, mixed_band=0.01)

    for policy_score in (None, 0.7, 0.805, 0.83):
        imitation = scheduler.imitation_weight(0.8, policy_score)
        reinforcement = scheduler.reinforcement_weight(0.8, policy_score)
        assert 0.0 <= imitation <= 1.0
        assert 0.0 <= reinforcement <= 1.0
        assert imitation + reinforcement == 1.0
