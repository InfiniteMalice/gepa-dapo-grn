import pytest
import torch

from gepa_dapo_grn.config import RewardMixerConfig
from gepa_dapo_grn.reward_mixers import mix_reward_vectors, multi_objective_scalarize


def test_reward_mixer_weights_and_clipping() -> None:
    reward_vectors = [
        {"a": 1.0, "b": 0.0},
        {"a": 2.0, "b": 1.0},
    ]
    config = RewardMixerConfig(weights={"a": 1.0, "b": 2.0}, normalize=False, clip_max=2.5)
    scalar, stats = mix_reward_vectors(reward_vectors, config)
    expected = torch.tensor([1.0, 4.0])
    clipped = torch.clamp(expected, max=2.5)
    assert torch.allclose(scalar, clipped)
    assert stats["reward_weight/a"] == 1.0
    assert stats["reward_weight/b"] == 2.0


def test_reward_mixer_single_sample_normalization_is_finite() -> None:
    reward_vectors = [{"quality": 3.0}]
    config = RewardMixerConfig(normalize=True, default_weight=1.0)
    scalar, stats = mix_reward_vectors(reward_vectors, config)
    assert torch.isfinite(scalar).all()
    assert stats["reward_mean/quality"] == 3.0
    assert stats["reward_std/quality"] == pytest.approx(1e-6)
    assert stats["reward_scalar/std"] == 0.0


def test_reward_mixer_does_not_treat_deception_specially() -> None:
    reward_vectors = [{"helpfulness": 1.0, "deception": 10.0}]
    config = RewardMixerConfig(weights={"helpfulness": 1.0}, normalize=False)
    scalar, _ = mix_reward_vectors(reward_vectors, config)
    assert scalar.item() == 1.0


def test_multi_objective_scalarization_with_gates() -> None:
    reward_vectors = [{"a": 5.0, "b": -3.0}]
    scalar, stats = multi_objective_scalarize(
        reward_vectors,
        weights={"a": 1.0, "b": 1.0},
        normalize=False,
        gates={"a": (None, 2.0), "b": (-1.0, None)},
    )
    assert scalar.item() == pytest.approx(1.0)
    assert stats["reward_weight/a"] == 1.0
