"""Top-level backend-aware trainer factory."""

from __future__ import annotations

from typing import Optional, Union

import torch

from gepa_dapo_grn.config import (
    DAPOConfig,
    GRNConfig,
    MaxRLConfig,
    RewardMixerConfig,
    TrainerBackendConfig,
)
from gepa_dapo_grn.curriculum import CurriculumTracker
from gepa_dapo_grn.dapo_core import DAPOTrainer
from gepa_dapo_grn.maxrl_core import MaxRLTrainer
from gepa_dapo_grn.policy_interfaces import Policy
from gepa_dapo_grn.safety_controller import SafetyController

TrainerType = Union[DAPOTrainer, MaxRLTrainer]


def make_trainer(
    policy: Policy,
    optimizer: torch.optim.Optimizer,
    backend_config: Optional[TrainerBackendConfig] = None,
    dapo_config: Optional[DAPOConfig] = None,
    maxrl_config: Optional[MaxRLConfig] = None,
    grn_config: Optional[GRNConfig] = None,
    reward_mixer: Optional[RewardMixerConfig] = None,
    curriculum: Optional[CurriculumTracker] = None,
    safety_controller: Optional[SafetyController] = None,
) -> TrainerType:
    """Instantiate a trainer from a backend selector config."""

    selector = (backend_config or TrainerBackendConfig()).validated_backend()
    if selector == "maxrl":
        incompatible_inputs = []
        if reward_mixer is not None:
            incompatible_inputs.append("reward_mixer")
        if dapo_config is not None:
            incompatible_inputs.append("dapo_config")
        if incompatible_inputs:
            joined = ", ".join(incompatible_inputs)
            raise ValueError(f"Invalid config for backend='maxrl': incompatible inputs: {joined}")
        resolved = maxrl_config or MaxRLConfig(enabled=True)
        return MaxRLTrainer(
            policy=policy,
            optimizer=optimizer,
            config=resolved,
            grn_config=grn_config,
            curriculum=curriculum,
            safety_controller=safety_controller,
        )
    if maxrl_config is not None and maxrl_config.enabled:
        raise ValueError("Invalid config for backend='dapo': maxrl_config.enabled must be False")
    return DAPOTrainer(
        policy=policy,
        optimizer=optimizer,
        config=dapo_config,
        grn_config=grn_config,
        reward_mixer=reward_mixer,
        curriculum=curriculum,
        safety_controller=safety_controller,
    )
