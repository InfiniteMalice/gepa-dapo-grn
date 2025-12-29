"""GEPA-aware DAPO training library with optional GRN support."""

from gepa_dapo_grn.config import DAPOConfig, GRNConfig, RewardMixerConfig
from gepa_dapo_grn.curriculum import CurriculumTracker, TaskStats
from gepa_dapo_grn.dapo_core import DAPOBatch, DAPOStepResult, DAPOTrainer
from gepa_dapo_grn.eval_hooks import EvalHook, EvalResult, run_eval
from gepa_dapo_grn.gepa_interfaces import GEPAFeedback
from gepa_dapo_grn.grn import (
    GlobalResponseNorm,
    maybe_apply_grn,
    maybe_wrap_policy_heads,
    restore_policy_heads,
    wrap_head_with_grn,
)
from gepa_dapo_grn.integration.hf_lm import HuggingFaceLMPolicy
from gepa_dapo_grn.logging_utils import MetricsLogger
from gepa_dapo_grn.policy_interfaces import Policy, PolicyOutput
from gepa_dapo_grn.reward_mixers import mix_reward_vectors
from gepa_dapo_grn.safety_controller import SafetyController

__all__ = [
    "DAPOBatch",
    "DAPOConfig",
    "DAPOStepResult",
    "DAPOTrainer",
    "CurriculumTracker",
    "EvalHook",
    "EvalResult",
    "GEPAFeedback",
    "GlobalResponseNorm",
    "GRNConfig",
    "HuggingFaceLMPolicy",
    "maybe_apply_grn",
    "maybe_wrap_policy_heads",
    "MetricsLogger",
    "Policy",
    "PolicyOutput",
    "RewardMixerConfig",
    "SafetyController",
    "TaskStats",
    "mix_reward_vectors",
    "restore_policy_heads",
    "run_eval",
    "wrap_head_with_grn",
]
