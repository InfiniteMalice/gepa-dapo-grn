"""GEPA-aware DAPO training library with optional GRN support."""

from gepa_dapo_grn.config import DAPOConfig, GRNConfig, RewardMixerConfig
from gepa_dapo_grn.dapo_core import DAPOBatch, DAPOStepResult, DAPOTrainer
from gepa_dapo_grn.eval_hooks import EvalHook, EvalResult, run_eval
from gepa_dapo_grn.grn import GlobalResponseNorm, maybe_apply_grn, wrap_head_with_grn
from gepa_dapo_grn.logging_utils import MetricsLogger
from gepa_dapo_grn.policy_interfaces import HuggingFaceLMPolicy, Policy, PolicyOutput
from gepa_dapo_grn.reward_mixers import mix_reward_vectors
from gepa_dapo_grn.sampling import CurriculumTracker

__all__ = [
    "CurriculumTracker",
    "DAPOBatch",
    "DAPOConfig",
    "DAPOStepResult",
    "DAPOTrainer",
    "EvalHook",
    "EvalResult",
    "GRNConfig",
    "GlobalResponseNorm",
    "HuggingFaceLMPolicy",
    "MetricsLogger",
    "Policy",
    "PolicyOutput",
    "RewardMixerConfig",
    "maybe_apply_grn",
    "mix_reward_vectors",
    "run_eval",
    "wrap_head_with_grn",
]
