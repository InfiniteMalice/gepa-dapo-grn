"""GEPA-aware DAPO training library with optional GRN support."""

from gepa_dapo_grn._version import __version__
from gepa_dapo_grn.config import (
    DAPOConfig,
    GRNConfig,
    MaxRLConfig,
    RewardMixerConfig,
    TrainerBackendConfig,
)
from gepa_dapo_grn.curriculum import CurriculumTracker, SimpleTextComposer, TaskComposer
from gepa_dapo_grn.dapo_core import DAPOTrainer
from gepa_dapo_grn.gepa_interfaces import GEPAFeedback
from gepa_dapo_grn.grn import GlobalResponseNorm
from gepa_dapo_grn.maxrl_core import MaxRLTrainer
from gepa_dapo_grn.safety_controller import SafetyController
from gepa_dapo_grn.trainer import make_trainer
from gepa_dapo_grn.verifiers import Verifier, VerifierResult

__all__ = [
    "TrainerBackendConfig",
    "DAPOConfig",
    "MaxRLConfig",
    "GRNConfig",
    "RewardMixerConfig",
    "GEPAFeedback",
    "Verifier",
    "VerifierResult",
    "DAPOTrainer",
    "MaxRLTrainer",
    "make_trainer",
    "CurriculumTracker",
    "TaskComposer",
    "SimpleTextComposer",
    "SafetyController",
    "GlobalResponseNorm",
    "__version__",
]
