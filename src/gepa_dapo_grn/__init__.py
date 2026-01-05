"""GEPA-aware DAPO training library with optional GRN support."""

from gepa_dapo_grn.config import DAPOConfig, GRNConfig, RewardMixerConfig
from gepa_dapo_grn.curriculum import CurriculumTracker
from gepa_dapo_grn.dapo_core import DAPOTrainer
from gepa_dapo_grn.gepa_interfaces import GEPAFeedback
from gepa_dapo_grn.grn import GlobalResponseNorm
from gepa_dapo_grn.safety_controller import SafetyController

__version__ = "0.1.0"

__all__ = [
    "DAPOConfig",
    "GRNConfig",
    "RewardMixerConfig",
    "GEPAFeedback",
    "DAPOTrainer",
    "CurriculumTracker",
    "SafetyController",
    "GlobalResponseNorm",
    "__version__",
]
