"""GEPA-aware DAPO training library with optional GRN support."""

from importlib.metadata import PackageNotFoundError, version

from gepa_dapo_grn.config import DAPOConfig, GRNConfig, RewardMixerConfig
from gepa_dapo_grn.curriculum import CurriculumTracker
from gepa_dapo_grn.dapo_core import DAPOTrainer
from gepa_dapo_grn.gepa_interfaces import GEPAFeedback
from gepa_dapo_grn.grn import GlobalResponseNorm
from gepa_dapo_grn.safety_controller import SafetyController

try:
    __version__ = version("gepa-dapo-grn")
except PackageNotFoundError:
    __version__ = "0.0.0"

__all__ = [
    "__version__",
    "DAPOConfig",
    "DAPOTrainer",
    "CurriculumTracker",
    "GEPAFeedback",
    "GlobalResponseNorm",
    "GRNConfig",
    "RewardMixerConfig",
    "SafetyController",
]
