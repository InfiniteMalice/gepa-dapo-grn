"""GEPA-aware DAPO training library with optional GRN support."""

from gepa_dapo_grn._version import __version__
from gepa_dapo_grn.active_grpo import (
    ActiveGRPOScheduler,
    ActiveReference,
    ActiveReferenceStore,
    PromotionCandidate,
    PromotionGateConfig,
    TrainingMode,
    evaluate_promotion,
)
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
from gepa_dapo_grn.graph import (
    Claim,
    Contradiction,
    EvidenceLink,
    GraphEdge,
    GraphNode,
    ReasoningGraph,
    build_graph_feedback,
)
from gepa_dapo_grn.grn import GlobalResponseNorm
from gepa_dapo_grn.maxrl_core import MaxRLBatch, MaxRLStepResult, MaxRLTrainer
from gepa_dapo_grn.safety_controller import SafetyController
from gepa_dapo_grn.trainer import make_trainer
from gepa_dapo_grn.trainers import GraphActiveDapoConfig, GraphActiveDapoTrainer, TrainingDecision
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
    "MaxRLBatch",
    "MaxRLStepResult",
    "MaxRLTrainer",
    "make_trainer",
    "CurriculumTracker",
    "TaskComposer",
    "SimpleTextComposer",
    "SafetyController",
    "GlobalResponseNorm",
    "ActiveGRPOScheduler",
    "ActiveReference",
    "ActiveReferenceStore",
    "TrainingMode",
    "PromotionCandidate",
    "PromotionGateConfig",
    "evaluate_promotion",
    "GraphNode",
    "GraphEdge",
    "Claim",
    "EvidenceLink",
    "Contradiction",
    "ReasoningGraph",
    "build_graph_feedback",
    "GraphActiveDapoConfig",
    "GraphActiveDapoTrainer",
    "TrainingDecision",
    "__version__",
]
