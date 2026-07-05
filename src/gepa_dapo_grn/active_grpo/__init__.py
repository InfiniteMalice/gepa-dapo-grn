"""Optional Active-GRPO-style adaptive reference utilities."""

from gepa_dapo_grn.active_grpo.promotion import (
    PromotionCandidate,
    PromotionGateConfig,
    evaluate_promotion,
)
from gepa_dapo_grn.active_grpo.references import ActiveReference, ActiveReferenceStore
from gepa_dapo_grn.active_grpo.scheduler import ActiveGRPOScheduler, TrainingMode

__all__ = [
    "ActiveGRPOScheduler",
    "ActiveReference",
    "ActiveReferenceStore",
    "PromotionCandidate",
    "PromotionGateConfig",
    "TrainingMode",
    "evaluate_promotion",
]
