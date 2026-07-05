"""Optional graph-native public reasoning feedback utilities."""

from gepa_dapo_grn.graph.feedback import build_graph_feedback
from gepa_dapo_grn.graph.metrics import (
    claim_support_coverage,
    contradiction_count,
    critical_unsupported_claim_count,
    evidence_path_count,
    graph_completeness,
    graph_to_answer_alignment_score,
    unsupported_claim_count,
)
from gepa_dapo_grn.graph.schema import (
    ALLOWED_EDGE_TYPES,
    ALLOWED_NODE_TYPES,
    Claim,
    Contradiction,
    EvidenceLink,
    GraphEdge,
    GraphNode,
    ReasoningGraph,
)
from gepa_dapo_grn.graph.validators import (
    ValidationResult,
    validate_claim_support,
    validate_graph_for_promotion,
    validate_no_orphan_evidence,
    validate_no_unresolved_critical_contradictions,
)

__all__ = [
    "ALLOWED_EDGE_TYPES",
    "ALLOWED_NODE_TYPES",
    "Claim",
    "Contradiction",
    "EvidenceLink",
    "GraphEdge",
    "GraphNode",
    "ReasoningGraph",
    "ValidationResult",
    "build_graph_feedback",
    "claim_support_coverage",
    "contradiction_count",
    "critical_unsupported_claim_count",
    "evidence_path_count",
    "graph_completeness",
    "graph_to_answer_alignment_score",
    "unsupported_claim_count",
    "validate_claim_support",
    "validate_graph_for_promotion",
    "validate_no_orphan_evidence",
    "validate_no_unresolved_critical_contradictions",
]
