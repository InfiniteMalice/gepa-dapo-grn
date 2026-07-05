"""Active-GRPO-style adaptive reference demo with deterministic scores."""

from gepa_dapo_grn.active_grpo import (
    ActiveGRPOScheduler,
    ActiveReference,
    ActiveReferenceStore,
    PromotionCandidate,
    PromotionGateConfig,
    evaluate_promotion,
)
from gepa_dapo_grn.graph import Claim, EvidenceLink, GraphEdge, GraphNode, ReasoningGraph

store = ActiveReferenceStore()
store.set(ActiveReference(prompt_id="p1", reference_output="baseline answer", reference_score=0.72))

store.update_candidate("p1", "verified improved answer", 0.78)
reference = store.get("p1")
assert reference is not None

scheduler = ActiveGRPOScheduler(margin=0.02, mixed_band=0.01)
mode = scheduler.choose_mode(reference.reference_score, reference.best_policy_score)

graph = ReasoningGraph(
    nodes=[
        GraphNode(id="e1", type="evidence", label="external verifier"),
        GraphNode(id="c1", type="claim", label="Improved answer passes verifier"),
    ],
    edges=[GraphEdge(id="s1", source="e1", target="c1", type="supports")],
    claims=[Claim(id="c1", text="Improved answer passes verifier", critical=True)],
    evidence_links=[EvidenceLink(id="l1", claim_id="c1", evidence_node_id="e1")],
)
candidate = PromotionCandidate(
    prompt_id="p1",
    output="verified improved answer",
    score=0.78,
    verifier_source="unit-test-verifier",
    graph=graph,
)
promotion = evaluate_promotion(candidate, reference, PromotionGateConfig())
if promotion.passed:
    store.promote("p1", candidate.output, candidate.score, metadata={"source": "demo"})

print({"mode": mode.value, "promotion": promotion.to_dict(), "reference": store.to_dict()["p1"]})
