"""Combine graph feedback and active references with a fake verifier."""

from gepa_dapo_grn.graph import Claim, EvidenceLink, GraphEdge, GraphNode, ReasoningGraph
from gepa_dapo_grn.trainers import GraphActiveDapoTrainer


def verifier(_prompt, output, _reference):
    return {"score": 0.85 if "verified" in output else 0.55}


graph = ReasoningGraph(
    nodes=[
        GraphNode(id="e1", type="evidence", label="deterministic verifier"),
        GraphNode(id="c1", type="claim", label="The answer is verified"),
    ],
    edges=[GraphEdge(id="s1", source="e1", target="c1", type="supports")],
    claims=[Claim(id="c1", text="The answer is verified", critical=True)],
    evidence_links=[EvidenceLink(id="l1", claim_id="c1", evidence_node_id="e1")],
)

trainer = GraphActiveDapoTrainer(verifier_fn=verifier)
decision = trainer.decide(
    prompt_id="demo-prompt",
    prompt="Return a verified answer.",
    reference_output="baseline answer",
    reference_score=0.6,
    policy_candidates=["unverified answer", "verified answer"],
    graph=graph,
    final_answer="The answer is verified by a deterministic public check.",
)

print(decision.to_dict())
