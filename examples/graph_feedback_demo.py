"""Small public graph feedback demo with no external APIs."""

from gepa_dapo_grn.graph import (
    Claim,
    EvidenceLink,
    GraphEdge,
    GraphNode,
    ReasoningGraph,
    build_graph_feedback,
)

graph = ReasoningGraph(
    nodes=[
        GraphNode(id="mechanism", type="mechanism", label="retrieval practice strengthens recall"),
        GraphNode(id="assumption", type="assumption", label="same study population"),
        GraphNode(id="constraint", type="constraint", label="short-term recall only"),
        GraphNode(id="evidence", type="evidence", label="public controlled study"),
        GraphNode(id="claim", type="claim", label="Retrieval practice improves short-term recall"),
    ],
    edges=[
        GraphEdge(id="support", source="evidence", target="claim", type="supports"),
        GraphEdge(id="bound", source="constraint", target="claim", type="bounds"),
    ],
    claims=[
        Claim(
            id="claim",
            text="Retrieval practice improves short-term recall",
            critical=True,
        )
    ],
    evidence_links=[
        EvidenceLink(
            id="evidence-link",
            claim_id="claim",
            evidence_node_id="evidence",
            description="Public experimental result supports the claim.",
            score=0.9,
        )
    ],
)

feedback = build_graph_feedback(
    graph,
    final_answer="Retrieval practice improves short-term recall under the stated constraints.",
)

print(feedback.to_dict())
