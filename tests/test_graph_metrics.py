from gepa_dapo_grn.graph import (
    Claim,
    Contradiction,
    EvidenceLink,
    GraphEdge,
    GraphNode,
    ReasoningGraph,
    claim_support_coverage,
    contradiction_count,
    graph_completeness,
    graph_to_answer_alignment_score,
    unsupported_claim_count,
)


def test_support_coverage_and_unsupported_claims() -> None:
    graph = ReasoningGraph(
        nodes=[GraphNode(id="e1", type="evidence", label="paper")],
        claims=[
            Claim(id="c1", text="Supported claim"),
            Claim(id="c2", text="Unsupported claim"),
        ],
        evidence_links=[EvidenceLink(id="l1", claim_id="c1", evidence_node_id="e1")],
    )

    assert claim_support_coverage(graph) == 0.5
    assert unsupported_claim_count(graph) == 1


def test_contradiction_count_counts_objects_and_edges() -> None:
    graph = ReasoningGraph(
        nodes=[
            GraphNode(id="c1", type="claim", label="A"),
            GraphNode(id="c2", type="counterclaim", label="not A"),
        ],
        edges=[GraphEdge(id="e1", source="c2", target="c1", type="contradicts")],
        claims=[Claim(id="c1", text="A"), Claim(id="c2", text="not A")],
        contradictions=[Contradiction(id="k1", claim_id="c1", counterclaim_id="c2")],
    )

    assert contradiction_count(graph) == 2


def test_completeness_increases_with_reasoning_artifacts() -> None:
    small = ReasoningGraph(claims=[Claim(id="c1", text="A")])
    richer = ReasoningGraph(
        nodes=[
            GraphNode(id="m1", type="mechanism", label="mechanism"),
            GraphNode(id="a1", type="assumption", label="assumption"),
            GraphNode(id="k1", type="constraint", label="constraint"),
            GraphNode(id="e1", type="evidence", label="evidence"),
        ],
        claims=[Claim(id="c1", text="A")],
        evidence_links=[EvidenceLink(id="l1", claim_id="c1", evidence_node_id="e1")],
    )

    assert graph_completeness(richer) > graph_completeness(small)


def test_answer_alignment_default_and_judge_override() -> None:
    graph = ReasoningGraph(claims=[Claim(id="c1", text="Treatment improves recall")])

    assert graph_to_answer_alignment_score(graph, "The treatment improves recall on tests.") == 1.0
    assert graph_to_answer_alignment_score(graph, "", judge_fn=lambda _graph, _answer: 0.7) == 0.7
