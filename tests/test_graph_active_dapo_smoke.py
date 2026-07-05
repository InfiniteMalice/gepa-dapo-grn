from gepa_dapo_grn.active_grpo import TrainingMode
from gepa_dapo_grn.active_grpo.promotion import PromotionGateConfig
from gepa_dapo_grn.active_grpo.references import ActiveReference, ActiveReferenceStore
from gepa_dapo_grn.graph import Claim, EvidenceLink, GraphEdge, GraphNode, ReasoningGraph
from gepa_dapo_grn.trainers import GraphActiveDapoConfig, GraphActiveDapoTrainer


class FakeBaseTrainer:
    def __init__(self) -> None:
        self.decisions = []

    def record_feedback(self, decision) -> None:
        self.decisions.append(decision)


def _graph() -> ReasoningGraph:
    return ReasoningGraph(
        nodes=[
            GraphNode(id="e1", type="evidence", label="test"),
            GraphNode(id="c1", type="claim", label="Candidate is verified"),
        ],
        edges=[GraphEdge(id="s1", source="e1", target="c1", type="supports")],
        claims=[Claim(id="c1", text="Candidate is verified", critical=True)],
        evidence_links=[EvidenceLink(id="l1", claim_id="c1", evidence_node_id="e1")],
    )


def _verifier(_prompt, output, _reference):
    return {"score": 0.9 if "better" in output else 0.6}


def test_graph_active_dapo_trainer_smoke() -> None:
    base = FakeBaseTrainer()
    trainer = GraphActiveDapoTrainer(base_trainer=base, verifier_fn=_verifier)

    decision = trainer.decide(
        prompt_id="p1",
        prompt="prompt",
        reference_output="reference",
        reference_score=0.6,
        policy_candidates=["better candidate"],
        graph=_graph(),
        final_answer="The candidate is verified.",
    )

    assert decision.mode == TrainingMode.REINFORCE
    assert decision.promoted_reference is False
    assert decision.feedback.tags["has_graph"] == 1.0
    assert base.decisions == [decision]


def test_graph_active_dapo_does_not_require_graph_when_disabled() -> None:
    trainer = GraphActiveDapoTrainer(
        config=GraphActiveDapoConfig(use_graph_feedback=False),
        verifier_fn=_verifier,
    )

    decision = trainer.decide(
        prompt_id="p1",
        prompt="prompt",
        reference_output="reference",
        reference_score=0.6,
        policy_candidates=["better candidate"],
    )

    assert decision.feedback.tags["reinforcement_weight"] == 1.0
    assert "has_graph" not in decision.feedback.tags


def test_existing_reference_does_not_rescore_reference_output() -> None:
    calls = []

    def verifier(prompt, output, reference):
        calls.append((prompt, output, reference))
        return {"score": 0.9 if output == "candidate" else 0.1}

    store = ActiveReferenceStore(
        {"p1": ActiveReference(prompt_id="p1", reference_output="stored", reference_score=0.8)}
    )
    trainer = GraphActiveDapoTrainer(reference_store=store, verifier_fn=verifier)

    decision = trainer.decide(
        prompt_id="p1",
        prompt="prompt",
        reference_output="new reference should not be scored",
        policy_candidates=["candidate"],
    )

    assert decision.reference_score == 0.8
    assert calls == [("prompt", "candidate", "stored")]


def test_candidate_verifier_source_overrides_decide_fallback() -> None:
    trainer = GraphActiveDapoTrainer(
        config=GraphActiveDapoConfig(promotion_enabled=True),
        promotion_config=PromotionGateConfig(require_external_verifier=True),
    )

    decision = trainer.decide(
        prompt_id="p1",
        prompt="prompt",
        reference_output="reference",
        reference_score=0.5,
        policy_candidates=[{"output": "candidate", "score": 0.9, "verifier_source": "self"}],
        graph=_graph(),
        verifier_source="external",
    )

    assert decision.promoted_reference is False
    assert decision.diagnostics["verifier_source"] == "self"
    assert (
        "external verifier is required for promotion" in decision.diagnostics["promotion"]["errors"]
    )


def test_decide_verifier_source_is_fallback_for_candidate_score() -> None:
    trainer = GraphActiveDapoTrainer(
        config=GraphActiveDapoConfig(promotion_enabled=True),
        promotion_config=PromotionGateConfig(require_external_verifier=True),
    )

    decision = trainer.decide(
        prompt_id="p1",
        prompt="prompt",
        reference_output="reference",
        reference_score=0.5,
        policy_candidates=[{"output": "candidate", "score": 0.9}],
        graph=_graph(),
        verifier_source="unit-test-verifier",
    )

    assert decision.promoted_reference is True
    assert decision.diagnostics["verifier_source"] == "unit-test-verifier"


def test_verifier_fn_source_is_used_for_promotion() -> None:
    def verifier(_prompt, _output, _reference):
        return {"score": 0.9, "verifier_source": "unit-test-verifier"}

    trainer = GraphActiveDapoTrainer(
        config=GraphActiveDapoConfig(promotion_enabled=True),
        verifier_fn=verifier,
    )

    decision = trainer.decide(
        prompt_id="p1",
        prompt="prompt",
        reference_output="reference",
        reference_score=0.5,
        policy_candidates=["candidate"],
        graph=_graph(),
    )

    assert decision.promoted_reference is True
    assert decision.diagnostics["verifier_source"] == "unit-test-verifier"
