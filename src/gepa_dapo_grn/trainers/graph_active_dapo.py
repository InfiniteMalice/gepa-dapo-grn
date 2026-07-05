"""Lightweight optional wrapper combining graph feedback and active references."""

from __future__ import annotations

from dataclasses import field
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Union

from gepa_dapo_grn._compat import dataclass
from gepa_dapo_grn.active_grpo.promotion import (
    PromotionCandidate,
    PromotionGateConfig,
    evaluate_promotion,
)
from gepa_dapo_grn.active_grpo.references import ActiveReference, ActiveReferenceStore
from gepa_dapo_grn.active_grpo.scheduler import ActiveGRPOScheduler, TrainingMode
from gepa_dapo_grn.gepa_interfaces import GEPAFeedback
from gepa_dapo_grn.graph.feedback import build_graph_feedback
from gepa_dapo_grn.graph.schema import ReasoningGraph

Candidate = Union[str, Mapping[str, Any]]
VerifierFn = Callable[[str, str, Optional[str]], Union[float, Mapping[str, Any]]]
GraphExtractorFn = Callable[[Any], Optional[ReasoningGraph]]


@dataclass(slots=True)
class GraphActiveDapoConfig:
    """Configuration for the optional Graph-Active-DAPO wrapper."""

    use_graph_feedback: bool = True
    use_active_references: bool = True
    promotion_enabled: bool = False
    graph_reward_weight: float = 0.2
    active_reference_margin: float = 0.02


@dataclass(slots=True)
class TrainingDecision:
    """Structured decision emitted by the optional wrapper."""

    prompt_id: str
    mode: TrainingMode
    imitation_weight: float
    reinforcement_weight: float
    selected_candidate: str
    candidate_score: float
    reference_score: float
    promoted_reference: bool
    feedback: GEPAFeedback
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prompt_id": self.prompt_id,
            "mode": self.mode.value,
            "imitation_weight": self.imitation_weight,
            "reinforcement_weight": self.reinforcement_weight,
            "selected_candidate": self.selected_candidate,
            "candidate_score": self.candidate_score,
            "reference_score": self.reference_score,
            "promoted_reference": self.promoted_reference,
            "feedback": self.feedback.to_dict(),
            "diagnostics": dict(self.diagnostics),
        }


class GraphActiveDapoTrainer:
    """Optional wrapper around an existing DAPO/GRN trainer or optimizer."""

    def __init__(
        self,
        base_trainer: Optional[Any] = None,
        reference_store: Optional[ActiveReferenceStore] = None,
        config: Optional[GraphActiveDapoConfig] = None,
        graph_extractor: Optional[GraphExtractorFn] = None,
        verifier_fn: Optional[VerifierFn] = None,
        promotion_config: Optional[PromotionGateConfig] = None,
    ) -> None:
        self.base_trainer = base_trainer
        self.reference_store = reference_store or ActiveReferenceStore()
        self.config = config or GraphActiveDapoConfig()
        self.graph_extractor = graph_extractor
        self.verifier_fn = verifier_fn
        self.promotion_config = promotion_config or PromotionGateConfig(
            margin=self.config.active_reference_margin
        )
        self.scheduler = ActiveGRPOScheduler(margin=self.config.active_reference_margin)

    def decide(
        self,
        prompt_id: str,
        prompt: str,
        reference_output: str,
        policy_candidates: Sequence[Candidate],
        reference_score: Optional[float] = None,
        graph: Optional[ReasoningGraph] = None,
        public_graph_artifact: Optional[Any] = None,
        final_answer: Optional[str] = None,
        verifier_source: str = "external",
        safety_violation: bool = False,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> TrainingDecision:
        """Build a decision and delegate feedback to the wrapped trainer.

        If ``reference_store`` already contains ``prompt_id``, the stored
        ``ActiveReference`` is reused. In that case ``reference_output`` and
        ``reference_score`` are ignored and the existing reference is not
        rescored or overwritten. Those inputs are only used when creating a new
        reference for an unseen prompt.
        """

        if not policy_candidates:
            raise ValueError("policy_candidates must contain at least one candidate")

        reference = self.reference_store.get(prompt_id)
        if reference is None:
            resolved_reference_score = (
                float(reference_score)
                if reference_score is not None
                else self._score(prompt, reference_output, None)["score"]
            )
            reference = ActiveReference(
                prompt_id=prompt_id,
                reference_output=reference_output,
                reference_score=resolved_reference_score,
                metadata=dict(metadata or {}),
            )
            self.reference_store.set(reference)

        scored_candidates = [
            self._normalize_candidate(
                prompt,
                candidate,
                reference.reference_output,
                verifier_source,
            )
            for candidate in policy_candidates
        ]
        selected = max(scored_candidates, key=lambda item: item["score"])
        selected_output = str(selected["output"])
        selected_score = float(selected["score"])
        selected_verifier_source = str(selected["verifier_source"])

        if self.config.use_active_references:
            self.reference_store.update_candidate(
                prompt_id,
                selected_output,
                selected_score,
                metadata=selected.get("metadata"),
            )

        mode = (
            self.scheduler.choose_mode(reference.reference_score, selected_score)
            if self.config.use_active_references
            else TrainingMode.REINFORCE
        )
        imitation_weight = (
            self.scheduler.imitation_weight(reference.reference_score, selected_score)
            if self.config.use_active_references
            else 0.0
        )
        reinforcement_weight = 1.0 - imitation_weight

        resolved_graph = graph
        if resolved_graph is None and self.graph_extractor is not None:
            resolved_graph = self.graph_extractor(public_graph_artifact)

        answer = final_answer if final_answer is not None else selected_output
        feedback = self._build_feedback(
            selected_score=selected_score,
            reference_score=reference.reference_score,
            mode=mode,
            imitation_weight=imitation_weight,
            reinforcement_weight=reinforcement_weight,
            graph=resolved_graph,
            final_answer=answer,
        )

        promoted = False
        promotion_result = None
        if self.config.promotion_enabled and self.config.use_active_references:
            candidate = PromotionCandidate(
                prompt_id=prompt_id,
                output=selected_output,
                score=selected_score,
                verifier_source=selected_verifier_source,
                graph=resolved_graph,
                safety_violation=safety_violation,
                metadata=dict(metadata or {}),
            )
            promotion_result = evaluate_promotion(candidate, reference, self.promotion_config)
            if promotion_result.passed:
                self.reference_store.promote(
                    prompt_id,
                    selected_output,
                    selected_score,
                    source="policy",
                    metadata={"promotion": promotion_result.to_dict()},
                )
                promoted = True

        decision = TrainingDecision(
            prompt_id=prompt_id,
            mode=mode,
            imitation_weight=imitation_weight,
            reinforcement_weight=reinforcement_weight,
            selected_candidate=selected_output,
            candidate_score=selected_score,
            reference_score=reference.reference_score,
            promoted_reference=promoted,
            feedback=feedback,
            diagnostics={
                "candidate_count": len(policy_candidates),
                "promotion": promotion_result.to_dict() if promotion_result else None,
                "verifier_source": selected_verifier_source,
                "metadata": dict(metadata or {}),
            },
        )
        self._delegate(decision)
        return decision

    def build_decision(self, *args: Any, **kwargs: Any) -> TrainingDecision:
        """Alias for callers that prefer a builder-style name."""

        return self.decide(*args, **kwargs)

    def _normalize_candidate(
        self,
        prompt: str,
        candidate: Candidate,
        reference_output: str,
        default_verifier_source: str,
    ) -> Dict[str, Any]:
        if isinstance(candidate, Mapping):
            output = str(candidate.get("output", ""))
            if "score" in candidate:
                score = float(candidate["score"])
                metadata = dict(candidate.get("metadata") or {})
                verifier_source = str(
                    candidate.get(
                        "verifier_source",
                        metadata.get("verifier_source", default_verifier_source),
                    )
                )
            else:
                scored = self._score(prompt, output, reference_output, default_verifier_source)
                score = scored["score"]
                metadata = scored["metadata"]
                verifier_source = scored["verifier_source"]
            return {
                "output": output,
                "score": score,
                "metadata": metadata,
                "verifier_source": verifier_source,
            }

        output = str(candidate)
        scored = self._score(prompt, output, reference_output, default_verifier_source)
        return {
            "output": output,
            "score": scored["score"],
            "metadata": scored["metadata"],
            "verifier_source": scored["verifier_source"],
        }

    def _score(
        self,
        prompt: str,
        output: str,
        reference_output: Optional[str],
        default_verifier_source: str = "external",
    ) -> Dict[str, Any]:
        if self.verifier_fn is None:
            return {
                "score": 0.0,
                "metadata": {},
                "verifier_source": default_verifier_source,
            }
        result = self.verifier_fn(prompt, output, reference_output)
        if isinstance(result, Mapping):
            metadata = dict(result.get("metadata") or {})
            return {
                "score": float(result.get("score", 0.0)),
                "metadata": metadata,
                "verifier_source": str(
                    result.get(
                        "verifier_source",
                        result.get(
                            "source",
                            metadata.get("verifier_source", default_verifier_source),
                        ),
                    )
                ),
            }
        return {
            "score": float(result),
            "metadata": {},
            "verifier_source": default_verifier_source,
        }

    def _build_feedback(
        self,
        selected_score: float,
        reference_score: float,
        mode: TrainingMode,
        imitation_weight: float,
        reinforcement_weight: float,
        graph: Optional[ReasoningGraph],
        final_answer: str,
    ) -> GEPAFeedback:
        base_rewards = {
            "candidate_score": selected_score,
            "reference_score": reference_score,
        }
        base_tags = {
            "imitation_weight": imitation_weight,
            "reinforcement_weight": reinforcement_weight,
        }

        if self.config.use_graph_feedback and graph is not None:
            feedback = build_graph_feedback(
                graph,
                final_answer,
                base_rewards=base_rewards,
                base_tags=base_tags,
            )
            for key in (
                "graph_completeness",
                "claim_support",
                "contradiction_penalty",
                "unsupported_claim_penalty",
                "graph_answer_alignment",
            ):
                feedback.rewards[key] *= self.config.graph_reward_weight
        else:
            feedback = GEPAFeedback(
                rewards=base_rewards,
                tags={key: float(value) for key, value in base_tags.items()},
                meta={},
            )

        feedback.meta["training_mode"] = mode.value
        feedback.meta["uses_graph_feedback"] = self.config.use_graph_feedback and graph is not None
        feedback.meta["uses_active_references"] = self.config.use_active_references
        return feedback

    def _delegate(self, decision: TrainingDecision) -> None:
        if self.base_trainer is None:
            return
        for method_name in ("record_feedback", "consume_feedback", "train_on_feedback"):
            method = getattr(self.base_trainer, method_name, None)
            if callable(method):
                method(decision)
                return
