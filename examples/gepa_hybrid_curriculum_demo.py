"""Demo of verifier-first curriculum freshness with composition depth growth."""

from __future__ import annotations

import random
from typing import Dict, List

from gepa_dapo_grn.config import DAPOConfig, GRNConfig
from gepa_dapo_grn.curriculum import CurriculumTracker, SimpleTextComposer
from gepa_dapo_grn.gepa_interfaces import GEPAFeedback, VerifierResult
from gepa_dapo_grn.logging_utils import MetricsLogger, summarize_feedback
from gepa_dapo_grn.safety_controller import SafetyController


def boolean_verifier(signal: float) -> VerifierResult:
    passed = signal > 0.7
    return VerifierResult(
        passed=passed,
        score=signal,
        confidence=min(1.0, 0.5 + signal / 2.0),
        coverage=1.0,
        diagnostics={"verifier_fail_rate": float(not passed)},
    )


def make_feedback(
    task_id: str, rewards: Dict[str, float], signal: float, depth: int
) -> GEPAFeedback:
    verifier = boolean_verifier(signal).as_tags()
    tags = {"risk_score": max(0.0, 1.0 - signal), "calibration_error": 1.0 - signal}
    meta = {"task_id": task_id, "composition_depth": str(depth)}
    return GEPAFeedback(rewards=rewards, tags=tags, verifier=verifier, meta=meta, abstained=False)


def generate_batch(task_id: str, depth: int) -> List[GEPAFeedback]:
    feedbacks = []
    base = 0.85 if task_id == "easy" else 0.45
    for _ in range(4):
        low = base - 0.1
        high = base + 0.1
        sample = random.uniform(low, high)
        adjusted = sample - 0.05 * depth
        signal = max(0.0, min(1.0, adjusted))
        rewards = {"quality": signal}
        feedbacks.append(make_feedback(task_id, rewards, signal=signal, depth=depth))
    return feedbacks


def main() -> None:
    random.seed(0)
    curriculum = CurriculumTracker(
        decay=0.8, reward_weights={"quality": 1.0}, min_samples_for_saturation=4
    )
    safety = SafetyController(
        tag_risk_weights={"risk_score": 1.0, "calibration_error": 0.5},
        verifier_risk_weights={"verifier_fail_rate": 1.0},
    )
    dapo_config = DAPOConfig()
    grn_config = GRNConfig()
    composer = SimpleTextComposer()
    logger = MetricsLogger(prefix="curriculum")

    tasks = {"easy": ["classify A", "classify B"], "hard": ["reason A", "reason B"]}

    for step in range(12):
        task_id = curriculum.choose_task(list(tasks.keys()))
        depth = curriculum.current_depth(task_id)
        _composed_prompt = composer.compose(tasks[task_id], depth=depth)

        feedbacks = generate_batch(task_id, depth)
        for feedback in feedbacks:
            curriculum.update(task_id, feedback)
            safety.update(feedback)
        safety.adjust_configs(dapo_config, grn_config)

        task_stats = curriculum.describe_task(task_id)
        metrics = {
            **summarize_feedback(feedbacks),
            "sample_weight": curriculum.sample_weight(task_id),
            "clip_ratio": dapo_config.clip_ratio,
            "kl_coeff": dapo_config.kl_coeff,
            "composition_depth": task_stats.get("composition_depth", 0.0),
            "verifier_pass_rate_ema": task_stats.get("verifier_pass_rate_ema", 0.0),
            "coverage": task_stats.get("coverage", 1.0),
            **safety.describe(),
        }
        if step % 2 == 0:
            logger.log(metrics)


if __name__ == "__main__":
    main()
