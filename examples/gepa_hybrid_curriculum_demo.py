"""Demo of curriculum sampling and safety control using GEPA-style feedback."""

from __future__ import annotations

import random
from typing import Dict, List

from gepa_dapo_grn.config import DAPOConfig, GRNConfig
from gepa_dapo_grn.curriculum import CurriculumTracker
from gepa_dapo_grn.gepa_interfaces import GEPAFeedback
from gepa_dapo_grn.logging_utils import MetricsLogger
from gepa_dapo_grn.safety_controller import SafetyController


def make_feedback(task_id: str, rewards: Dict[str, float], tags: Dict[str, float]) -> GEPAFeedback:
    return GEPAFeedback(rewards=rewards, tags=tags, meta={"task_id": task_id}, abstained=False)


def generate_batch(task_id: str) -> List[GEPAFeedback]:
    feedbacks = []
    for _ in range(4):
        if task_id == "truthful":
            rewards = {"truth": random.uniform(0.7, 1.0)}
            tags = {"deception": random.uniform(0.0, 0.2)}
        else:
            rewards = {"truth": random.uniform(0.0, 0.4)}
            tags = {"deception": random.uniform(0.6, 1.0)}
        feedbacks.append(make_feedback(task_id, rewards, tags))
    return feedbacks


def main() -> None:
    random.seed(0)
    curriculum = CurriculumTracker(
        decay=0.8,
        reward_weights={"truth": 1.0},
        tag_weights={"deception": -1.0},
    )
    safety = SafetyController(tag_risk_weights={"deception": 1.0})
    dapo_config = DAPOConfig()
    grn_config = GRNConfig()
    logger = MetricsLogger(prefix="curriculum")

    for step in range(10):
        task_id = "truthful" if step % 2 == 0 else "deceptive"
        feedbacks = generate_batch(task_id)
        for feedback in feedbacks:
            curriculum.update(task_id, feedback)
            safety.update(feedback)

        safety.adjust_configs(dapo_config, grn_config)
        metrics = {
            "task_id": 1.0 if task_id == "truthful" else 0.0,
            "sample_weight": curriculum.sample_weight(task_id),
            "clip_ratio": dapo_config.clip_ratio,
            "kl_coeff": dapo_config.kl_coeff,
            **safety.describe(),
        }
        logger.log(metrics)


if __name__ == "__main__":
    main()
