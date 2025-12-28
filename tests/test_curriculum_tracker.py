import torch

from gepa_dapo_grn.sampling import CurriculumTracker


def test_curriculum_tracker_updates_and_weights() -> None:
    tracker = CurriculumTracker(decay=0.5)
    reward_vectors = [
        {"easy": 1.0, "hard": 0.1},
        {"easy": 1.0, "hard": 0.0},
    ]
    stats = tracker.update(reward_vectors)
    assert "easy" in stats
    assert "hard" in stats

    weights = tracker.sampling_weights(["easy", "hard"])
    assert torch.isclose(weights.sum(), torch.tensor(1.0))
    assert weights[1] > weights[0]
