from gepa_dapo_grn.config import DAPOConfig, GRNConfig
from gepa_dapo_grn.gepa_interfaces import GEPAFeedback
from gepa_dapo_grn.safety_controller import SafetyController


def test_safety_controller_adjusts_configs() -> None:
    dapo_config = DAPOConfig(clip_ratio=0.2, kl_coeff=0.1, adaptive_kl=False)
    grn_config = GRNConfig(enabled=False)
    controller = SafetyController(
        decay=0.5,
        tag_risk_weights={"risk": 1.0},
        risk_tolerance=0.0,
        adjustment_scale=1.0,
        grn_enable_threshold=0.1,
    )

    feedback = GEPAFeedback(tags={"risk": 1.0})
    controller.update(feedback)
    controller.adjust_configs(dapo_config, grn_config)

    assert dapo_config.clip_ratio < 0.2
    assert dapo_config.kl_coeff > 0.1
    assert grn_config.enabled is True
