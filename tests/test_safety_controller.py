from gepa_dapo_grn.config import DAPOConfig, GRNConfig
from gepa_dapo_grn.gepa_interfaces import GEPAFeedback
from gepa_dapo_grn.safety_controller import SafetyController


def test_safety_controller_adjusts_configs_from_risk_tags() -> None:
    dapo_config = DAPOConfig(clip_ratio=0.2, kl_coeff=0.1, adaptive_kl=False)
    grn_config = GRNConfig(enabled=False)
    controller = SafetyController(
        decay=0.5,
        tag_risk_weights={"risk_score": 1.0},
        risk_tolerance=0.0,
        adjustment_scale=1.0,
        grn_enable_threshold=0.1,
    )

    feedback = GEPAFeedback(tags={"risk_score": 1.0})
    controller.update(feedback)
    controller.adjust_configs(dapo_config, grn_config)

    assert dapo_config.clip_ratio < 0.2
    assert dapo_config.kl_coeff > 0.1
    assert grn_config.enabled is True


def test_safety_controller_uses_configurable_verifier_signals() -> None:
    dapo_config = DAPOConfig(clip_ratio=0.2, kl_coeff=0.1, adaptive_kl=False)
    grn_config = GRNConfig(enabled=False)
    controller = SafetyController(
        verifier_risk_weights={"verifier_fail_rate": 2.0},
        adjustment_scale=1.0,
        risk_tolerance=0.2,
    )
    controller.update(GEPAFeedback(verifier={"verifier_fail_rate": 0.8}))
    controller.adjust_configs(dapo_config, grn_config)

    assert dapo_config.clip_ratio < 0.2
    assert dapo_config.kl_coeff > 0.1
    assert controller.sampling_multiplier() < 1.0


def test_adjust_grn_config_maxrl_path_threshold_and_baseline_behavior() -> None:
    controller = SafetyController(risk_tolerance=0.2, grn_enable_threshold=0.3)
    grn_config = GRNConfig(enabled=False)

    controller._risk_score = lambda: 0.4  # type: ignore[method-assign]
    controller.adjust_grn_config(grn_config)
    assert grn_config.enabled is False

    controller._risk_score = (  # type: ignore[method-assign]
        lambda: controller.grn_enable_threshold + controller.risk_tolerance
    )
    controller.adjust_grn_config(grn_config)
    assert grn_config.enabled is False

    controller._risk_score = lambda: 0.9  # type: ignore[method-assign]
    controller.adjust_grn_config(grn_config)
    assert grn_config.enabled is True

    sticky_controller = SafetyController(risk_tolerance=0.2, grn_enable_threshold=0.9)
    sticky_grn = GRNConfig(enabled=True)
    sticky_controller._risk_score = lambda: 0.2  # type: ignore[method-assign]
    sticky_controller.adjust_grn_config(sticky_grn)
    assert sticky_grn.enabled is True
