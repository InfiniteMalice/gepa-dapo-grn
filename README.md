# gepa-dapo-grn

`gepa-dapo-grn` is a standalone reinforcement learning engine for research workflows. It is
GEPA-shaped but GEPA-agnostic, providing DAPO optimization, curriculum tracking, safety
controls, verifier-first hooks, and optional Global Response Normalization (GRN).

## What this library is

- **Standalone RL engine** with a stable public API under `gepa_dapo_grn.*`.
- **GEPA-shaped but GEPA-agnostic**: feedback is structured reward/tag/verifier dictionaries.
- **Supports DAPO + curriculum + safety + GRN** with conservative defaults and GRN disabled
  unless explicitly enabled.

## Practical guidance (v0.2.0)

- **Verifier-first**: use `VerifierResult` and `GEPAFeedback.verifier` for pass/fail, scores,
  confidence, coverage, and diagnostics.
- **Composition curriculum**: `CurriculumTracker` tracks saturation and composition depth; use
  `TaskComposer` (or `SimpleTextComposer`) to generate harder tasks as easier ones saturate.
- **Soft gating option**: set `DAPOConfig(use_soft_gating=True)` to smoothly down-weight ratio
  outliers instead of hard clipping.
- **Deception handling policy**: do not apply built-in deception penalties. Treat deception-like
  signals as tags/risk/controller inputs (abstention, calibration, verifier constraints).
- **GRN placement guidance**: GRN is off by default. Prefer enabling only named policy/value
  modules via `include_modules`/`exclude_modules`, and keep probe/interpretability modules
  unwrapped unless explicitly included.

## Install

```bash
pip install gepa-dapo-grn
```

Optional extras:

- `gepa-dapo-grn[hf]` adds HuggingFace integration helpers.
- `gepa-dapo-grn[dev]` installs test and formatting tools.

## Minimal example (CPU-safe)

```python
from gepa_dapo_grn import DAPOTrainer, DAPOConfig, GEPAFeedback, RewardMixerConfig

fb = GEPAFeedback(
    rewards={"truth": 1.0, "helpfulness": 0.5},
    tags={"risk_score": 0.1},
    verifier={"verifier_pass": 1.0, "verifier_confidence": 0.9},
    meta={"task_id": "demo"},
    abstained=False,
)
```

## Public API

Public API is defined by `__init__.py` exports. Anything not exported there is considered
internal and may change without notice.

## Versioning policy

This project follows semantic versioning:

- `0.x.y` while interfaces are still evolving
- bump **minor** for interface changes
- bump **patch** for bugfixes only

See [CHANGELOG.md](CHANGELOG.md) for release notes.

## License

MIT (see [LICENSE](LICENSE)).
