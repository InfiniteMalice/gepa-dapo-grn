# gepa-dapo-grn

`gepa-dapo-grn` is a standalone reinforcement learning engine for research workflows. It is
GEPA-shaped but GEPA-agnostic, providing DAPO optimization, curriculum tracking, safety
controls, verifier-first hooks, and optional Global Response Normalization (GRN).

## What this library is

- **Standalone RL engine** with a stable public API under `gepa_dapo_grn.*`.
- **GEPA-shaped but GEPA-agnostic**: feedback is structured reward/tag/verifier dictionaries.
- **Supports DAPO + curriculum + safety + GRN** with conservative defaults and GRN disabled
  unless explicitly enabled.

## Practical guidance (v0.2.1)

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


## Building and installing a local wheel safely

If you build multiple versions locally, avoid `pip install dist/*.whl` because pip will try to
install all matching wheel files (which can include multiple versions of this same package and
fail with `ResolutionImpossible`).

Use this sequence instead:

```bash
rm -rf build dist *.egg-info
python -m pip install --upgrade build
python -m build
python scripts/install_local_wheel.py --prune-other-versions
```

This follows pip's suggested fix to remove conflicting versions before install:
by default, `scripts/install_local_wheel.py` only locates and installs the
current version wheel and does not delete files in `dist/`. Cleanup is opt-in:
`--remove-version <version>` deletes wheel files for the specified version(s),
and `--prune-other-versions` removes other non-current wheel files so pip
receives exactly one path.


## Publishing to PyPI safely

A common cause of `InvalidDistribution` during `twine upload dist/*` is stale or non-package
artifacts left in `dist/`. Use a clean build and upload only the current version artifacts:

```bash
rm -rf build dist *.egg-info
python -m pip install --upgrade build twine
python -m build
PROJECT_VERSION=$(python - <<'PY2'
from pathlib import Path
try:
    import tomllib as toml
except ImportError:
    import tomli as toml

data = toml.loads(Path('pyproject.toml').read_text(encoding='utf-8'))
print(data['project']['version'])
PY2
)
twine check dist/gepa_dapo_grn-${PROJECT_VERSION}.tar.gz dist/gepa_dapo_grn-${PROJECT_VERSION}-*.whl
twine upload dist/gepa_dapo_grn-${PROJECT_VERSION}.tar.gz dist/gepa_dapo_grn-${PROJECT_VERSION}-*.whl
```

This avoids uploading unrelated files and ensures both `Name` and `Version` metadata come from the
freshly built distributions only.

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
