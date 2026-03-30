# Changelog

## [0.3.0] - 2026-03-30

- MaxRLTrainer: optional MaxRL-inspired verifier-first backend alongside the
  existing DAPO trainer.
- Introduced backend selection configuration (`TrainerBackendConfig`) and a
  `make_trainer(...)` factory for choosing `backend="dapo"` or `backend="maxrl"` without code edits.
- Included a verifier utility module and retained verifier-to-tag mapping support for GEPA
  feedback.
- Provided MaxRL logging and metrics for success count/rate, zero-success batches, sample count,
  verifier coverage, per-task saturation, and objective/KL values.
- Updated documentation and examples with backend guidance and a CPU-safe MaxRL verifier demo.
- Preserved the no-explicit-deception-penalty policy: deception-like signals remain tags,
  safety/controller inputs, and logging/analysis signals rather than scalar reward defaults.

## [0.2.1] - 2026-03-12

- Audited packaging metadata and synchronized the project version across `pyproject.toml`,
  runtime `__version__`, and release documentation to avoid release/upload mismatches.
- Added packaging metadata tests to catch dependency/version drift before publishing.
- Fixed wheel smoke-install workflow to install exactly one freshly built wheel, avoiding pip
  resolver conflicts when multiple package versions exist in `dist/`.
- Documented safe local build/install commands that select a single wheel version instead of
  using `dist/*.whl`.
- Added `scripts/install_local_wheel.py` to install only one wheel and optionally prune older
  local wheel versions before install.
- Added `--remove-version` support to `scripts/install_local_wheel.py` so local wheel versions
  like `0.1.0` can be explicitly removed before install.
- Documented deprecation of local wheel v0.1.0 in `scripts/install_local_wheel.py`;
  removal is opt-in via `--remove-version` and/or `--prune-other-versions`.
- Tightened wheel selection: ambiguous multi-match wheel patterns now raise an explicit error
  instead of silently picking one file.

## [0.2.0] - 2026-02-15

- Removed built-in explicit deception penalties; deception-like signals are treated as
  auxiliary tags/verifier/safety-controller signals by default.
- Added verifier-first interfaces (`Verifier`, `VerifierResult`, `GEPAFeedback.verifier`) and
  logging support for verifier pass-rate and coverage.
- Added curriculum freshness features: saturation detection, composition depth tracking,
  sampling updates, `TaskComposer` protocol, and `SimpleTextComposer` utility.
- Added optional soft ratio gating in DAPO (`use_soft_gating`, `gating_temperature`) as an
  alternative to hard clipping.
- Added GRN placement filters (`include_modules`, `exclude_modules`) and probe protection so
  interpretability modules are not wrapped unless explicitly selected.
- Updated examples for verifier hooks, composition-depth growth, and hard-vs-soft gating demos.

## 0.1.0

- Initial public release of `gepa-dapo-grn` with DAPO training, curriculum tracking,
  safety controls, and optional GRN support.
