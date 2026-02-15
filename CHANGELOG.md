# Changelog

## [0.2.1] - 2026-02-15

- Audited packaging metadata and synchronized the project version across `pyproject.toml`,
  runtime `__version__`, and release documentation to avoid release/upload mismatches.
- Added packaging metadata tests to catch dependency/version drift before publishing.

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
