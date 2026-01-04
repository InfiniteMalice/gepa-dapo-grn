# gepa-dapo-grn

`gepa-dapo-grn` is a standalone reinforcement learning engine for research workflows. It is
GEPA-shaped but GEPA-agnostic, providing DAPO optimization, curriculum tracking, safety
controls, and optional Global Response Normalization (GRN) without coupling to any external
GEPA repositories.

## What this library is

- **Standalone RL engine** with a stable public API under `gepa_dapo_grn.*`.
- **GEPA-shaped but GEPA-agnostic**: feedback is just structured reward/tag dictionaries.
- **Supports DAPO + curriculum + safety + GRN** with conservative defaults and GRN disabled
  unless explicitly enabled.

## Install

```bash
pip install gepa-dapo-grn
```

Optional extras:

- `gepa-dapo-grn[hf]` adds HuggingFace integration helpers.
- `gepa-dapo-grn[dev]` installs test and formatting tools.

## Minimal example (CPU-safe)

```python
from gepa_dapo_grn import (
    DAPOTrainer,
    DAPOConfig,
    RewardMixerConfig,
    GEPAFeedback,
)

# minimal fake feedback example
fb = GEPAFeedback(
    rewards={"truth": 1.0, "harm": -0.5},
    tags={},
    meta={"task_id": "demo"},
    abstained=False,
)
```

## Public API

Public API is defined by `__init__.py` exports. Anything not exported there is considered
internal and may change without notice.

## What this library intentionally does NOT do

- ship datasets
- provide prompt logic
- implement scoring logic
- encode opinions on ethics or safety policy

## Versioning policy

This project follows semantic versioning:

- `0.x.y` while interfaces are still evolving
- bump **minor** for interface changes
- bump **patch** for bugfixes only

See [CHANGELOG.md](CHANGELOG.md) for release notes.

## Build artifacts

To build source and wheel distributions:

```bash
python -m build
```

## PyPI publishing checklist

1. Install build and upload tooling:

   ```bash
   pip install build twine
   ```

2. Build artifacts:

   ```bash
   python -m build
   ```

3. Check distributions:

   ```bash
   twine check dist/*
   ```

4. Upload to TestPyPI first:

   ```bash
   twine upload --repository testpypi dist/*
   ```

5. Install from TestPyPI to validate:

   ```bash
   pip install -i https://test.pypi.org/simple/ gepa-dapo-grn
   ```

6. Upload to PyPI:

   ```bash
   twine upload dist/*
   ```

## License

MIT (see [LICENSE](LICENSE)).
