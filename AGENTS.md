# Agent standards

These instructions apply to the entire repository.

## Expectations

- Use `$superpowers` before non-trivial work to select the right specialized workflow and
  surface repository constraints before editing.
- Use `$repo-quality-gate` for every non-trivial code change, refactor, architecture change,
  test change, agent workflow change, reward/scoring change, or repository-maintenance task.
- Do not skip the quality-gate sequence: task spec, design plan, implementation, tests,
  quality gate, and final summary.
- See `docs/REPO_STANDARDS.md` and `docs/BEADS.md` for standards and Beads guidance.
- If either skill is unavailable in the active agent environment, state that limitation before
  continuing and follow the closest documented fallback only when the maintainer approves or
  the task is documentation-only and low risk.
- Keep Python lines to 100 characters or fewer.
- Run `ruff check --fix .` and `black .` after making Python changes.
- Prefer explicit, well-typed functions and modules with clear responsibilities.
- Keep imports ordered: standard library, third-party, then local imports.
- Add or update tests for behavior changes when practical.
