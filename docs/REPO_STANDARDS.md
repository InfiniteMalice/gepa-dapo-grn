# Repository Standards

These standards complement `AGENTS.md`, which remains the canonical agent instruction file.

## Required Agent Skills

- Use `$superpowers` before non-trivial work to select the right workflow and surface repository
  constraints before editing.
- Use `$repo-quality-gate` for every non-trivial code change, refactor, architecture change, test
  change, agent workflow change, reward/scoring change, or repository-maintenance task.
- Preserve the quality-gate sequence: task spec, design plan, implementation, tests, quality gate,
  and final summary.

If either skill is unavailable, disclose the limitation and use a documented fallback only with
maintainer approval or for low-risk documentation-only work.

## Verification

For documentation-only changes, verify the instructions render clearly. For code changes, run the
formatters, tests, and checks listed in `AGENTS.md`.
