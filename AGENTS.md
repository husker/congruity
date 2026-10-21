# Working on Congruity

Read README.md for the project and docs/RUNBOOK.md for the research workflow.

## Conventions

- Use Python 3.13 and uv. Do not use pip directly.
- Run `uv run ruff check .` and `uv run pytest` after code changes.
- Keep timezone-aware datetimes and lazy initialization of external clients.
- Keep candidate generation separate from the benchmark, baseline and grader.
- Preserve the "prioritize, don't veto" selection policy: model reviews supply
  critique to test, not automatic vetoes. Do not introduce new model-verdict gates.
- Search only on the training split; reserve holdout confirmation for the end.
- Keep scientific claims tied to dated findings, datasets and actual measurements.
  An LLM novelty review is not independent evidence of novelty.
- Use `uv run --with <package>` for optional experiment dependencies.
- Keep raw prompts, stderr, downloaded documents, local configuration and credentials
  out of Git. Share reviewed fixtures or findings instead. See
  docs/DATA_AND_EXECUTION.md and docs/PUBLIC_SNAPSHOT.md.
- The public snapshot intentionally omits private run history. Never reconstruct
  missing historical evidence by inventing data.

## Layout

- `pipeline/`: compose orchestration, model I/O, grading and dashboard.
- `computable_packs/`: benchmark contracts, datasets, baselines and record sources.
- `domain_packs/`: compose-time hints and problem templates.
- `experiments/`: domain implementations and verification harnesses.
- `findings/`: dated historical research reports; INDEX.md indexes outcomes.
- `examples/demo/`: synthetic offline dashboard fixture.
- `tests/`: pipeline and numerical regression tests.

Generated candidate code runs locally and is not sandboxed by the benchmark
harness. Use an isolated environment for experiments.
