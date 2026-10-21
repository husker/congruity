# Spec: Computable Problem Packs

**Date:** 2026-05-31
**Status:** approved design, awaiting implementation plan
**Source:** User clarified that Congruity should not depend on the user's domain
expertise. The project should first target problems whose results can be
verified computationally by AI.

## Goal

Add a first-class "computable problem pack" layer to Congruity. A pack defines a
problem family where the system can verify candidate transfers by running code:
there is an executable evaluator, a validator, baseline commands or adapters,
datasets or synthetic generators, and a machine-readable bar for success.

This shifts Congruity from "compose ideas, then improvise a benchmark" to:

1. Choose a problem family with an executable judge.
2. Decompose the problem into a reusable shape.
3. Search cross-disciplinary papers for matching shapes.
4. Generate candidate compositions.
5. Verify them against predeclared baselines and metrics.

The first implementation should build the framework plus three seed packs:
`short_string_compression`, `small_points_optimization`, and
`cache_replacement`.

## Non-Expert Constraint

The user should not need to provide field expertise. The pack must carry enough
machine-readable guidance for an agent to decide whether a candidate is testable,
what baselines it must beat, which metric to use, and how to run a smoke check.

The system can still report uncertainty, but that uncertainty must be about
computable artifacts, for example missing dataset, missing executable baseline,
or record-only SOTA. It must not ask the user to adjudicate technical merit.

## Core Concepts

### Problem Shape

Each pack declares a shape:

- `objective`: what is being optimized or predicted.
- `input_representation`: what inputs look like.
- `output_representation`: what outputs must look like.
- `constraints`: validity conditions outputs must satisfy.
- `failure_modes`: common reasons transferred methods fail.
- `verification_kind`: one of `exact_oracle`, `roundtrip`, `benchmark`,
  `backtest`, `record_comparison`, or `property_test`.

The shape is the bridge between fields. Compose should eventually search for
papers that solve problems with the same shape even when their vocabulary differs.

### Bar Levels

Each pack exposes tiered success bars:

- `naive`: simple baselines that catch non-working methods.
- `strong_reproducible`: strong local baselines that can be run by the agent.
- `record_cached`: public record tables, leaderboard scores, or cached reference
  values that may be compared without reimplementing the full SOTA.

The first useful win is usually against `strong_reproducible`. A `record_cached`
win is stronger, but packs may declare it unavailable or comparison-only.

### AI-Verifiable

A pack is AI-verifiable when it has:

- At least one runnable evaluator.
- At least one validity validator.
- At least one runnable baseline at `strong_reproducible` or a clear reason why
  only `naive` is available.
- A smoke command that finishes locally.
- A structured preflight verdict explaining missing pieces.

## File Layout

Add a new top-level directory:

```text
computable_packs/
  short_string_compression/
    pack.json
    evaluator.json
    baselines.json
    datasets.json
    sota.json
    gotchas.md
  small_points_optimization/
    pack.json
    evaluator.json
    baselines.json
    datasets.json
    sota.json
    gotchas.md
  cache_replacement/
    pack.json
    evaluator.json
    baselines.json
    datasets.json
    sota.json
    gotchas.md
```

This is separate from `domain_packs/`. Domain packs are broad topical helpers.
Computable packs are executable verification contracts.

## Data Model

Add `pipeline/computable.py` with dataclasses or Pydantic models for:

- `ComputablePack`
- `ProblemShape`
- `EvaluatorSpec`
- `BaselineSpec`
- `DatasetSpec`
- `SotaSpec`
- `PreflightResult`

The loader should read a pack directory, validate every JSON file, and expose one
merged Python object.

### Required Fields

`pack.json`:

- `id`
- `display_name`
- `description`
- `problem_shape`
- `recommended_problem_template`
- `compose_query_hints`

`evaluator.json`:

- `metric_primary`
- `higher_is_better`
- `validator`
- `smoke_command`
- `full_command`
- `pass_criteria`

`baselines.json`:

- `baselines`: list of `{id, name, level, command, notes}`

`datasets.json`:

- `datasets`: list of `{id, name, kind, availability, command, notes}`

`sota.json`:

- `bar_levels`: list of `{level, available, comparator, source, notes}`

## CLI

Add a command surface:

```bash
uv run python -m pipeline.computable list
uv run python -m pipeline.computable inspect short_string_compression
uv run python -m pipeline.computable match --problem "..."
uv run python -m pipeline.computable preflight short_string_compression
```

### `list`

Print available pack IDs and display names.

### `inspect`

Print the merged pack as JSON so an agent or user can see the problem shape,
baselines, datasets, and bar levels.

### `match`

Given a free-form problem statement, return candidate packs ranked by simple
deterministic text overlap against pack keywords, query hints, and templates.
This can be improved later with LLM matching, but the first version should be
offline and testable.

### `preflight`

Return JSON:

```json
{
  "pack_id": "short_string_compression",
  "computable": true,
  "ai_verifiable": true,
  "missing": [],
  "bar_levels_available": ["naive", "strong_reproducible", "record_cached"],
  "recommended_compose_constraints": ["..."],
  "smoke_command": "..."
}
```

Preflight should be purely structural in the first implementation. It checks that
declared commands and required fields exist; it does not have to execute the
commands until a later phase.

## Seed Pack Details

### `short_string_compression`

Shape:

- Objective: reduce encoded byte length for short records.
- Validator: lossless round-trip for every sample.
- Metric: compression ratio, lower is better.
- Strong baselines: Brotli with dictionary, Zstandard with dictionary, HPACK for
  HTTP-like headers.
- Datasets: real HTTP headers from existing `experiments/c1_short_string/`,
  JSON snippets, synthetic templated records.
- Useful cross-field analogies: biosequence tokenization, finite-state language
  models, adaptive coding, small-alphabet transforms.

Smoke command should reuse the existing short-string bench with transient deps.

### `small_points_optimization`

Shape:

- Objective: maximize a geometric objective over point configurations.
- Validator: bounds, no invalid coordinates, objective recomputation from output.
- Metric: objective value, higher is better.
- Strong baselines: simulated annealing, basin hopping, CMA-ES, differential
  evolution, full active repair where available.
- Datasets: deterministic `n` values and seed grids; published reference values
  when available.
- Useful cross-field analogies: physics energy minimization, molecular packing,
  mesh relaxation, active-set repair.

Smoke command should target a small `n` so it finishes quickly.

### `cache_replacement`

Shape:

- Objective: maximize cache hit rate or minimize miss ratio under capacity.
- Validator: capacity never exceeded, deterministic trace replay, no future
  leakage unless the baseline is explicitly oracle.
- Metric: hit rate, higher is better.
- Strong baselines: LRU, ARC, SIEVE, S3FIFO, FIFO, LFU.
- Datasets: synthetic Zipf, loop, scan, scan-with-hot, plus optional public
  traces if available.
- Useful cross-field analogies: online scheduling, bandits, prefetching,
  time-to-next-use prediction, streaming sketches.

Smoke command should reuse the existing cache replacement bench.

## Integration Boundaries

First implementation:

- Adds pack files, models, loaders, CLI, and tests.
- Does not change `pipeline.compose` behavior.
- Does not run new compose jobs.
- Does not create new experiment results.

Follow-up implementation:

- Add `--computable-pack <id>` to compose.
- Inject pack shape, baselines, evaluator, and gotchas into dimension,
  composition, and recipe prompts.
- Make recipe generation prefer pack baselines and bar levels.
- Allow autoverify to call pack-declared baseline adapters instead of asking
  codegen to invent baselines.

## Error Handling

Loader errors should identify the pack ID, file, and missing field. CLI commands
should return non-zero for invalid packs and print a concise error to stderr.

`preflight` should be tolerant of incomplete packs. It should return
`computable=false` with a `missing` list rather than crashing, unless the pack
directory itself is unreadable.

## Testing

Add tests for:

- Pack discovery finds the three seed packs.
- Each seed pack validates.
- `inspect` returns merged JSON.
- `match` ranks an obvious problem to the expected pack.
- `preflight` returns `ai_verifiable=true` for the three seed packs.
- A fixture with a missing evaluator reports `computable=false` and a useful
  missing-field list.

Run:

```bash
uv run pytest tests/test_computable.py
uv run pytest
uv run ruff check .
```

Note: current ruff failures in generated `experiments/auto/.../bench.py` are
outside this feature. The implementation plan should either exclude generated
auto benches from ruff or clean those generated artifacts before using ruff as a
completion gate.

## Acceptance Criteria

- `uv run python -m pipeline.computable list` shows the three seed packs.
- `uv run python -m pipeline.computable inspect short_string_compression` prints
  valid merged JSON.
- `uv run python -m pipeline.computable match --problem "compress short HTTP
  headers better than HPACK"` ranks `short_string_compression` first.
- `uv run python -m pipeline.computable preflight small_points_optimization`
  returns `computable=true` and a smoke command.
- Tests cover loader, CLI helpers, matching, and preflight.
- No compose or autoverify behavior changes in this slice.

## Self-Review Notes

- No placeholders are left for the first slice.
- Scope is limited to framework plus three seed packs.
- Compose integration is explicitly deferred.
- The user is not asked to provide domain expertise; packs carry executable
  verification contracts.
