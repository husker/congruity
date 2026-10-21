# Computable Problem Packs Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a computable problem pack framework plus three seed packs so Congruity can target problem families that an AI agent can verify by running code.

**Architecture:** Create a new `pipeline.computable` module with strict Pydantic models, pack loading, deterministic matching, structural preflight, and a small CLI. Keep pack data in a new `computable_packs/` directory, separate from broad `domain_packs/`. Do not change compose or autoverify behavior in this slice.

**Tech Stack:** Python 3.11+, Pydantic v2, argparse, pytest, ruff, existing `uv` workflow.

---

## File Structure

- Create `pipeline/computable.py`
  - Owns schemas, pack loading, deterministic matching, preflight, and CLI entry point.
- Create `tests/test_computable.py`
  - Covers pack discovery, validation, JSON inspect shape, matching, preflight, and invalid-pack diagnostics.
- Create `computable_packs/short_string_compression/`
  - Executable verification contract for lossless short-record compression.
- Create `computable_packs/small_points_optimization/`
  - Executable verification contract for geometric point-configuration optimization.
- Create `computable_packs/cache_replacement/`
  - Executable verification contract for cache-policy trace replay.
- Modify `pyproject.toml`
  - Exclude generated `experiments/auto/` benches from ruff, because those are quarantined generated artifacts and currently block `uv run ruff check .`.

---

### Task 1: Write Computable Pack Tests

**Files:**
- Create: `tests/test_computable.py`

- [ ] **Step 1: Create the failing test file**

Create `tests/test_computable.py` with this content:

```python
"""Tests for computable problem pack loading, matching, and preflight."""

from __future__ import annotations

import json


def test_list_packs_includes_three_seed_packs():
    from pipeline.computable import list_packs

    names = list_packs()

    assert names == [
        "cache_replacement",
        "short_string_compression",
        "small_points_optimization",
    ]


def test_load_short_string_pack_has_executable_contract():
    from pipeline.computable import load_pack

    pack = load_pack("short_string_compression")

    assert pack.pack.id == "short_string_compression"
    assert pack.pack.problem_shape.verification_kind == "roundtrip"
    assert pack.evaluator.metric_primary == "compression_ratio"
    assert pack.evaluator.higher_is_better is False
    assert any(b.level == "strong_reproducible" for b in pack.baselines.baselines)
    assert any(d.id == "real_http_headers" for d in pack.datasets.datasets)
    assert pack.gotchas


def test_pack_to_dict_is_json_serializable():
    from pipeline.computable import load_pack, pack_to_dict

    pack = load_pack("cache_replacement")
    data = pack_to_dict(pack)

    assert data["pack"]["id"] == "cache_replacement"
    assert data["evaluator"]["metric_primary"] == "hit_rate"
    assert data["gotchas"]
    json.dumps(data)


def test_match_problem_ranks_short_string_first():
    from pipeline.computable import match_problem

    matches = match_problem("compress short HTTP headers better than HPACK")

    assert matches[0]["pack_id"] == "short_string_compression"
    assert matches[0]["score"] > 0


def test_match_problem_ranks_cache_first():
    from pipeline.computable import match_problem

    matches = match_problem("design a cache eviction policy for scan workloads")

    assert matches[0]["pack_id"] == "cache_replacement"
    assert matches[0]["score"] > 0


def test_preflight_seed_packs_are_ai_verifiable():
    from pipeline.computable import preflight_pack

    for name in [
        "short_string_compression",
        "small_points_optimization",
        "cache_replacement",
    ]:
        result = preflight_pack(name)
        assert result.computable is True
        assert result.ai_verifiable is True
        assert result.missing == []
        assert "strong_reproducible" in result.bar_levels_available
        assert result.smoke_command
        assert result.recommended_compose_constraints


def test_preflight_reports_missing_files(tmp_path):
    from pipeline.computable import preflight_pack

    d = tmp_path / "broken_pack"
    d.mkdir()
    (d / "pack.json").write_text(json.dumps({
        "id": "broken_pack",
        "display_name": "Broken Pack",
        "description": "Missing executable files",
        "problem_shape": {
            "objective": "x",
            "input_representation": "x",
            "output_representation": "x",
            "constraints": ["x"],
            "failure_modes": ["x"],
            "verification_kind": "benchmark",
        },
        "recommended_problem_template": "x",
        "compose_query_hints": ["x"],
    }))

    result = preflight_pack("broken_pack", root=tmp_path)

    assert result.computable is False
    assert result.ai_verifiable is False
    assert "evaluator.json" in result.missing
    assert "baselines.json" in result.missing
    assert "datasets.json" in result.missing
    assert "sota.json" in result.missing


def test_main_list_subcommand_prints_seed_pack(capsys):
    from pipeline.computable import main

    rc = main(["list"])
    captured = capsys.readouterr()

    assert rc == 0
    assert "short_string_compression" in captured.out
    assert "cache_replacement" in captured.out


def test_main_inspect_subcommand_emits_json(capsys):
    from pipeline.computable import main

    rc = main(["inspect", "short_string_compression"])
    captured = capsys.readouterr()

    assert rc == 0
    parsed = json.loads(captured.out)
    assert parsed["pack"]["id"] == "short_string_compression"
    assert parsed["evaluator"]["validator"] == "lossless_roundtrip"


def test_main_match_subcommand_emits_ranked_json(capsys):
    from pipeline.computable import main

    rc = main(["match", "--problem", "maximize points in a square by physics relaxation"])
    captured = capsys.readouterr()

    assert rc == 0
    parsed = json.loads(captured.out)
    assert parsed[0]["pack_id"] == "small_points_optimization"


def test_main_preflight_subcommand_emits_json(capsys):
    from pipeline.computable import main

    rc = main(["preflight", "cache_replacement"])
    captured = capsys.readouterr()

    assert rc == 0
    parsed = json.loads(captured.out)
    assert parsed["pack_id"] == "cache_replacement"
    assert parsed["ai_verifiable"] is True
```

- [ ] **Step 2: Run the new tests to verify they fail**

Run:

```bash
uv run pytest tests/test_computable.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'pipeline.computable'`.

- [ ] **Step 3: Commit the failing tests**

```bash
git add tests/test_computable.py
git commit -m "test: add computable pack contract tests"
```

---

### Task 2: Implement the Computable Pack Module

**Files:**
- Create: `pipeline/computable.py`
- Test: `tests/test_computable.py`

- [ ] **Step 1: Create `pipeline/computable.py`**

Create `pipeline/computable.py` with this content:

```python
"""Computable problem packs.

A computable pack is an executable verification contract for one problem shape.
It tells Congruity which outputs are valid, which metric to use, which baselines
are available, and what success bars can be checked by code.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

PACKS_DIR = Path(__file__).resolve().parent.parent / "computable_packs"
BAR_LEVELS = ("naive", "strong_reproducible", "record_cached")
VERIFICATION_KINDS = (
    "exact_oracle",
    "roundtrip",
    "benchmark",
    "backtest",
    "record_comparison",
    "property_test",
)


class _StrictBase(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ProblemShape(_StrictBase):
    objective: str = Field(min_length=1)
    input_representation: str = Field(min_length=1)
    output_representation: str = Field(min_length=1)
    constraints: list[str] = Field(min_length=1)
    failure_modes: list[str] = Field(min_length=1)
    verification_kind: Literal[
        "exact_oracle",
        "roundtrip",
        "benchmark",
        "backtest",
        "record_comparison",
        "property_test",
    ]


class PackMeta(_StrictBase):
    id: str = Field(min_length=1)
    display_name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    problem_shape: ProblemShape
    recommended_problem_template: str = Field(min_length=1)
    compose_query_hints: list[str] = Field(min_length=1)


class EvaluatorSpec(_StrictBase):
    metric_primary: str = Field(min_length=1)
    higher_is_better: bool
    validator: str = Field(min_length=1)
    smoke_command: str = Field(min_length=1)
    full_command: str = Field(min_length=1)
    pass_criteria: str = Field(min_length=1)


class BaselineSpec(_StrictBase):
    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    level: Literal["naive", "strong_reproducible", "record_cached"]
    command: str = ""
    notes: str = ""


class BaselinesFile(_StrictBase):
    baselines: list[BaselineSpec] = Field(min_length=1)


class DatasetSpec(_StrictBase):
    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    kind: str = Field(min_length=1)
    availability: str = Field(min_length=1)
    command: str = ""
    notes: str = ""


class DatasetsFile(_StrictBase):
    datasets: list[DatasetSpec] = Field(min_length=1)


class SotaLevel(_StrictBase):
    level: Literal["naive", "strong_reproducible", "record_cached"]
    available: bool
    comparator: str = Field(min_length=1)
    source: str = ""
    notes: str = ""


class SotaFile(_StrictBase):
    bar_levels: list[SotaLevel] = Field(min_length=1)


class ComputablePack(_StrictBase):
    pack: PackMeta
    evaluator: EvaluatorSpec
    baselines: BaselinesFile
    datasets: DatasetsFile
    sota: SotaFile
    gotchas: str = ""


class PreflightResult(_StrictBase):
    pack_id: str
    computable: bool
    ai_verifiable: bool
    missing: list[str] = Field(default_factory=list)
    bar_levels_available: list[str] = Field(default_factory=list)
    recommended_compose_constraints: list[str] = Field(default_factory=list)
    smoke_command: str = ""


def list_packs(root: Path = PACKS_DIR) -> list[str]:
    """Return computable pack ids in deterministic order."""
    if not root.exists():
        return []
    return sorted(p.name for p in root.iterdir() if p.is_dir())


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as e:
        raise ValueError(f"{path}: invalid JSON: {e}") from e


def load_pack(name: str, root: Path = PACKS_DIR) -> ComputablePack:
    """Load and validate one computable pack."""
    d = root / name
    if not d.is_dir():
        raise FileNotFoundError(f"computable pack {name!r} not found at {d}")

    try:
        pack = PackMeta.model_validate(_read_json(d / "pack.json"))
        evaluator = EvaluatorSpec.model_validate(_read_json(d / "evaluator.json"))
        baselines = BaselinesFile.model_validate(_read_json(d / "baselines.json"))
        datasets = DatasetsFile.model_validate(_read_json(d / "datasets.json"))
        sota = SotaFile.model_validate(_read_json(d / "sota.json"))
    except FileNotFoundError as e:
        raise FileNotFoundError(f"{d}: missing required file {Path(e.filename).name}") from e
    except ValidationError as e:
        raise ValueError(f"{d}: schema validation failed: {e}") from e

    if pack.id != name:
        raise ValueError(f"{d}: pack.json id {pack.id!r} does not match directory {name!r}")

    gotchas_path = d / "gotchas.md"
    gotchas = gotchas_path.read_text() if gotchas_path.exists() else ""
    return ComputablePack(
        pack=pack,
        evaluator=evaluator,
        baselines=baselines,
        datasets=datasets,
        sota=sota,
        gotchas=gotchas,
    )


def pack_to_dict(pack: ComputablePack) -> dict:
    """JSON-serializable representation of a loaded pack."""
    return pack.model_dump()


def _tokens(text: str) -> set[str]:
    return {
        t
        for t in re.findall(r"[a-z0-9]+", text.lower())
        if len(t) >= 3
    }


def _pack_search_text(pack: ComputablePack) -> str:
    p = pack.pack
    s = p.problem_shape
    parts = [
        p.id,
        p.display_name,
        p.description,
        p.recommended_problem_template,
        s.objective,
        s.input_representation,
        s.output_representation,
        s.verification_kind,
        " ".join(s.constraints),
        " ".join(s.failure_modes),
        " ".join(p.compose_query_hints),
        " ".join(b.name for b in pack.baselines.baselines),
        " ".join(d.name for d in pack.datasets.datasets),
    ]
    return " ".join(parts)


def match_problem(problem: str, root: Path = PACKS_DIR) -> list[dict]:
    """Rank packs by deterministic token overlap against the problem text."""
    query = _tokens(problem)
    matches = []
    for name in list_packs(root):
        pack = load_pack(name, root=root)
        haystack = _tokens(_pack_search_text(pack))
        overlap = sorted(query & haystack)
        score = len(overlap)
        matches.append({
            "pack_id": name,
            "display_name": pack.pack.display_name,
            "score": score,
            "matched_terms": overlap,
        })
    matches.sort(key=lambda m: (-m["score"], m["pack_id"]))
    return matches


def _missing_files(pack_dir: Path) -> list[str]:
    required = ["pack.json", "evaluator.json", "baselines.json", "datasets.json", "sota.json"]
    return [name for name in required if not (pack_dir / name).exists()]


def preflight_pack(name: str, root: Path = PACKS_DIR) -> PreflightResult:
    """Return a structural computability verdict for one pack."""
    pack_dir = root / name
    missing = _missing_files(pack_dir)
    if missing:
        return PreflightResult(
            pack_id=name,
            computable=False,
            ai_verifiable=False,
            missing=missing,
            bar_levels_available=[],
            recommended_compose_constraints=[],
            smoke_command="",
        )

    try:
        pack = load_pack(name, root=root)
    except (FileNotFoundError, ValueError) as e:
        return PreflightResult(
            pack_id=name,
            computable=False,
            ai_verifiable=False,
            missing=[str(e)],
            bar_levels_available=[],
            recommended_compose_constraints=[],
            smoke_command="",
        )

    available_bars = [
        b.level
        for b in pack.sota.bar_levels
        if b.available
    ]
    has_strong_baseline = any(
        b.level == "strong_reproducible" and bool(b.command)
        for b in pack.baselines.baselines
    )
    has_dataset = bool(pack.datasets.datasets)
    has_smoke = bool(pack.evaluator.smoke_command)
    computable = bool(has_smoke and pack.evaluator.validator and pack.evaluator.metric_primary)
    ai_verifiable = computable and has_dataset and has_strong_baseline
    constraints = [
        f"objective: {pack.pack.problem_shape.objective}",
        f"validator: {pack.evaluator.validator}",
        f"metric: {pack.evaluator.metric_primary}",
        f"pass: {pack.evaluator.pass_criteria}",
    ]

    missing_fields = []
    if not has_smoke:
        missing_fields.append("evaluator.smoke_command")
    if not has_dataset:
        missing_fields.append("datasets.datasets")
    if not has_strong_baseline:
        missing_fields.append("baselines.strong_reproducible.command")

    return PreflightResult(
        pack_id=name,
        computable=computable,
        ai_verifiable=ai_verifiable,
        missing=missing_fields,
        bar_levels_available=available_bars,
        recommended_compose_constraints=constraints,
        smoke_command=pack.evaluator.smoke_command,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Computable problem pack inspection")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list", help="list available computable packs")

    inspect_p = sub.add_parser("inspect", help="print merged pack JSON")
    inspect_p.add_argument("name")

    match_p = sub.add_parser("match", help="rank packs for a free-form problem")
    match_p.add_argument("--problem", required=True)

    preflight_p = sub.add_parser("preflight", help="check whether a pack is AI-verifiable")
    preflight_p.add_argument("name")

    args = parser.parse_args(argv if argv is not None else sys.argv[1:])

    try:
        if args.cmd == "list":
            for name in list_packs():
                pack = load_pack(name)
                print(f"{name}\t{pack.pack.display_name}")
            return 0
        if args.cmd == "inspect":
            print(json.dumps(pack_to_dict(load_pack(args.name)), indent=2))
            return 0
        if args.cmd == "match":
            print(json.dumps(match_problem(args.problem), indent=2))
            return 0
        if args.cmd == "preflight":
            print(json.dumps(preflight_pack(args.name).model_dump(), indent=2))
            return 0
    except (FileNotFoundError, ValueError) as e:
        print(str(e), file=sys.stderr)
        return 1

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Run the tests to verify the module exists but packs are missing**

Run:

```bash
uv run pytest tests/test_computable.py -v
```

Expected: FAIL because `list_packs()` returns `[]` and seed pack files do not exist yet.

- [ ] **Step 3: Commit the module**

```bash
git add pipeline/computable.py
git commit -m "feat: add computable pack loader"
```

---

### Task 3: Add the Three Seed Packs

**Files:**
- Create: `computable_packs/short_string_compression/pack.json`
- Create: `computable_packs/short_string_compression/evaluator.json`
- Create: `computable_packs/short_string_compression/baselines.json`
- Create: `computable_packs/short_string_compression/datasets.json`
- Create: `computable_packs/short_string_compression/sota.json`
- Create: `computable_packs/short_string_compression/gotchas.md`
- Create: `computable_packs/small_points_optimization/pack.json`
- Create: `computable_packs/small_points_optimization/evaluator.json`
- Create: `computable_packs/small_points_optimization/baselines.json`
- Create: `computable_packs/small_points_optimization/datasets.json`
- Create: `computable_packs/small_points_optimization/sota.json`
- Create: `computable_packs/small_points_optimization/gotchas.md`
- Create: `computable_packs/cache_replacement/pack.json`
- Create: `computable_packs/cache_replacement/evaluator.json`
- Create: `computable_packs/cache_replacement/baselines.json`
- Create: `computable_packs/cache_replacement/datasets.json`
- Create: `computable_packs/cache_replacement/sota.json`
- Create: `computable_packs/cache_replacement/gotchas.md`
- Test: `tests/test_computable.py`

- [ ] **Step 1: Add `short_string_compression`**

Create `computable_packs/short_string_compression/pack.json`:

```json
{
  "id": "short_string_compression",
  "display_name": "Short String Compression",
  "description": "Lossless compression for short structured records where every candidate must round-trip exactly and beat strong codec baselines on byte ratio.",
  "problem_shape": {
    "objective": "Minimize encoded byte length for short records while preserving exact decoding.",
    "input_representation": "Sequences of 8-256 byte records such as HTTP headers, JSON snippets, log lines, or templated identifiers.",
    "output_representation": "A byte string plus any shared dictionary or model state allowed by the benchmark.",
    "constraints": [
      "Every decoded record must exactly match its input.",
      "Shared dictionaries or model state must be trained only on training data.",
      "Compression ratio must include transmitted side information unless explicitly pre-shared."
    ],
    "failure_modes": [
      "Lossy transforms that appear to compress but fail round-trip.",
      "Dictionary leakage from held-out test data.",
      "Beating stateless codecs while losing to stateful HPACK on HTTP-like data."
    ],
    "verification_kind": "roundtrip"
  },
  "recommended_problem_template": "Compress short structured records (8-256 bytes) with exact round-trip and a compression ratio better than the strongest stateful or dictionary baseline on a held-out corpus.",
  "compose_query_hints": [
    "short string compression",
    "HTTP header compression",
    "biosequence tokenization",
    "finite state coding",
    "dictionary compression",
    "arithmetic coding",
    "small alphabet transforms"
  ]
}
```

Create `computable_packs/short_string_compression/evaluator.json`:

```json
{
  "metric_primary": "compression_ratio",
  "higher_is_better": false,
  "validator": "lossless_roundtrip",
  "smoke_command": "uv run --with brotli --with zstandard --with hpack python -m experiments.c1_short_string.bench",
  "full_command": "uv run --with brotli --with zstandard --with hpack python -m experiments.c1_short_string.bench",
  "pass_criteria": "A candidate must round-trip every held-out record and beat the strongest applicable baseline by at least 2% relative compression ratio."
}
```

Create `computable_packs/short_string_compression/baselines.json`:

```json
{
  "baselines": [
    {
      "id": "gzip",
      "name": "gzip",
      "level": "naive",
      "command": "uv run python -m experiments.c1_short_string.bench",
      "notes": "Generic fallback baseline; insufficient as the final bar."
    },
    {
      "id": "brotli_dictionary",
      "name": "Brotli with shared dictionary",
      "level": "strong_reproducible",
      "command": "uv run --with brotli python -m experiments.c1_short_string.bench",
      "notes": "Strong stateless dictionary baseline for short records."
    },
    {
      "id": "zstd_dictionary",
      "name": "Zstandard with trained dictionary",
      "level": "strong_reproducible",
      "command": "uv run --with zstandard python -m experiments.c1_short_string.bench",
      "notes": "Strong stateless dictionary baseline."
    },
    {
      "id": "hpack_stateful",
      "name": "HPACK stateful HTTP header compression",
      "level": "record_cached",
      "command": "uv run --with hpack python -m experiments.c1_short_string.bench",
      "notes": "Specialist HTTP baseline; only applies to HTTP-like header streams."
    }
  ]
}
```

Create `computable_packs/short_string_compression/datasets.json`:

```json
{
  "datasets": [
    {
      "id": "real_http_headers",
      "name": "Real HTTP headers cache",
      "kind": "local_json",
      "availability": "local_if_present",
      "command": "test -f data/real_http_headers.json",
      "notes": "Collected by experiments/c1_short_string/fetch_real_headers.py."
    },
    {
      "id": "synthetic_templated_records",
      "name": "Synthetic templated short records",
      "kind": "generated",
      "availability": "always",
      "command": "uv run python -m experiments.c1_short_string.bench",
      "notes": "Used for smoke tests when real corpora are unavailable."
    },
    {
      "id": "json_snippets",
      "name": "Short JSON snippets",
      "kind": "generated_or_collected",
      "availability": "optional",
      "command": "",
      "notes": "Useful off-HTTP generalization set."
    }
  ]
}
```

Create `computable_packs/short_string_compression/sota.json`:

```json
{
  "bar_levels": [
    {
      "level": "naive",
      "available": true,
      "comparator": "gzip ratio on the same held-out records",
      "source": "stdlib gzip",
      "notes": "Only a sanity check."
    },
    {
      "level": "strong_reproducible",
      "available": true,
      "comparator": "best of Brotli dictionary and Zstandard dictionary on the same split",
      "source": "experiments/c1_short_string/bench.py",
      "notes": "Primary local bar for non-HTTP data."
    },
    {
      "level": "record_cached",
      "available": true,
      "comparator": "HPACK stateful compression for HTTP-like header streams",
      "source": "hpack package plus existing bench",
      "notes": "Specialist bar; candidates that only beat stateless codecs are partial wins."
    }
  ]
}
```

Create `computable_packs/short_string_compression/gotchas.md`:

```markdown
# Short String Compression Gotchas

- Round-trip validation is mandatory for every evaluated record.
- Include dictionary or model side information in the byte budget unless the
  benchmark explicitly declares it pre-shared.
- Do not train dictionaries, tokenizers, or templates on held-out records.
- HPACK is the strongest specialist bar for HTTP-like streams; a method that
  beats Brotli or Zstd but loses to HPACK is only a partial result.
- Compression ratio and latency are different metrics. Do not claim one from the
  other.
```

- [ ] **Step 2: Add `small_points_optimization`**

Create `computable_packs/small_points_optimization/pack.json`:

```json
{
  "id": "small_points_optimization",
  "display_name": "Small Points Optimization",
  "description": "Geometric point-configuration optimization with deterministic validators and strong local optimization baselines.",
  "problem_shape": {
    "objective": "Maximize a recomputable geometric objective over finite point sets.",
    "input_representation": "Problem size n, geometric domain, random seed, and objective parameters.",
    "output_representation": "A finite list of point coordinates in the declared domain.",
    "constraints": [
      "All coordinates must lie inside the declared domain.",
      "The score must be recomputed from emitted coordinates.",
      "Invalid or non-finite coordinates fail validation before scoring."
    ],
    "failure_modes": [
      "Candidate improves against weak simulated annealing but loses to tuned CMA-ES or differential evolution.",
      "Candidate depends on a surrogate objective that does not improve the true geometric metric.",
      "Candidate violates domain bounds or produces near-degenerate configurations."
    ],
    "verification_kind": "benchmark"
  },
  "recommended_problem_template": "Optimize a finite point configuration in a bounded geometric domain; beat tuned reproducible optimizers on the recomputed objective across fixed seeds.",
  "compose_query_hints": [
    "geometric optimization",
    "point packing",
    "energy minimization",
    "molecular packing",
    "mesh relaxation",
    "active set repair",
    "simulated annealing",
    "CMA-ES"
  ]
}
```

Create `computable_packs/small_points_optimization/evaluator.json`:

```json
{
  "metric_primary": "objective_value",
  "higher_is_better": true,
  "validator": "bounds_and_objective_recomputation",
  "smoke_command": "uv run --with numpy --with scipy --with cma --with jax python -m experiments.heilbronn.bench",
  "full_command": "uv run --with numpy --with scipy --with cma --with jax python -m experiments.heilbronn.bench",
  "pass_criteria": "A candidate must pass coordinate validation and beat the best strong reproducible baseline by at least 5% mean objective value across fixed seeds."
}
```

Create `computable_packs/small_points_optimization/baselines.json`:

```json
{
  "baselines": [
    {
      "id": "random_restart",
      "name": "Random restart search",
      "level": "naive",
      "command": "uv run python -m experiments.heilbronn.bench",
      "notes": "Sanity baseline only."
    },
    {
      "id": "simulated_annealing",
      "name": "Tuned simulated annealing",
      "level": "strong_reproducible",
      "command": "uv run --with numpy --with scipy python -m experiments.heilbronn.bench",
      "notes": "Standard local-search baseline used in prior verification."
    },
    {
      "id": "basin_hopping",
      "name": "SciPy basin hopping",
      "level": "strong_reproducible",
      "command": "uv run --with numpy --with scipy python -m experiments.heilbronn.bench",
      "notes": "Useful but must be tuned; defaults are weak."
    },
    {
      "id": "cma_es",
      "name": "CMA-ES",
      "level": "strong_reproducible",
      "command": "uv run --with cma --with numpy python -m experiments.heilbronn.bench",
      "notes": "Strong optimizer bar for smooth or relaxed formulations."
    },
    {
      "id": "published_reference",
      "name": "Published reference values",
      "level": "record_cached",
      "command": "",
      "notes": "Comparison-only unless coordinates and exact objective definitions are local."
    }
  ]
}
```

Create `computable_packs/small_points_optimization/datasets.json`:

```json
{
  "datasets": [
    {
      "id": "fixed_n_seed_grid",
      "name": "Fixed n and seed grid",
      "kind": "generated",
      "availability": "always",
      "command": "uv run python -m experiments.heilbronn.bench",
      "notes": "Primary reproducible benchmark shape."
    },
    {
      "id": "published_reference_values",
      "name": "Published geometric reference values",
      "kind": "reference_table",
      "availability": "optional",
      "command": "",
      "notes": "Used as record-cached comparison when objective definitions match exactly."
    }
  ]
}
```

Create `computable_packs/small_points_optimization/sota.json`:

```json
{
  "bar_levels": [
    {
      "level": "naive",
      "available": true,
      "comparator": "random restart or untuned local search",
      "source": "experiments/heilbronn/bench.py",
      "notes": "Only catches broken methods."
    },
    {
      "level": "strong_reproducible",
      "available": true,
      "comparator": "best of tuned SA, basin hopping, CMA-ES, and differential evolution where wired",
      "source": "experiments/heilbronn/",
      "notes": "Primary AI-verifiable bar."
    },
    {
      "level": "record_cached",
      "available": true,
      "comparator": "published reference objective value when the exact objective matches",
      "source": "reference values cited in findings",
      "notes": "Comparison-only unless exact record coordinates are local."
    }
  ]
}
```

Create `computable_packs/small_points_optimization/gotchas.md`:

```markdown
# Small Points Optimization Gotchas

- Recompute the objective from emitted coordinates; never trust a method's own
  reported score.
- Compare against tuned optimizers, not default `scipy.optimize.basinhopping`.
- Use multiple fixed seeds and report mean plus spread.
- A surrogate objective win is not a result unless it improves the true metric.
- Published records may be for subtly different objectives or constraints.
```

- [ ] **Step 3: Add `cache_replacement`**

Create `computable_packs/cache_replacement/pack.json`:

```json
{
  "id": "cache_replacement",
  "display_name": "Cache Replacement",
  "description": "Online cache eviction policies verified by deterministic trace replay, capacity checks, and strong cache-policy baselines.",
  "problem_shape": {
    "objective": "Maximize hit rate or minimize miss ratio under a fixed cache capacity.",
    "input_representation": "A deterministic access trace and cache capacity.",
    "output_representation": "Eviction decisions made online during trace replay.",
    "constraints": [
      "Cache capacity must never be exceeded.",
      "Non-oracle policies cannot use future accesses.",
      "Every policy must replay the same trace with the same capacity."
    ],
    "failure_modes": [
      "Future leakage disguised as prediction.",
      "Synthetic Zipf-only evaluation that hides scan or loop failures.",
      "Prefetching mechanisms misapplied as eviction policies."
    ],
    "verification_kind": "benchmark"
  },
  "recommended_problem_template": "Design an online cache eviction policy that beats SIEVE, ARC, and S3FIFO hit rate on scan, loop, and skewed traces at fixed cache capacities.",
  "compose_query_hints": [
    "cache replacement",
    "online scheduling",
    "eviction policy",
    "prefetching",
    "time to next use",
    "streaming sketches",
    "bandit allocation"
  ]
}
```

Create `computable_packs/cache_replacement/evaluator.json`:

```json
{
  "metric_primary": "hit_rate",
  "higher_is_better": true,
  "validator": "capacity_and_no_future_leakage",
  "smoke_command": "uv run python -m experiments.cache_replacement.bench",
  "full_command": "uv run python -m experiments.cache_replacement.bench",
  "pass_criteria": "A candidate must obey capacity, avoid future leakage, and beat the best strong reproducible baseline by at least 1 percentage point hit rate on non-Zipf stress traces."
}
```

Create `computable_packs/cache_replacement/baselines.json`:

```json
{
  "baselines": [
    {
      "id": "fifo",
      "name": "FIFO",
      "level": "naive",
      "command": "uv run python -m experiments.cache_replacement.bench",
      "notes": "Simple queue baseline."
    },
    {
      "id": "lru",
      "name": "LRU",
      "level": "naive",
      "command": "uv run python -m experiments.cache_replacement.bench",
      "notes": "Common baseline but not sufficient alone."
    },
    {
      "id": "arc",
      "name": "ARC",
      "level": "strong_reproducible",
      "command": "uv run python -m experiments.cache_replacement.bench",
      "notes": "Adaptive recency/frequency baseline."
    },
    {
      "id": "sieve",
      "name": "SIEVE",
      "level": "strong_reproducible",
      "command": "uv run python -m experiments.cache_replacement.bench",
      "notes": "Strong simple eviction baseline from prior verification."
    },
    {
      "id": "s3fifo",
      "name": "S3FIFO",
      "level": "strong_reproducible",
      "command": "uv run python -m experiments.cache_replacement.bench",
      "notes": "Strong FIFO-family baseline."
    }
  ]
}
```

Create `computable_packs/cache_replacement/datasets.json`:

```json
{
  "datasets": [
    {
      "id": "synthetic_zipf",
      "name": "Synthetic Zipf traces",
      "kind": "generated",
      "availability": "always",
      "command": "uv run python -m experiments.cache_replacement.bench",
      "notes": "Useful but too easy as the only benchmark."
    },
    {
      "id": "synthetic_loop_scan_hot",
      "name": "Loop, scan, and scan-with-hot traces",
      "kind": "generated",
      "availability": "always",
      "command": "uv run python -m experiments.cache_replacement.bench",
      "notes": "Stress traces that expose most weak eviction policies."
    },
    {
      "id": "public_block_traces",
      "name": "Public block traces",
      "kind": "external_optional",
      "availability": "optional",
      "command": "",
      "notes": "SNIA or MSR-style traces when available locally."
    }
  ]
}
```

Create `computable_packs/cache_replacement/sota.json`:

```json
{
  "bar_levels": [
    {
      "level": "naive",
      "available": true,
      "comparator": "FIFO and LRU on the same traces",
      "source": "experiments/cache_replacement/bench.py",
      "notes": "Sanity bar only."
    },
    {
      "level": "strong_reproducible",
      "available": true,
      "comparator": "best of ARC, SIEVE, and S3FIFO on the same traces",
      "source": "experiments/cache_replacement/",
      "notes": "Primary local bar."
    },
    {
      "level": "record_cached",
      "available": false,
      "comparator": "published trace-specific leaderboard or industrial benchmark",
      "source": "",
      "notes": "Not currently available in the repo."
    }
  ]
}
```

Create `computable_packs/cache_replacement/gotchas.md`:

```markdown
# Cache Replacement Gotchas

- Zipf-only traces are not enough; include loop, scan, and scan-with-hot traces.
- Check capacity after every access.
- Unless a policy is explicitly an oracle baseline, it cannot inspect future
  accesses.
- Successor prediction is usually a prefetching signal, not an eviction signal.
- Compare against SIEVE, ARC, and S3FIFO before claiming a meaningful win.
```

- [ ] **Step 4: Run the tests**

Run:

```bash
uv run pytest tests/test_computable.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit the seed packs**

```bash
git add computable_packs tests/test_computable.py
git commit -m "feat: add computable seed packs"
```

---

### Task 4: Verify CLI Behavior

**Files:**
- Modify only if needed: `pipeline/computable.py`
- Test: `tests/test_computable.py`

- [ ] **Step 1: Run list command**

Run:

```bash
uv run python -m pipeline.computable list
```

Expected output includes:

```text
cache_replacement	Cache Replacement
short_string_compression	Short String Compression
small_points_optimization	Small Points Optimization
```

- [ ] **Step 2: Run inspect command**

Run:

```bash
uv run python -m pipeline.computable inspect short_string_compression
```

Expected: JSON with `pack.id` equal to `short_string_compression` and
`evaluator.validator` equal to `lossless_roundtrip`.

- [ ] **Step 3: Run match command**

Run:

```bash
uv run python -m pipeline.computable match --problem "compress short HTTP headers better than HPACK"
```

Expected: first JSON object has `"pack_id": "short_string_compression"`.

- [ ] **Step 4: Run preflight command**

Run:

```bash
uv run python -m pipeline.computable preflight cache_replacement
```

Expected: JSON includes:

```json
{
  "pack_id": "cache_replacement",
  "computable": true,
  "ai_verifiable": true
}
```

- [ ] **Step 5: Run all computable tests again**

Run:

```bash
uv run pytest tests/test_computable.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit any CLI fixes**

If no fixes were needed, skip this step. If `pipeline/computable.py` changed:

```bash
git add pipeline/computable.py tests/test_computable.py
git commit -m "fix: polish computable CLI behavior"
```

---

### Task 5: Exclude Generated Auto Benches From Ruff

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Update ruff config**

Modify the existing `[tool.ruff]` section in `pyproject.toml` from:

```toml
[tool.ruff]
line-length = 100
target-version = "py311"
```

to:

```toml
[tool.ruff]
line-length = 100
target-version = "py311"
extend-exclude = ["experiments/auto"]
```

- [ ] **Step 2: Run ruff**

Run:

```bash
uv run ruff check .
```

Expected: PASS. The current generated-auto-bench lint failures should disappear.

- [ ] **Step 3: Commit ruff boundary**

```bash
git add pyproject.toml
git commit -m "chore: exclude generated auto benches from ruff"
```

---

### Task 6: Full Verification

**Files:**
- No new files unless a prior task exposed a bug.

- [ ] **Step 1: Run focused tests**

Run:

```bash
uv run pytest tests/test_computable.py -v
```

Expected: PASS.

- [ ] **Step 2: Run full tests**

Run:

```bash
uv run pytest
```

Expected: PASS.

- [ ] **Step 3: Run ruff**

Run:

```bash
uv run ruff check .
```

Expected: PASS.

- [ ] **Step 4: Run representative CLI commands**

Run:

```bash
uv run python -m pipeline.computable list
uv run python -m pipeline.computable match --problem "maximize geometric point layout with physics energy relaxation"
uv run python -m pipeline.computable preflight small_points_optimization
```

Expected:

- `list` prints all three seed packs.
- `match` ranks `small_points_optimization` first.
- `preflight` returns `computable=true` and `ai_verifiable=true`.

- [ ] **Step 5: Inspect git status**

Run:

```bash
git status --short
```

Expected: only intended changes from this plan are present. Existing unrelated
untracked generated artifacts may remain; do not remove or stage them unless they
were created by this implementation.

---

## Self-Review

Spec coverage:

- Framework layer: Task 2.
- Three seed packs: Task 3.
- CLI commands: Tasks 2 and 4.
- Structural preflight: Task 2.
- Tests: Tasks 1, 3, 4, and 6.
- Compose integration deferred: no task changes compose or autoverify.
- Non-expert computability constraint: encoded in pack schemas, preflight, bar
  levels, validators, and seed pack content.

Placeholder scan:

- No placeholder markers or unspecified implementation steps.
- Every created file has concrete content.
- Every command has an expected result.

Type consistency:

- Test names match functions defined in `pipeline/computable.py`.
- `PreflightResult` fields match CLI JSON assertions.
- Pack JSON fields match Pydantic model names.
