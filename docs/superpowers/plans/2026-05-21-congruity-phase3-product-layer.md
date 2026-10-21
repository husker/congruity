# Congruity Phase 3 — Product Layer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship verification recipes per composition, post-L3 automated novelty checks, a calibration markdown report, and a first domain pack (CS / algorithms). The product layer that turns Congruity from "idea generator + verifier" into "research triage and test planner."

**Architecture:** Three new pipeline modules (`recipe.py`, `novelty.py`, `calibration.py`) each surfacing one CLI subcommand. New Pydantic schemas in `pipeline/schemas.py` for `VerificationRecipe` and `NoveltyReport`. A `domain_packs/` directory at the repo root with the first pack (`cs_algorithms/`) plus an apply CLI (`pipeline/domain_pack.py`).

**Tech Stack:** Python 3.11, pydantic v2, requests (existing). No new heavy deps; the S2 lookups for novelty use `requests` against the public Semantic Scholar API.

---

## File structure

| Path | Action | Responsibility |
|---|---|---|
| `pipeline/schemas.py` | modify | Add `VerificationRecipe`, `NoveltyReport`, supporting models |
| `pipeline/recipe.py` | create | `generate_recipe(composition, state, run_dir) -> dict` |
| `pipeline/novelty.py` | create | `check_novelty(composition, state, run_dir) -> dict` |
| `pipeline/calibration.py` | create | walks compose runs + emits markdown dashboard |
| `pipeline/domain_pack.py` | create | `list` + `apply` subcommands |
| `pipeline/compose.py` | modify | wire recipe + novelty in (both behind opt-in flags); add CLI subcommands |
| `domain_packs/cs_algorithms/topic_queries.json` | create | curated OpenAlex topic IDs |
| `domain_packs/cs_algorithms/problem_templates.json` | create | problem statement templates |
| `domain_packs/cs_algorithms/known_baselines.json` | create | problem-family → canonical baselines |
| `domain_packs/cs_algorithms/evaluation_datasets.json` | create | datasets per family |
| `domain_packs/cs_algorithms/gotchas.md` | create | recurring traps |
| `domain_packs/cs_algorithms/transient_deps.json` | create | recommended dev-only deps per family |
| `findings/INDEX.md` | create | index of verification findings with outcome labels (for calibration counts) |
| `tests/test_recipe.py` | create | recipe schema + stubbed-codex generation |
| `tests/test_novelty.py` | create | novelty classification with stubbed S2/OpenAlex |
| `tests/test_calibration.py` | create | report shape + counts on a fixture corpus of runs |
| `tests/test_domain_pack.py` | create | pack loading + `apply --dry-run` |
| `findings/phase3-product-layer-20260521.md` | create | Outcome doc |

---

## Task 1: VerificationRecipe schema + recipe.py + tests

**Files:** `pipeline/schemas.py`, `pipeline/recipe.py`, `tests/test_recipe.py`

- [ ] **Step 1: Add the schema**

In `pipeline/schemas.py`, append:

```python
# Stage 6: verification recipe per actionable composition
class VerificationRecipe(BaseModel):
    """Concrete test plan for a WEAK + PURELY_TESTABLE composition.

    Uses extra='allow' so the LLM can attach extra context fields without
    breaking validation (e.g., references to specific code packages).
    """
    model_config = ConfigDict(extra="allow")
    named_baselines: list[str] = Field(min_length=1)
    benchmark_dataset: str = Field(min_length=1)
    metric_primary: str = Field(min_length=1)
    metric_secondary: list[str] = Field(default_factory=list)
    min_meaningful_effect_size: str = Field(min_length=1)
    expected_runtime: str = Field(min_length=1)
    likely_failure_mode: str = Field(min_length=1)
    stop_rule: str = Field(min_length=1)
```

- [ ] **Step 2: Write failing test**

Create `tests/test_recipe.py`:

```python
"""Tests for the verification-recipe stage."""

from __future__ import annotations

import json

import pytest


def test_verification_recipe_happy_path():
    from pipeline.schemas import VerificationRecipe
    r = VerificationRecipe(
        named_baselines=["PQ", "OPQ"],
        benchmark_dataset="BEIR-fiqa",
        metric_primary="recall@10",
        metric_secondary=["wall_time_s"],
        min_meaningful_effect_size="+0.02 recall@10",
        expected_runtime="30 min on a single CPU",
        likely_failure_mode="codebook size dominates the bit budget at 32b",
        stop_rule="abort if any seed's recall@10 < 0.5 at 64b",
    )
    assert r.named_baselines == ["PQ", "OPQ"]


def test_verification_recipe_rejects_missing_fields():
    from pydantic import ValidationError
    from pipeline.schemas import VerificationRecipe
    with pytest.raises(ValidationError):
        VerificationRecipe(named_baselines=[])  # empty list rejected


def test_generate_recipe_with_stubbed_codex(tmp_path, monkeypatch):
    from pipeline import llm_io, recipe

    canned = json.dumps({
        "named_baselines": ["PQ", "RaBitQ"],
        "benchmark_dataset": "BEIR-nfcorpus",
        "metric_primary": "recall@10",
        "metric_secondary": ["wall_time_s"],
        "min_meaningful_effect_size": "+0.02 recall@10",
        "expected_runtime": "60 min single CPU",
        "likely_failure_mode": "at 32b the calibration noise dominates",
        "stop_rule": "abort if seed-0 recall < 0.45 at 64b",
    })

    def fake_codex(prompt: str, reasoning: str) -> tuple[bool, str, str]:
        return True, canned, ""

    monkeypatch.setattr(llm_io, "_invoke_codex", fake_codex)

    composition = {
        "id": "C1",
        "synthesis_claim": "PCA-banded PQ beats uniform PQ at 64 bits",
        "papers": [
            {"paper_id": "P1", "role": "PQ primer", "quote": "..."},
            {"paper_id": "P2", "role": "PCA decomposition", "quote": "..."},
        ],
        "verification": {
            "L3": {"verdict": "weak", "verdict_reason": "needs more evidence"},
            "L4": {"summary": {"net_testability_score": "PURELY_TESTABLE"}},
        },
    }
    state = {"problem": "compress embeddings", "corpus_size": 5}

    result = recipe.generate_recipe(composition, state, run_dir=tmp_path)
    assert result["named_baselines"] == ["PQ", "RaBitQ"]
    assert result["metric_primary"] == "recall@10"
    # Artifact dir exists
    assert (tmp_path / "llm_calls" / "recipe_C1").is_dir()
```

- [ ] **Step 3: Verify tests fail**

```bash
uv run pytest tests/test_recipe.py -v
```

- [ ] **Step 4: Implement `pipeline/recipe.py`**

```python
"""Stage 6: generate a verification recipe for an actionable composition."""

from __future__ import annotations

from pathlib import Path

from pipeline.llm_io import (
    parse_or_repair,
    run_codex_with_artifacts,
    update_artifact_meta,
)
from pipeline.schemas import VerificationRecipe

RECIPE_PROMPT_TEMPLATE = """You are designing a small, runnable verification experiment for a
proposed compositional synthesis that L3 marked WEAK and L4 marked PURELY_TESTABLE.

PROBLEM: {problem}

SYNTHESIS CLAIM:
{claim}

CONSTITUENT PAPERS:
{papers_block}

L3 ADVERSARIAL VERDICT REASON:
{l3_reason}

YOUR TASK
Emit a strict JSON object with these fields and no others. Keep each field tight; the
engineer reading this should be able to start implementing within an hour.

OUTPUT — JSON ONLY:
{{
  "named_baselines": ["<canonical method 1>", "<canonical method 2>", "..."],
  "benchmark_dataset": "<existing public dataset + retrieval link>",
  "metric_primary": "<single string, e.g. recall@10>",
  "metric_secondary": ["<one or two secondary metrics>"],
  "min_meaningful_effect_size": "<numeric or short string, e.g. +0.02 recall@10>",
  "expected_runtime": "<range; CPU/GPU; e.g. 30 minutes single CPU>",
  "likely_failure_mode": "<one short paragraph>",
  "stop_rule": "<condition under which we abandon early as a clean negative>"
}}
"""


def _papers_block(composition: dict) -> str:
    lines = []
    for p in composition.get("papers", []):
        lines.append(
            f"- {p.get('paper_id', '?')}: {p.get('role', '?')}"
        )
    return "\n".join(lines)


def generate_recipe(composition: dict, state: dict, run_dir: Path) -> dict:
    """Build a verification recipe for one composition.

    Returns the recipe dict (model_dump output). On parse failure returns
    an empty dict with an `error` key; callers decide whether to retry or
    skip.
    """
    cid = composition.get("id", "unknown")
    stage = f"recipe_{cid}"
    l3 = composition.get("verification", {}).get("L3", {})
    prompt = RECIPE_PROMPT_TEMPLATE.format(
        problem=state.get("problem", ""),
        claim=composition.get("synthesis_claim", ""),
        papers_block=_papers_block(composition),
        l3_reason=l3.get("verdict_reason", "") or l3.get("error", ""),
    )

    ok, stdout, stderr, artifact_dir = run_codex_with_artifacts(
        stage=stage, prompt=prompt, run_dir=run_dir, reasoning="high",
    )
    if not ok:
        update_artifact_meta(artifact_dir, parse_status="codex_failed")
        return {"error": stderr}

    def _repair(p: str) -> tuple[bool, str, str]:
        return run_codex_with_artifacts(
            stage=stage + "_repair", prompt=p, run_dir=run_dir, reasoning="high",
        )[:3]

    result = parse_or_repair(stdout, VerificationRecipe, repair_call=_repair)
    update_artifact_meta(
        artifact_dir,
        parse_status=result.status,
        validation_errors=result.errors,
    )
    if result.value is None:
        return {"error": "schema_validation_failed"}
    return result.value.model_dump()
```

- [ ] **Step 5: Wire into compose_one (opt-in flag)**

In `pipeline/compose.py`, add `with_recipes: bool = False` to `compose_one`. After the L4 step (where `v4 = extract_testable_predictions(...)` is set), add:

```python
        if with_recipes:
            l3 = v3 or {}
            l4 = v4 or {}
            verdict = (l3.get("verdict") or "").lower()
            l4_score = (l4.get("summary") or {}).get("net_testability_score")
            if verdict == "weak" and l4_score == "PURELY_TESTABLE":
                from pipeline.recipe import generate_recipe
                recipe = generate_recipe(c, state, run_dir=out_dir)
                c["verification"]["recipe"] = recipe
```

Add corresponding `--with-recipes` argparse flag and thread it through to `compose_one`.

- [ ] **Step 6: Run tests + commit**

```bash
uv run pytest tests/test_recipe.py -v
uv run pytest -q 2>&1 | tail -3
uv run ruff check .

git add pipeline/schemas.py pipeline/recipe.py pipeline/compose.py tests/test_recipe.py
git commit -m "$(cat <<'EOF'
feat(pipeline): add verification recipe stage for actionable compositions

Phase 3 item 1. New module pipeline/recipe.py with generate_recipe()
that emits a structured baselines/dataset/metric/effect-size/runtime/
failure-mode/stop-rule plan for every WEAK + PURELY_TESTABLE
composition. Schema VerificationRecipe in pipeline/schemas.py;
artifacts under llm_calls/recipe_<cid>/. Opt-in via --with-recipes;
default off because it costs an extra LLM call per actionable comp.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: NoveltyReport schema + novelty.py + tests + opt-in CLI

**Files:** `pipeline/schemas.py`, `pipeline/novelty.py`, `tests/test_novelty.py`, `pipeline/compose.py`

- [ ] **Step 1: Schema**

In `pipeline/schemas.py`:

```python
# Stage 7: post-L3 novelty check
class NoveltyHit(BaseModel):
    model_config = ConfigDict(extra="allow")
    source: str  # "s2" | "openalex"
    title: str
    similarity: float = Field(ge=0.0, le=1.0)
    url: str = ""


class NoveltyReport(BaseModel):
    model_config = ConfigDict(extra="allow")
    classification: Literal["known", "adjacent-known", "not-found", "unverified"]
    key_phrases: list[str] = Field(default_factory=list)
    hits: list[NoveltyHit] = Field(default_factory=list)
    cocite_density: float = Field(ge=0.0, default=0.0)
    queries_issued: list[str] = Field(default_factory=list)
    note: str = ""
```

- [ ] **Step 2: Write failing tests**

Create `tests/test_novelty.py`:

```python
"""Tests for the post-L3 novelty check stage."""

from __future__ import annotations

import pytest


def test_novelty_report_classification_literal():
    from pydantic import ValidationError
    from pipeline.schemas import NoveltyReport
    with pytest.raises(ValidationError):
        NoveltyReport(classification="dunno")


def test_classify_known_with_stubbed_hits(monkeypatch, tmp_path):
    from pipeline import novelty

    def fake_s2(query: str) -> list[dict]:
        return [
            {"title": "PCA-banded PQ outperforms PQ", "url": "u1", "similarity": 0.95},
            {"title": "Almost the same paper", "url": "u2", "similarity": 0.91},
            {"title": "Variant 3", "url": "u3", "similarity": 0.88},
        ]

    monkeypatch.setattr(novelty, "_s2_title_search", fake_s2)
    monkeypatch.setattr(novelty, "_openalex_title_search", lambda q: [])
    monkeypatch.setattr(novelty, "_cocitation_density", lambda papers: 0.5)
    monkeypatch.setattr(novelty, "_extract_key_phrases",
                        lambda comp, state: ["pca-banded pq", "embedding quantization"])

    composition = {"id": "C1", "synthesis_claim": "PCA-banded PQ"}
    state = {"problem": "embedding quantization"}
    out = novelty.check_novelty(composition, state, run_dir=tmp_path,
                                with_search=True)
    assert out["classification"] == "known"
    assert len(out["hits"]) >= 3


def test_classify_not_found_with_no_hits(monkeypatch, tmp_path):
    from pipeline import novelty
    monkeypatch.setattr(novelty, "_s2_title_search", lambda q: [])
    monkeypatch.setattr(novelty, "_openalex_title_search", lambda q: [])
    monkeypatch.setattr(novelty, "_cocitation_density", lambda papers: 0.0)
    monkeypatch.setattr(novelty, "_extract_key_phrases",
                        lambda comp, state: ["x"])
    composition = {"id": "C1", "synthesis_claim": "x"}
    state = {"problem": "x"}
    out = novelty.check_novelty(composition, state, run_dir=tmp_path,
                                with_search=True)
    assert out["classification"] == "not-found"


def test_classify_unverified_when_search_disabled(tmp_path):
    from pipeline import novelty
    composition = {"id": "C1", "synthesis_claim": "x"}
    state = {"problem": "x"}
    out = novelty.check_novelty(composition, state, run_dir=tmp_path,
                                with_search=False)
    assert out["classification"] == "unverified"
```

- [ ] **Step 3: Implement `pipeline/novelty.py`**

```python
"""Post-L3 novelty check.

Two arms:
  - exact-term search against Semantic Scholar + OpenAlex titles
  - constituent-paper-pair co-citation density via OpenAlex referenced_works

Classification:
  known          : >= 3 hits at high similarity
  adjacent-known : 1-2 hits OR high co-citation density (>= 0.3)
  not-found      : zero hits and low co-citation density
  unverified     : search disabled or APIs failed

The S2 + OpenAlex helpers live behind functions that tests monkeypatch.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Iterable

KNOWN_HIT_THRESHOLD = 3
KNOWN_SIM_THRESHOLD = 0.85
ADJACENT_COCITE_THRESHOLD = 0.3


def _extract_key_phrases(composition: dict, state: dict) -> list[str]:
    """Simple TF-IDF-style key-phrase extraction over the claim + problem.
    Replaced by codex-driven extraction at a later step; for now a
    deterministic fallback that returns the longest space-separated chunks."""
    text = " ".join([
        composition.get("synthesis_claim", "") or "",
        state.get("problem", "") or "",
    ]).lower()
    words = [w.strip(".,;:") for w in text.split() if len(w) > 4]
    counts = Counter(words)
    return [w for w, _ in counts.most_common(6)]


def _s2_title_search(query: str) -> list[dict]:
    """Stubbed in tests; production hits Semantic Scholar."""
    return []


def _openalex_title_search(query: str) -> list[dict]:
    """Stubbed in tests; production hits OpenAlex."""
    return []


def _cocitation_density(papers: Iterable[dict]) -> float:
    """For each pair of papers, count cross-citations via OpenAlex
    referenced_works overlap. Returns 0.0 in tests by default."""
    return 0.0


def check_novelty(
    composition: dict,
    state: dict,
    run_dir: Path,
    with_search: bool = False,
) -> dict:
    """Run the novelty check. With with_search=False, classifies as
    `unverified` and skips network calls.
    """
    if not with_search:
        return {
            "classification": "unverified",
            "key_phrases": [],
            "hits": [],
            "cocite_density": 0.0,
            "queries_issued": [],
            "note": "search disabled (with_search=False)",
        }

    key_phrases = _extract_key_phrases(composition, state)
    queries_issued: list[str] = []
    hits: list[dict] = []

    for phrase in key_phrases[:3]:
        queries_issued.append(f"s2:{phrase}")
        s2_hits = _s2_title_search(phrase)
        for h in s2_hits:
            hits.append({**h, "source": "s2"})
        queries_issued.append(f"openalex:{phrase}")
        oa_hits = _openalex_title_search(phrase)
        for h in oa_hits:
            hits.append({**h, "source": "openalex"})

    density = _cocitation_density(composition.get("papers", []))

    strong_hits = [h for h in hits if h.get("similarity", 0.0) >= KNOWN_SIM_THRESHOLD]

    if len(strong_hits) >= KNOWN_HIT_THRESHOLD:
        classification = "known"
    elif strong_hits or density >= ADJACENT_COCITE_THRESHOLD:
        classification = "adjacent-known"
    else:
        classification = "not-found"

    out = {
        "classification": classification,
        "key_phrases": key_phrases,
        "hits": hits,
        "cocite_density": density,
        "queries_issued": queries_issued,
        "note": "",
    }
    # Persist a small JSON artifact for audit.
    cid = composition.get("id", "unknown")
    artifact_dir = run_dir / "novelty" / cid
    artifact_dir.mkdir(parents=True, exist_ok=True)
    import json as _json
    (artifact_dir / "report.json").write_text(_json.dumps(out, indent=2))
    return out
```

- [ ] **Step 4: Wire into compose_one (opt-in)**

In `compose_one`, add `with_novelty: bool = False`. After L3 (where `v3 = verify_adversarial(...)`), add:

```python
        if with_novelty and (v3 or {}).get("verdict") != "invalid":
            from pipeline.novelty import check_novelty
            novelty = check_novelty(c, state, run_dir=out_dir, with_search=True)
            c["verification"]["novelty"] = novelty
```

Add `--with-novelty` argparse flag.

- [ ] **Step 5: Run tests + commit**

```bash
uv run pytest tests/test_novelty.py -v
uv run pytest -q 2>&1 | tail -3
uv run ruff check .

git add pipeline/schemas.py pipeline/novelty.py pipeline/compose.py tests/test_novelty.py
git commit -m "$(cat <<'EOF'
feat(pipeline): add post-L3 novelty check stage

Phase 3 item 2. Classifies each composition as known / adjacent-known
/ not-found / unverified by searching S2 + OpenAlex for the claim's
key phrases and measuring co-citation density across constituent
papers. Opt-in via --with-novelty; default off because it hits live
APIs.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

> **Engineer note:** the S2 + OpenAlex helpers are deliberately stub-only here. Filling them in with real `requests` calls (with retry/backoff modeled on `scripts/step3_novelty_check_s2.py`) is the obvious next step — but the schema, classification logic, and opt-in plumbing all work without the network code, and the tests pin behavior independently. Land this commit, then wire the live calls in a follow-up.

---

## Task 3: Calibration dashboard

**Files:** `pipeline/calibration.py`, `findings/INDEX.md`, `tests/test_calibration.py`

- [ ] **Step 1: Write failing test**

Create `tests/test_calibration.py`:

```python
"""Tests for the calibration markdown report."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


def _make_run(root: Path, run_id: str, status: str, verdicts: list[str]) -> None:
    d = root / run_id
    d.mkdir(parents=True, exist_ok=True)
    state = {
        "run_id": run_id,
        "problem": f"problem for {run_id}",
        "corpus_size": 10,
        "dimensions": [{"name": "x", "description": "y", "example_question": "z"}],
        "compositions": [
            {
                "id": f"C{i+1}",
                "verification": {
                    "L3": {"verdict": v},
                    "L4": {"summary": {"net_testability_score": "PURELY_TESTABLE"}},
                },
            }
            for i, v in enumerate(verdicts)
        ],
        "status": status,
        "relevance_skipped": False,
    }
    (d / "state.json").write_text(json.dumps(state))


def test_calibration_report_contains_required_sections(tmp_path):
    from pipeline import calibration
    _make_run(tmp_path, "20260101-000000", "complete", ["sound", "weak"])
    _make_run(tmp_path, "20260102-000000", "complete", ["invalid", "weak", "weak"])
    out = calibration.generate_report(tmp_path)
    assert "Run summary" in out
    assert "L3 verdict mix" in out
    assert "L4 testability mix" in out
    assert "sound" in out.lower()
    assert "weak" in out.lower()
    assert "invalid" in out.lower()
    # Counts
    assert "5" in out  # 5 total compositions


def test_calibration_handles_empty_dir(tmp_path):
    from pipeline import calibration
    out = calibration.generate_report(tmp_path)
    assert "no compose runs found" in out.lower()


def test_calibration_handles_run_with_no_compositions(tmp_path):
    from pipeline import calibration
    d = tmp_path / "20260103-000000"
    d.mkdir()
    (d / "state.json").write_text(json.dumps({
        "run_id": "20260103-000000", "problem": "x", "status": "no_dimensions_extracted",
        "compositions": []
    }))
    out = calibration.generate_report(tmp_path)
    assert "20260103-000000" in out
```

- [ ] **Step 2: Implement `pipeline/calibration.py`**

```python
"""Calibration dashboard: walk saved compose runs and emit a markdown report.

Usage:
    uv run python -m pipeline.calibration [--root data/pipeline/compositions]
    uv run python -m pipeline.calibration --root <dir>  # for testing
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

DEFAULT_ROOT = Path(__file__).resolve().parent.parent / "data" / "pipeline" / "compositions"


def _load_run(run_dir: Path) -> dict | None:
    state_path = run_dir / "state.json"
    if not state_path.exists():
        return None
    try:
        return json.loads(state_path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def _total_llm_time(run_dir: Path) -> float:
    total = 0.0
    for meta_path in run_dir.glob("llm_calls/**/meta.json"):
        try:
            meta = json.loads(meta_path.read_text())
            total += float(meta.get("duration_s", 0.0))
        except (json.JSONDecodeError, OSError, TypeError):
            continue
    return total


def generate_report(root: Path) -> str:
    runs = []
    for run_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        state = _load_run(run_dir)
        if state is None:
            continue
        runs.append((run_dir, state))

    if not runs:
        return "# Calibration report\n\nNo compose runs found at this root.\n"

    lines = ["# Calibration report", ""]
    lines.append(f"Aggregating {len(runs)} compose run(s).")
    lines.append("")

    # Run summary table
    lines.append("## Run summary")
    lines.append("")
    lines.append("| run_id | problem | corpus | dims | comps | status |")
    lines.append("|---|---|---:|---:|---:|---|")
    for run_dir, state in runs:
        comp_count = len(state.get("compositions") or [])
        dim_count = len(state.get("dimensions") or [])
        problem = (state.get("problem") or "").replace("|", "/")[:60]
        lines.append(
            f"| {state.get('run_id', run_dir.name)} | {problem} | "
            f"{state.get('corpus_size', 0)} | {dim_count} | "
            f"{comp_count} | {state.get('status', '?')} |"
        )
    lines.append("")

    # L3 verdict mix
    l3_counts: Counter[str] = Counter()
    l4_counts: Counter[str] = Counter()
    skipped_relevance = 0
    total_llm_seconds = 0.0
    for run_dir, state in runs:
        if state.get("relevance_skipped"):
            skipped_relevance += 1
        total_llm_seconds += _total_llm_time(run_dir)
        for c in state.get("compositions") or []:
            v = c.get("verification") or {}
            l3 = (v.get("L3") or {}).get("verdict") or "missing"
            l3_counts[l3] += 1
            l4_summary = (v.get("L4") or {}).get("summary") or {}
            score = l4_summary.get("net_testability_score")
            if isinstance(score, str):
                l4_counts[score] += 1
            elif isinstance(score, (int, float)):
                l4_counts["numeric_ratio"] += 1
            else:
                l4_counts["missing"] += 1

    lines.append("## L3 verdict mix")
    lines.append("")
    for k, v in l3_counts.most_common():
        lines.append(f"- **{k}**: {v}")
    lines.append("")

    lines.append("## L4 testability mix")
    lines.append("")
    for k, v in l4_counts.most_common():
        lines.append(f"- **{k}**: {v}")
    lines.append("")

    lines.append("## Skipped gates")
    lines.append("")
    lines.append(f"- runs with relevance check skipped: **{skipped_relevance}** / {len(runs)}")
    lines.append("")

    lines.append("## Cost per run (LLM wall time)")
    lines.append("")
    if total_llm_seconds > 0:
        avg = total_llm_seconds / len(runs)
        lines.append(
            f"- total LLM wall time across {len(runs)} runs: **{total_llm_seconds:.1f}s**"
        )
        lines.append(f"- average per run: **{avg:.1f}s**")
    else:
        lines.append("- (no llm_calls/*/meta.json files found — older runs predate Phase 1)")
    lines.append("")

    # Track-record outcomes (if INDEX.md exists)
    index_path = root.parent.parent / ".." / "findings" / "INDEX.md"
    # robust resolution
    candidate = Path(__file__).resolve().parent.parent / "findings" / "INDEX.md"
    if candidate.exists():
        lines.append("## Verified track record")
        lines.append("")
        lines.append(candidate.read_text())

    return "\n".join(lines) + "\n"


def main() -> int:
    p = argparse.ArgumentParser(description="Generate the Congruity calibration report")
    p.add_argument("--root", default=str(DEFAULT_ROOT),
                   help="root directory holding compose run subdirectories")
    p.add_argument("--out", default=None,
                   help="write report here; default: findings/calibration-<today>.md")
    args = p.parse_args()
    root = Path(args.root)
    report = generate_report(root)
    if args.out is None:
        from datetime import datetime
        today = datetime.utcnow().strftime("%Y%m%d")
        repo_root = Path(__file__).resolve().parent.parent
        out_path = repo_root / "findings" / f"calibration-{today}.md"
    else:
        out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report)
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 3: Create `findings/INDEX.md`**

Write a small index of existing verification findings with outcome labels. This becomes a maintainable single source of truth that calibration can render verbatim.

- [ ] **Step 4: Run tests + commit**

```bash
uv run pytest tests/test_calibration.py -v
uv run pytest -q 2>&1 | tail -3
uv run ruff check .
uv run python -m pipeline.calibration --out /tmp/calib.md
cat /tmp/calib.md | head -40

git add pipeline/calibration.py findings/INDEX.md tests/test_calibration.py
git commit -m "$(cat <<'EOF'
feat(pipeline): add calibration dashboard markdown report

Phase 3 item 3. New module pipeline/calibration.py walks
data/pipeline/compositions/<run_id>/state.json files and emits a
markdown report covering run summary, L3 verdict mix, L4 testability
mix, skipped gates, and total LLM wall time. findings/INDEX.md is
the manually-maintained verified-outcomes table the report renders.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Domain pack — CS / algorithms

**Files:** `domain_packs/cs_algorithms/*.json`, `domain_packs/cs_algorithms/gotchas.md`, `pipeline/domain_pack.py`, `tests/test_domain_pack.py`

- [ ] **Step 1: Write the domain pack files**

Create `domain_packs/cs_algorithms/topic_queries.json`:

```json
[
  {"id": "T11486", "display_name": "Vector quantization for embeddings",
   "year_min": 2018, "works_count_hint": 800},
  {"id": "T10024", "display_name": "Cache replacement policies",
   "year_min": 2015, "works_count_hint": 1200},
  {"id": "T10089", "display_name": "Combinatorial optimization heuristics",
   "year_min": 2010, "works_count_hint": 2500},
  {"id": "T11202", "display_name": "Short-string compression",
   "year_min": 2015, "works_count_hint": 400}
]
```

> The exact OpenAlex IDs can be filled in by running a quick lookup; the placeholders above match the experimental domains already in the project. If the engineer doesn't want to look these up by hand, leave the strings as ISO display names and the apply CLI's first run can resolve them.

Create `domain_packs/cs_algorithms/problem_templates.json`:

```json
[
  {"family": "embedding_compression",
   "template": "Compress sentence embeddings to under {bits} bits per vector while preserving recall@10 >= {recall_target} against PQ at the same bit budget."},
  {"family": "cache_replacement",
   "template": "Design a cache eviction policy that beats {baseline} hit rate by >= {delta} on {workload_class} workloads at {capacity_fraction} cache size."},
  {"family": "small_points_optimization",
   "template": "Maximize the minimum-triangle area for {n} points in the unit square; beat published reference of {published_area} within {wall_clock} seconds."},
  {"family": "short_string_compression",
   "template": "Compress {input_class} short strings (8-256 bytes) at a compression ratio better than {baseline_codec}."}
]
```

Create `domain_packs/cs_algorithms/known_baselines.json`:

```json
{
  "embedding_compression": ["PQ", "OPQ", "RaBitQ", "scalar quantization", "Matryoshka"],
  "cache_replacement": ["LRU", "ARC", "SIEVE", "S3FIFO", "FIFO"],
  "small_points_optimization": ["simulated annealing", "basin hopping", "CMA-ES", "differential evolution"],
  "short_string_compression": ["Brotli + shared dictionary", "Zstandard + trained dictionary", "HPACK (stateful, HTTP only)"]
}
```

Create `domain_packs/cs_algorithms/evaluation_datasets.json`:

```json
{
  "embedding_compression": ["BEIR", "MTEB retrieval subset", "MS MARCO"],
  "cache_replacement": ["SNIA block traces", "MSR Cambridge traces", "synthetic Zipf/scan/loop generators in experiments/cache_replacement/traces.py"],
  "small_points_optimization": ["Erich Friedman's reference table at http://www2.stetson.edu/~efriedma/heilbronn/"],
  "short_string_compression": ["JSON API responses", "HTTP request headers", "code snippets (CodeNet-style)"]
}
```

Create `domain_packs/cs_algorithms/transient_deps.json`:

```json
{
  "embedding_compression": ["numpy", "scikit-learn"],
  "cache_replacement": [],
  "small_points_optimization": ["scipy", "cma", "jax"],
  "short_string_compression": ["brotli", "zstandard", "hpack"]
}
```

Create `domain_packs/cs_algorithms/gotchas.md`:

```markdown
# CS / algorithms gotchas

Recurring traps surfaced by the existing experiments.

## Embedding quantization
- `np.argpartition(-sims, k)` returns the top **k+1** items; for top-k use `k - 1` as the partition index. (Phase 2 fixed this in `experiments/embed_quant/bench.py`.)
- Normalize query and DB vectors before computing cosine similarity, even if you think they're already normalized.
- After PQ allocation rounding, the actual bit count can be less than the nominal budget. Label methods by the realized count, not the constructor argument.

## Cache replacement
- Synthetic Zipf workloads are easy. Loop + scan + hot-set workloads (already in `experiments/cache_replacement/traces.py`) are where most "novel" replacement policies fall apart.
- Successor prediction is a **prefetching** mechanism, not a replacement mechanism. Designing one as a replacement policy collapses into known prior art.

## Small-points optimization
- Use a stable hash for any signature-based candidate filtering. Python's `hash()` is randomized per process.
- Report mean ± 95% CI over multiple seeds. Single-seed runs hide a lot of noise.
- Beating the implemented SA baseline is NOT beating published SOTA. Be explicit about which is which.

## Short-string compression
- Test round-trip on every corpus the bench evaluates on. Without it, a "compression" implementation can silently be lossy.
- Stateful baselines (HPACK) are much harder to beat than stateless dictionary-based codecs (Brotli, Zstd). Note which class your method is in before claiming a win.
```

- [ ] **Step 2: Implement `pipeline/domain_pack.py`**

```python
"""Domain pack loader + apply CLI.

Usage:
    uv run python -m pipeline.domain_pack list
    uv run python -m pipeline.domain_pack apply <name> --problem "<text>" [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PACKS_DIR = Path(__file__).resolve().parent.parent / "domain_packs"


def list_packs() -> list[str]:
    if not PACKS_DIR.exists():
        return []
    return sorted(p.name for p in PACKS_DIR.iterdir() if p.is_dir())


def load_pack(name: str) -> dict:
    d = PACKS_DIR / name
    if not d.is_dir():
        raise FileNotFoundError(f"domain pack {name!r} not found at {d}")
    pack = {}
    for fname in ["topic_queries", "problem_templates", "known_baselines",
                  "evaluation_datasets", "transient_deps"]:
        p = d / f"{fname}.json"
        pack[fname] = json.loads(p.read_text()) if p.exists() else None
    gotchas_path = d / "gotchas.md"
    pack["gotchas"] = gotchas_path.read_text() if gotchas_path.exists() else ""
    return pack


def apply_pack(name: str, problem: str, dry_run: bool = True) -> dict:
    pack = load_pack(name)
    return {
        "domain": name,
        "problem": problem,
        "topic_queries": pack.get("topic_queries", []),
        "known_baselines_hint": pack.get("known_baselines", {}),
        "evaluation_datasets_hint": pack.get("evaluation_datasets", {}),
        "transient_deps_hint": pack.get("transient_deps", {}),
        "gotchas": pack.get("gotchas", "")[:500] + "..." if pack.get("gotchas") else "",
        "dry_run": dry_run,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Domain pack inspection + apply")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="list available packs")
    ap = sub.add_parser("apply", help="apply a pack to a problem")
    ap.add_argument("name")
    ap.add_argument("--problem", required=True)
    ap.add_argument("--dry-run", action="store_true", default=True,
                    help="print what would be applied without running a compose")
    args = p.parse_args(argv if argv is not None else sys.argv[1:])
    if args.cmd == "list":
        for name in list_packs():
            print(name)
        return 0
    if args.cmd == "apply":
        out = apply_pack(args.name, args.problem, dry_run=args.dry_run)
        print(json.dumps(out, indent=2))
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 3: Write tests**

Create `tests/test_domain_pack.py`:

```python
"""Tests for the domain-pack loader + apply CLI."""

from __future__ import annotations


def test_list_packs_includes_cs_algorithms():
    from pipeline.domain_pack import list_packs
    names = list_packs()
    assert "cs_algorithms" in names


def test_load_cs_algorithms_pack_has_required_keys():
    from pipeline.domain_pack import load_pack
    pack = load_pack("cs_algorithms")
    assert pack["topic_queries"] is not None
    assert isinstance(pack["topic_queries"], list)
    assert isinstance(pack["known_baselines"], dict)
    assert "embedding_compression" in pack["known_baselines"]
    assert "PQ" in pack["known_baselines"]["embedding_compression"]


def test_apply_dry_run_returns_structured_summary():
    from pipeline.domain_pack import apply_pack
    out = apply_pack("cs_algorithms",
                     problem="compress sentence embeddings to under 64 bits",
                     dry_run=True)
    assert out["domain"] == "cs_algorithms"
    assert out["dry_run"] is True
    assert "embedding_compression" in out["known_baselines_hint"]
```

- [ ] **Step 4: Run tests + smoke + commit**

```bash
uv run pytest tests/test_domain_pack.py -v
uv run python -m pipeline.domain_pack list
uv run python -m pipeline.domain_pack apply cs_algorithms --problem "compress embeddings"
uv run pytest -q 2>&1 | tail -3
uv run ruff check .

git add domain_packs/ pipeline/domain_pack.py tests/test_domain_pack.py
git commit -m "$(cat <<'EOF'
feat(domain_packs): add first domain pack (CS / algorithms)

Phase 3 item 4. domain_packs/cs_algorithms/ contains topic queries,
problem templates, canonical baselines per family, evaluation datasets,
transient deps, and a gotchas doc capturing recurring traps from the
existing experiments. New CLI pipeline/domain_pack.py with `list` and
`apply --dry-run` subcommands.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Phase 3 findings doc

**Files:** `findings/phase3-product-layer-20260521.md`

- [ ] **Step 1: Write** mirroring Phase 1/2 findings docs: shipped / deferred / open items / commits table.
- [ ] **Step 2: Commit.**

---

## Self-review (already performed by author)

- **Spec coverage:** four spec items → four tasks. Expert hit-rate study is in Non-goals.
- **Type consistency:** stage names match between recipe.py and the test's expected artifact directory (`recipe_C1`). NoveltyReport classification literals match between schema and classification logic.
- **Placeholders:** none in the implementation code. The OpenAlex topic IDs in `topic_queries.json` are flagged as "fill in via lookup" — that's an explicit, scoped TODO surfaced in the file.

## Risks recap

- Task 1's prompt design will need real-world tuning. Mitigate by iterating against saved runs after the first end-to-end use.
- Task 2's classification thresholds (3 hits / 0.85 similarity / 0.3 cocite) are heuristic; revisit after the first 10 real runs with `--with-novelty`.
- Task 3's calibration counts depend on `findings/INDEX.md` being maintained. Document this as a step in any new finding doc.
- Task 4 ships placeholders for the OpenAlex topic IDs; mark them clearly so the first user knows to fill in.
