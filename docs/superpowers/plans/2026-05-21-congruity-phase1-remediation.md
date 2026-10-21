# Congruity Phase 1 — Correctness & Auditability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land the five Phase 1 items from [docs/superpowers/specs/2026-05-21-congruity-phase1-remediation.md](../specs/2026-05-21-congruity-phase1-remediation.md): green lint, dual-write corpus with stable paper UIDs, per-stage LLM artifacts, Pydantic schema validation with one-shot repair, and a strict relevance gate.

**Architecture:** Two new modules — `pipeline/identity.py` (paper_uid) and `pipeline/llm_io.py` (`run_codex_with_artifacts` + `parse_or_repair`) — keep new code isolated. `pipeline/schemas.py` holds Pydantic v2 models for every LLM-emitted shape. `pipeline/corpus_gate.py` gains a `strict` parameter; `pipeline/compose.py` is the integration site for all wiring.

**Tech Stack:** Python 3.11, pytest, ruff, pydantic v2 (new), existing `requests`. Codex CLI invoked via subprocess (unchanged).

---

## File structure

| Path | Action | Responsibility |
|---|---|---|
| `pyproject.toml` | modify | Add `pydantic>=2` dep |
| `experiments/embed_quant/bench.py` | modify | Remove `exact`, `calib_db` dead assignments |
| `pipeline/pair_filter.py` | modify | Remove `d` dead assignment |
| `scripts/step3_kvcache_experiment.py`, `scripts/step3_novelty_check*.py`, `pipeline/discover.py`, `pipeline/_openalex.py` | modify | Auto-fix via `ruff --fix` |
| `pipeline/identity.py` | create | `paper_uid(paper) -> str` |
| `pipeline/llm_io.py` | create | `run_codex_with_artifacts`, `parse_or_repair` |
| `pipeline/schemas.py` | create | Pydantic models: `Composition`, `CompositionPaper`, `Dimension`, `PaperScore`, `L1Item`, `L3Verdict`, `L4Verdict` |
| `pipeline/corpus_gate.py` | modify | Add `strict` parameter to `validate_corpus`; honor in abort flow |
| `pipeline/compose.py` | modify | Add `paper_uid` on assembly, dual-write `corpus_raw.json` + `corpus_validated.json`, switch reprocess loader, migrate 7 `run_codex` call sites to wrapper, wire schemas, add `--strict-gate / --no-strict-gate` flag, propagate strictness to `validate_corpus` |
| `pipeline/render.py` *(if exists)* or wherever digest is rendered | modify | Add "⚠️ relevance check skipped" badge |
| `tests/test_identity.py` | create | paper_uid stability |
| `tests/test_corpus_persistence.py` | create | corpus_raw/validated.json layout |
| `tests/test_schemas.py` | create | Pydantic model happy/sad paths |
| `tests/test_llm_io.py` | create | parse_or_repair + run_codex_with_artifacts |
| `tests/test_corpus_gate.py` | modify | Add strict-mode tests |
| `findings/phase1-remediation-20260521.md` | create | Track-record update describing what changed |

---

## Task 1: Ruff hygiene

**Files:**
- Modify (auto-fix): `pipeline/_openalex.py`, `pipeline/discover.py`, `pipeline/pair_filter.py`, `scripts/step3_*.py`
- Modify (manual): `experiments/embed_quant/bench.py:47`, `experiments/embed_quant/bench.py:51`, `pipeline/pair_filter.py` (the `d` F841)

- [ ] **Step 1: Run auto-fix**

```bash
uv run ruff check . --fix
```

Expected: "Found 20 errors (17 fixed, 3 remaining)." or similar.

- [ ] **Step 2: Verify remaining issues are the expected 3 F841s**

```bash
uv run ruff check .
```

Expected: 3 errors, all F841, at `experiments/embed_quant/bench.py:47`, `:51`, and one in `pipeline/pair_filter.py`.

- [ ] **Step 3: Manually remove the 3 dead assignments**

In `experiments/embed_quant/bench.py`, delete the line `exact = topk_cosine(queries, db, k=K)` (and its preceding `print("computing exact top-k (brute force)...", flush=True)` if and only if the print refers solely to that computation — confirm by reading the surrounding 5 lines first; if the print is structural, keep it).

Delete `calib_db = db[:1000]` (and the comment immediately above it if it only describes that variable).

In `pipeline/pair_filter.py`, find the `d = pp["density"]` (or similar) assignment that's flagged and remove it.

- [ ] **Step 4: Verify green**

```bash
uv run ruff check .
uv run pytest
```

Expected: ruff `All checks passed!`, pytest `30 passed` (or higher if new tests have already been added by a parallel agent — adjust mentally).

- [ ] **Step 5: Commit**

```bash
git add -u
git commit -m "$(cat <<'EOF'
chore: clear all ruff failures (17 auto-fixed + 3 dead assignments)

Phase 1 item 1. Removes the unused exact/calib_db/d locals; Phase 2
will reintroduce exact and calib_db correctly when hardening the
embed_quant harness.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: paper_uid helper

**Files:**
- Create: `pipeline/identity.py`
- Test: `tests/test_identity.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_identity.py`:

```python
"""Tests for stable paper identifier generation."""

from __future__ import annotations

from pipeline.identity import paper_uid


def test_paper_uid_is_stable_for_same_openalex_id():
    a = {"openalex_id": "https://openalex.org/W123", "title": "X", "year": 2024}
    b = {"openalex_id": "https://openalex.org/W123", "title": "DIFFERENT", "year": 1999}
    # Different metadata but same openalex_id -> same uid (openalex wins)
    assert paper_uid(a) == paper_uid(b)


def test_paper_uid_differs_for_different_openalex_id():
    a = {"openalex_id": "https://openalex.org/W123", "title": "X", "year": 2024}
    b = {"openalex_id": "https://openalex.org/W456", "title": "X", "year": 2024}
    assert paper_uid(a) != paper_uid(b)


def test_paper_uid_falls_back_to_title_year_when_no_openalex():
    a = {"openalex_id": None, "title": "An attention mechanism", "year": 2023}
    b = {"title": "An attention mechanism", "year": 2023}  # missing key OK
    assert paper_uid(a) == paper_uid(b)


def test_paper_uid_distinguishes_title_year_pairs():
    a = {"title": "A", "year": 2023}
    b = {"title": "A", "year": 2024}
    assert paper_uid(a) != paper_uid(b)


def test_paper_uid_is_16_hex_chars():
    uid = paper_uid({"openalex_id": "https://openalex.org/W1"})
    assert len(uid) == 16
    assert all(c in "0123456789abcdef" for c in uid)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_identity.py -v
```

Expected: ImportError or ModuleNotFoundError on `pipeline.identity`.

- [ ] **Step 3: Implement `paper_uid`**

Create `pipeline/identity.py`:

```python
"""Stable paper identifiers.

paper_uid(paper) returns a 16-hex-char SHA-256 prefix derived from the
paper's most stable identifier. Used to cross-reference saved corpus
entries across re-indexing (P1..PN may change between runs; paper_uid
does not).
"""

from __future__ import annotations

import hashlib


def paper_uid(paper: dict) -> str:
    """Return a 16-hex-char stable identifier for a paper.

    Preference order:
        1. openalex_id (most authoritative when present)
        2. doi
        3. "title:" + title + ":" + year
    """
    key: str
    oid = paper.get("openalex_id")
    if oid:
        key = str(oid)
    elif paper.get("doi"):
        key = "doi:" + str(paper["doi"])
    else:
        title = str(paper.get("title", ""))
        year = str(paper.get("year", ""))
        key = f"title:{title}:{year}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
```

- [ ] **Step 4: Run test to verify it passes**

```bash
uv run pytest tests/test_identity.py -v
```

Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add pipeline/identity.py tests/test_identity.py
git commit -m "$(cat <<'EOF'
feat(pipeline): add paper_uid for stable cross-run paper identity

Phase 1 item 2 (foundation). 16-hex SHA-256 prefix; openalex_id wins,
falls back to doi, then "title:<t>:<year>".

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Dual-write corpus + reprocess loader

**Files:**
- Modify: `pipeline/compose.py:809–855` (assembly + post-gate reindex) and `:1074` (reprocess loader)
- Test: `tests/test_corpus_persistence.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_corpus_persistence.py`:

```python
"""Tests for corpus_raw.json + corpus_validated.json dual-write."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pipeline.compose import _persist_corpus_artifacts


def _make_paper(i: int, openalex_id: str | None = None) -> dict:
    return {
        "paper_id": f"P{i}",
        "openalex_id": openalex_id or f"https://openalex.org/W{i:04d}",
        "title": f"Paper {i}",
        "year": 2024,
        "abstract": "x" * 200,
        "source_topic": "test",
    }


def test_persist_corpus_artifacts_writes_raw_and_validated(tmp_path: Path):
    raw = [_make_paper(i) for i in range(1, 4)]
    dropped_ids = {"P2"}
    survivors = [p for p in raw if p["paper_id"] not in dropped_ids]
    _persist_corpus_artifacts(tmp_path, raw_corpus=raw, validated_corpus=survivors)

    raw_path = tmp_path / "corpus_raw.json"
    val_path = tmp_path / "corpus_validated.json"
    legacy_path = tmp_path / "corpus.json"

    assert raw_path.exists()
    assert val_path.exists()
    assert legacy_path.exists()

    raw_data = json.loads(raw_path.read_text())
    val_data = json.loads(val_path.read_text())
    legacy_data = json.loads(legacy_path.read_text())

    assert len(raw_data) == 3
    assert len(val_data) == 2
    assert legacy_data == val_data  # legacy mirrors validated

    # paper_uid is present on every paper
    assert all("paper_uid" in p for p in raw_data)
    assert all("paper_uid" in p for p in val_data)
    # paper_uid is stable across raw -> validated mapping (same paper, same uid)
    raw_uids_by_oid = {p["openalex_id"]: p["paper_uid"] for p in raw_data}
    for p in val_data:
        assert p["paper_uid"] == raw_uids_by_oid[p["openalex_id"]]
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_corpus_persistence.py -v
```

Expected: ImportError on `_persist_corpus_artifacts`.

- [ ] **Step 3: Add the helper to `pipeline/compose.py`**

Add this import near the top of `pipeline/compose.py` (with the other `from pipeline.*` imports):

```python
from pipeline.identity import paper_uid
```

Add this function in `pipeline/compose.py` just above `def compose_one(...)` (around line 779, before the orchestrator section):

```python
def _persist_corpus_artifacts(
    out_dir: Path,
    raw_corpus: list[dict],
    validated_corpus: list[dict],
) -> None:
    """Write corpus_raw.json (pre-gate) and corpus_validated.json (post-gate).

    Adds a paper_uid to every paper in-place. Also writes legacy
    corpus.json as a copy of corpus_validated.json for back-compat.
    """
    for p in raw_corpus:
        p.setdefault("paper_uid", paper_uid(p))
    for p in validated_corpus:
        p.setdefault("paper_uid", paper_uid(p))
    (out_dir / "corpus_raw.json").write_text(json.dumps(raw_corpus, indent=2))
    (out_dir / "corpus_validated.json").write_text(json.dumps(validated_corpus, indent=2))
    (out_dir / "corpus.json").write_text(json.dumps(validated_corpus, indent=2))
```

- [ ] **Step 4: Run test to verify it passes**

```bash
uv run pytest tests/test_corpus_persistence.py -v
```

Expected: 1 passed.

- [ ] **Step 5: Wire the helper into `compose_one`**

In `pipeline/compose.py`, locate the corpus assembly block (currently around lines 809–814):

```python
    # Re-index papers as P1..PN
    for i, p in enumerate(corpus, 1):
        p["paper_id"] = f"P{i}"
    papers_by_id = {p["paper_id"]: p for p in corpus}
    print(f"  corpus: {len(corpus)} papers from {len({p['source_topic'] for p in corpus})} topics")
    (out_dir / "corpus.json").write_text(json.dumps(corpus, indent=2))
```

Replace the final `(out_dir / "corpus.json").write_text(...)` line with:

```python
    # Snapshot the raw corpus now; validated/legacy copies are written
    # after the Step 1.5 gate has run (or here, in dry-run mode).
    raw_corpus_snapshot = [dict(p) for p in corpus]  # shallow copy preserves pre-drop state
```

Then, where the dry-run branch is (currently `if dry_run:` around line 824), insert a call to persist artifacts BEFORE writing `state.json`:

```python
    if dry_run:
        _persist_corpus_artifacts(out_dir, raw_corpus=raw_corpus_snapshot, validated_corpus=corpus)
        # Emit a fake compositions block so the digest renderer can shape itself
        state["status"] = "dry_run_complete"
        ...
```

Then, in the post-gate reindex block (currently around lines 850–859), after the survivors are reindexed, replace `(out_dir / "corpus.json").write_text(...)` (if any remains) and the existing reindex block to end with:

```python
        if report.drop_ids:
            corpus = [p for p in corpus if p["paper_id"] not in report.drop_ids]
            for i, p in enumerate(corpus, 1):
                p["paper_id"] = f"P{i}"
            papers_by_id = {p["paper_id"]: p for p in corpus}
            state["corpus_size"] = len(corpus)
            state["topics_covered"] = sorted({p["source_topic"] for p in corpus})
            print(f"  dropped {len(report.drop_ids)} papers; "
                  f"continuing with {len(corpus)}")
        state["corpus_gate"] = report.as_dict()
        _persist_corpus_artifacts(out_dir, raw_corpus=raw_corpus_snapshot, validated_corpus=corpus)
```

Also handle the `else` branch (`skip_gate`) — write the artifacts there too so legacy callers still get the files:

```python
    else:
        print("\nStep 1.5: corpus quality gate SKIPPED (--skip-gate)")
        _persist_corpus_artifacts(out_dir, raw_corpus=raw_corpus_snapshot, validated_corpus=corpus)
```

- [ ] **Step 6: Update the reprocess loader (compose.py:~1074)**

Locate:

```python
    corpus_path = run_dir / "corpus.json"
```

Replace with:

```python
    # Prefer the post-gate corpus; fall back to legacy single-file layout.
    corpus_path = run_dir / "corpus_validated.json"
    if not corpus_path.exists():
        corpus_path = run_dir / "corpus.json"
```

- [ ] **Step 7: Run all tests**

```bash
uv run pytest -v
```

Expected: all green (30 + new tests).

- [ ] **Step 8: Commit**

```bash
git add pipeline/compose.py tests/test_corpus_persistence.py
git commit -m "$(cat <<'EOF'
feat(compose): dual-write corpus_raw.json + corpus_validated.json

Phase 1 item 2. Adds paper_uid via pipeline.identity, snapshots the
pre-gate corpus, and writes both raw and validated artifacts after
Step 1.5. Reprocess loader prefers corpus_validated.json; corpus.json
remains as a backward-compat copy.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Add pydantic dependency

**Files:**
- Modify: `pyproject.toml`
- Modify: `uv.lock` (generated)

- [ ] **Step 1: Add the dep**

```bash
uv add 'pydantic>=2'
```

Expected: pyproject.toml's `[project] dependencies` now lists `pydantic>=2`; `uv.lock` updates.

- [ ] **Step 2: Verify import works**

```bash
uv run python -c "import pydantic; print(pydantic.VERSION)"
```

Expected: prints a 2.x version.

- [ ] **Step 3: Verify nothing broke**

```bash
uv run pytest
uv run ruff check .
```

Expected: both green.

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "$(cat <<'EOF'
chore: add pydantic v2 dependency

Foundation for Phase 1 item 4 (LLM output schema validation).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Pydantic schemas for LLM outputs

**Files:**
- Create: `pipeline/schemas.py`
- Test: `tests/test_schemas.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_schemas.py`:

```python
"""Tests for Pydantic models covering LLM stage outputs."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from pipeline.schemas import (
    Composition,
    CompositionPaper,
    Dimension,
    DimensionsOutput,
    L1Item,
    L3Verdict,
    L4Verdict,
    PaperScore,
    ScoresOutput,
)


def test_dimension_happy_path():
    d = Dimension(
        name="latency",
        description="end-to-end inference latency",
        example_question="how fast?",
    )
    assert d.name == "latency"


def test_dimension_missing_required_field_raises():
    with pytest.raises(ValidationError):
        Dimension(name="latency")  # missing description, example_question


def test_dimensions_output_wraps_list():
    out = DimensionsOutput(dimensions=[
        Dimension(name="a", description="b", example_question="c"),
    ])
    assert len(out.dimensions) == 1


def test_paper_score_happy_path():
    s = PaperScore(
        paper_id="P1",
        title_short="An attention mechanism",
        dimensions={"latency": {"score": 3, "contribution": "subquadratic"}},
    )
    assert s.paper_id == "P1"
    assert s.dimensions["latency"]["score"] == 3


def test_paper_score_rejects_score_above_5():
    with pytest.raises(ValidationError):
        PaperScore(
            paper_id="P1",
            title_short="x",
            dimensions={"latency": {"score": 7, "contribution": "x"}},
        )


def test_composition_happy_path():
    c = Composition(
        id="C1",
        synthesis_claim="A+B beats either alone",
        papers=[
            CompositionPaper(paper_id="P1", role="provides X", quote="..."),
            CompositionPaper(paper_id="P2", role="provides Y", quote="..."),
        ],
    )
    assert c.id == "C1"
    assert len(c.papers) == 2


def test_l3_verdict_happy_path():
    v = L3Verdict(verdict="WEAK", reasoning="not novel; similar to prior work")
    assert v.verdict == "WEAK"


def test_l3_verdict_rejects_unknown_verdict():
    with pytest.raises(ValidationError):
        L3Verdict(verdict="MAYBE", reasoning="x")


def test_l4_verdict_happy_path():
    v = L4Verdict(
        verdict="PURELY_TESTABLE",
        reasoning="all objections empirical",
    )
    assert v.verdict == "PURELY_TESTABLE"


def test_l1_item_happy_path():
    item = L1Item(consistent=True, issues=[])
    assert item.consistent
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_schemas.py -v
```

Expected: ImportError on `pipeline.schemas`.

- [ ] **Step 3: Implement schemas**

Create `pipeline/schemas.py`:

```python
"""Pydantic v2 models for every structured LLM output in the compose pipeline.

Field names and types track the existing prompt contracts in pipeline/compose.py.
Models reject unknown fields by default (extra='forbid') so silent schema drift
between prompt and parser is impossible.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class _StrictBase(BaseModel):
    model_config = ConfigDict(extra="forbid")


# Stage 2: dimensions
class Dimension(_StrictBase):
    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    example_question: str = Field(min_length=1)


class DimensionsOutput(_StrictBase):
    dimensions: list[Dimension]


# Stage 3: scores
class DimensionScore(_StrictBase):
    score: int = Field(ge=0, le=5)
    contribution: str = ""


class PaperScore(_StrictBase):
    paper_id: str = Field(min_length=1)
    title_short: str = ""
    dimensions: dict[str, DimensionScore]


class ScoresOutput(_StrictBase):
    scores: list[PaperScore]


# Stage 4: compositions
class CompositionPaper(_StrictBase):
    paper_id: str
    role: str
    quote: str = ""


class Composition(_StrictBase):
    id: str
    synthesis_claim: str
    papers: list[CompositionPaper]


class CompositionsOutput(_StrictBase):
    compositions: list[Composition]


# Stage 5 verification: L1 internal consistency
class L1Item(_StrictBase):
    consistent: bool
    issues: list[str] = []


# Stage 5 verification: L3 adversarial cross-check
class L3Verdict(_StrictBase):
    verdict: Literal["PASS", "WEAK", "INVALID"]
    reasoning: str


# Stage 5 verification: L4 testability classification
class L4Verdict(_StrictBase):
    verdict: Literal["PURELY_TESTABLE", "PARTIALLY_TESTABLE", "NOT_TESTABLE"]
    reasoning: str
```

- [ ] **Step 4: Run test to verify it passes**

```bash
uv run pytest tests/test_schemas.py -v
```

Expected: 10 passed.

- [ ] **Step 5: Commit**

```bash
git add pipeline/schemas.py tests/test_schemas.py
git commit -m "$(cat <<'EOF'
feat(pipeline): add Pydantic schemas for every LLM-emitted shape

Phase 1 item 4. Strict (extra='forbid') models for Dimension,
PaperScore, Composition, L1Item, L3Verdict, L4Verdict. Wiring into
compose.py parsing happens in a later task.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

> **Note for the engineer:** verify the actual L3/L4 verdict literals against existing saved compose runs by sampling a couple of `compose_*.json` files under `data/pipeline/compositions/`. If a literal in real data is different (e.g. `"weak"` lowercase), update the Literal — these are the prompt's contract, not invented.

---

## Task 6: `parse_or_repair` helper with one-shot retry

**Files:**
- Create: `pipeline/llm_io.py` (initial scaffold + `parse_or_repair`)
- Test: `tests/test_llm_io.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_llm_io.py`:

```python
"""Tests for the LLM I/O wrapper layer."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pipeline.llm_io import ParseResult, parse_or_repair
from pipeline.schemas import DimensionsOutput


def _good_dim_json() -> str:
    return json.dumps({
        "dimensions": [
            {"name": "x", "description": "y", "example_question": "z"},
        ]
    })


def test_parse_or_repair_happy_path():
    raw = "```json\n" + _good_dim_json() + "\n```"
    result = parse_or_repair(raw, DimensionsOutput, repair_call=None)
    assert isinstance(result, ParseResult)
    assert result.status == "ok"
    assert result.value.dimensions[0].name == "x"
    assert result.errors == []


def test_parse_or_repair_retries_once_on_invalid_json():
    calls: list[str] = []

    def fake_repair(repair_prompt: str) -> tuple[bool, str, str]:
        calls.append(repair_prompt)
        return (True, _good_dim_json(), "")

    bad_raw = "{ not json at all"
    result = parse_or_repair(bad_raw, DimensionsOutput, repair_call=fake_repair)
    assert result.status == "repaired"
    assert result.value.dimensions[0].name == "x"
    assert len(calls) == 1


def test_parse_or_repair_gives_up_after_one_failed_repair():
    def always_bad(_repair_prompt: str) -> tuple[bool, str, str]:
        return (True, "still not json", "")

    result = parse_or_repair("{ broken", DimensionsOutput, repair_call=always_bad)
    assert result.status == "failed"
    assert result.value is None
    assert len(result.errors) >= 1


def test_parse_or_repair_returns_failed_when_no_repair_caller_and_bad_input():
    result = parse_or_repair("not json", DimensionsOutput, repair_call=None)
    assert result.status == "failed"
    assert result.value is None
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_llm_io.py -v
```

Expected: ImportError on `pipeline.llm_io`.

- [ ] **Step 3: Implement `parse_or_repair`**

Create `pipeline/llm_io.py`:

```python
"""LLM I/O wrappers: artifact-saving codex invocation + schema-validated parsing.

This module wraps subprocess-based `codex exec` calls so that every stage:
  - persists prompt, stdout, stderr, parsed JSON, and meta.json under
    <run_dir>/llm_calls/<stage>/<NN>/
  - validates output against a Pydantic model with one shot at JSON
    repair before giving up
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Callable, Generic, TypeVar

from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)

# (ok, stdout, stderr)
RepairCall = Callable[[str], tuple[bool, str, str]]


@dataclass
class ParseResult(Generic[T]):
    status: str  # "ok" | "repaired" | "failed"
    value: T | None
    errors: list[str] = field(default_factory=list)
    raw: str = ""


def _extract_json_block(text: str) -> str | None:
    """Pull the first JSON object out of LLM output. Mirrors the legacy
    _parse_json_block heuristic in compose.py: fenced ```json ... ``` blocks
    first, then the first balanced {...}."""
    if not text:
        return None
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        return fenced.group(1)
    brace = re.search(r"(\{.*\})", text, re.DOTALL)
    if not brace:
        return None
    body = brace.group(1)
    depth = 0
    end = -1
    for i, c in enumerate(body):
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                end = i
                break
    if end > 0:
        return body[: end + 1]
    return None


def _validate(raw: str, model_cls: type[T]) -> tuple[T | None, list[str]]:
    block = _extract_json_block(raw)
    if block is None:
        return None, ["could not extract JSON block from response"]
    try:
        data = json.loads(block)
    except json.JSONDecodeError as e:
        return None, [f"JSONDecodeError: {e}"]
    try:
        return model_cls.model_validate(data), []
    except ValidationError as e:
        return None, [str(e)]


def parse_or_repair(
    raw: str,
    model_cls: type[T],
    repair_call: RepairCall | None,
) -> ParseResult[T]:
    """Try to parse `raw` into `model_cls`. On failure, optionally invoke
    `repair_call` once with a repair prompt; if that also fails, return
    a `failed` result with accumulated error messages.

    repair_call signature: (repair_prompt: str) -> (ok, stdout, stderr).
    """
    value, errors = _validate(raw, model_cls)
    if value is not None:
        return ParseResult(status="ok", value=value, errors=[], raw=raw)

    if repair_call is None:
        return ParseResult(status="failed", value=None, errors=errors, raw=raw)

    repair_prompt = (
        "Your previous response could not be parsed against the required "
        "JSON schema. Here is the validation error:\n\n"
        f"{errors[0]}\n\n"
        "Your previous response (verbatim):\n"
        f"{raw}\n\n"
        "Return ONLY a corrected JSON object that conforms to the schema. "
        "No prose, no fences."
    )
    ok, stdout, _stderr = repair_call(repair_prompt)
    if not ok:
        errors.append("repair call returned non-ok")
        return ParseResult(status="failed", value=None, errors=errors, raw=raw)

    value2, errors2 = _validate(stdout, model_cls)
    if value2 is not None:
        return ParseResult(status="repaired", value=value2, errors=[], raw=stdout)
    return ParseResult(status="failed", value=None, errors=errors + errors2, raw=stdout)
```

- [ ] **Step 4: Run test to verify it passes**

```bash
uv run pytest tests/test_llm_io.py -v
```

Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add pipeline/llm_io.py tests/test_llm_io.py
git commit -m "$(cat <<'EOF'
feat(pipeline): add parse_or_repair for schema-validated LLM output

Phase 1 item 4 (parsing layer). One repair retry on validation failure,
then status=failed with accumulated errors. No artifact persistence yet
- that lands in the next task.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: `run_codex_with_artifacts` wrapper

**Files:**
- Modify: `pipeline/llm_io.py` (add wrapper)
- Modify: `tests/test_llm_io.py` (add wrapper tests)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_llm_io.py`:

```python
def test_run_codex_with_artifacts_persists_full_call(tmp_path: Path, monkeypatch):
    from pipeline import llm_io

    # Stub the underlying codex invocation so the test never spawns a subprocess.
    def fake_invoker(prompt: str, reasoning: str) -> tuple[bool, str, str]:
        return True, json.dumps({"hello": "world"}), ""

    monkeypatch.setattr(llm_io, "_invoke_codex", fake_invoker)

    ok, stdout, stderr, artifact_dir = llm_io.run_codex_with_artifacts(
        stage="dimensions",
        prompt="some prompt text",
        run_dir=tmp_path,
        reasoning="high",
    )

    assert ok
    assert "hello" in stdout
    assert artifact_dir.is_dir()
    assert (artifact_dir / "prompt.txt").read_text() == "some prompt text"
    assert "hello" in (artifact_dir / "stdout.txt").read_text()
    meta = json.loads((artifact_dir / "meta.json").read_text())
    assert meta["stage"] == "dimensions"
    assert meta["reasoning"] == "high"
    assert meta["exit_code"] == 0
    assert "duration_s" in meta


def test_run_codex_with_artifacts_sequences_multiple_calls(tmp_path: Path, monkeypatch):
    from pipeline import llm_io

    monkeypatch.setattr(llm_io, "_invoke_codex",
                        lambda prompt, reasoning: (True, "{}", ""))

    dirs = []
    for _ in range(3):
        _, _, _, d = llm_io.run_codex_with_artifacts(
            stage="dimensions",
            prompt="x",
            run_dir=tmp_path,
            reasoning="high",
        )
        dirs.append(d.name)

    # Three calls under the same stage get sequential zero-padded names
    assert dirs == ["01", "02", "03"]
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_llm_io.py::test_run_codex_with_artifacts_persists_full_call -v
```

Expected: AttributeError or ImportError on `run_codex_with_artifacts`.

- [ ] **Step 3: Implement the wrapper**

Append to `pipeline/llm_io.py`:

```python
import subprocess
import time
from pathlib import Path

CODEX_TIMEOUT_S = 1500


def _invoke_codex(prompt: str, reasoning: str) -> tuple[bool, str, str]:
    """Real codex invocation. Mockable via monkeypatch in tests."""
    repo_root = Path(__file__).resolve().parent.parent
    try:
        result = subprocess.run(
            ["codex", "exec", "--skip-git-repo-check", "-",
             "-C", str(repo_root), "-s", "read-only",
             "-c", f'model_reasoning_effort="{reasoning}"'],
            input=prompt,
            capture_output=True,
            text=True,
            timeout=CODEX_TIMEOUT_S,
        )
        return (result.returncode == 0, result.stdout, result.stderr)
    except subprocess.TimeoutExpired:
        return (False, "", f"codex timeout ({CODEX_TIMEOUT_S}s)")
    except FileNotFoundError:
        return (False, "", "codex binary not found")


def _next_seq_dir(stage_dir: Path) -> Path:
    """Return <stage_dir>/<NN>/ where NN is the next zero-padded index."""
    stage_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(p.name for p in stage_dir.iterdir() if p.is_dir())
    next_n = len(existing) + 1
    return stage_dir / f"{next_n:02d}"


def run_codex_with_artifacts(
    stage: str,
    prompt: str,
    run_dir: Path,
    reasoning: str = "high",
) -> tuple[bool, str, str, Path]:
    """Invoke codex and persist prompt, stdout, stderr, and meta.json
    under <run_dir>/llm_calls/<stage>/<NN>/. Returns
    (ok, stdout, stderr, artifact_dir).
    """
    stage_dir = run_dir / "llm_calls" / stage
    artifact_dir = _next_seq_dir(stage_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    (artifact_dir / "prompt.txt").write_text(prompt)
    t0 = time.monotonic()
    ok, stdout, stderr = _invoke_codex(prompt, reasoning)
    duration_s = time.monotonic() - t0

    (artifact_dir / "stdout.txt").write_text(stdout)
    (artifact_dir / "stderr.txt").write_text(stderr)
    meta = {
        "stage": stage,
        "reasoning": reasoning,
        "exit_code": 0 if ok else 1,
        "duration_s": round(duration_s, 3),
        "parse_status": "pending",  # caller updates after schema validation
    }
    (artifact_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return ok, stdout, stderr, artifact_dir


def update_artifact_meta(artifact_dir: Path, **fields) -> None:
    """Merge fields into <artifact_dir>/meta.json. Used by callers to record
    parse_status, validation_errors, etc., after parse_or_repair runs."""
    meta_path = artifact_dir / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    meta.update(fields)
    meta_path.write_text(json.dumps(meta, indent=2))
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_llm_io.py -v
```

Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add pipeline/llm_io.py tests/test_llm_io.py
git commit -m "$(cat <<'EOF'
feat(pipeline): add run_codex_with_artifacts wrapper

Phase 1 item 3. Persists prompt/stdout/stderr/meta.json per call,
zero-padded sequence under llm_calls/<stage>/. update_artifact_meta
lets callers stamp parse_status after schema validation.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: Migrate compose.py to use the wrapper + schemas

**Files:**
- Modify: `pipeline/compose.py` (6 call sites: ~172, 235, 339, 481, 645, 717)

This task is the integration point. The pattern repeats 6 times: replace `run_codex` + `_parse_json_block` with `run_codex_with_artifacts` + `parse_or_repair`. We migrate one stage at a time, leaving the test suite green after every commit.

> **Engineer note:** the existing `_parse_json_block` stays in `compose.py` until all 7 call sites are migrated, since some intermediate commits may still use it. After the last migration, delete it.

- [ ] **Step 1: Add the imports + thread `run_dir` into each stage helper**

In `pipeline/compose.py` near the top:

```python
from pipeline.llm_io import (
    parse_or_repair,
    run_codex_with_artifacts,
    update_artifact_meta,
)
from pipeline.schemas import (
    CompositionsOutput,
    DimensionsOutput,
    L1Item,
    L3Verdict,
    L4Verdict,
    ScoresOutput,
)
```

- [ ] **Step 2: Migrate `extract_dimensions` (compose.py:166–175)**

Replace the function body with:

```python
def extract_dimensions(papers: list[dict], problem: str, run_dir: Path) -> dict:
    prompt = DIMENSIONS_PROMPT_TEMPLATE.format(
        problem=problem,
        n_papers=len(papers),
        corpus_summary=_corpus_summary(papers, with_index=True),
    )
    ok, stdout, stderr, artifact_dir = run_codex_with_artifacts(
        stage="dimensions", prompt=prompt, run_dir=run_dir, reasoning="high",
    )
    if not ok:
        update_artifact_meta(artifact_dir, parse_status="codex_failed")
        return {"error": stderr, "dimensions": []}

    def _repair(p: str) -> tuple[bool, str, str]:
        return run_codex_with_artifacts(
            stage="dimensions_repair", prompt=p, run_dir=run_dir, reasoning="high",
        )[:3]

    result = parse_or_repair(stdout, DimensionsOutput, repair_call=_repair)
    update_artifact_meta(
        artifact_dir,
        parse_status=result.status,
        validation_errors=result.errors,
    )
    if result.value is None:
        return {"error": "schema_validation_failed", "dimensions": []}
    return result.value.model_dump()
```

Then update the only caller of `extract_dimensions` (in `compose_one`, currently `extract_dimensions(corpus, problem)`) to pass `out_dir` as `run_dir`.

- [ ] **Step 3: Migrate the remaining 5 LLM-call functions using the same pattern**

For each function below, apply the exact transformation shown in Step 2 (`extract_dimensions`), substituting the function name, schema class, stage name, and reasoning level. After each function migration, run `uv run pytest -v` to confirm green before moving to the next.

| Function (current line) | Schema class | Stage name | Reasoning | Notes |
|---|---|---|---|---|
| `score_papers` (~229–238) | `ScoresOutput` | `scoring` | `"high"` | Caller is in `compose_one`, currently `score_papers(corpus, dimensions, problem)`; thread `run_dir=out_dir`. |
| `find_compositions` (~339) | `CompositionsOutput` | `compositions` | `"high"` | Caller in `compose_one`; thread `run_dir=out_dir`. |
| `verify_consistency` (~481) | `L1Item` | `verify_l1_<cid>` | `"medium"` | Runs once per composition. Use the composition id (e.g. `"verify_l1_C1"`) so artifact dirs stay separable. Caller iterates over `compositions` — thread `run_dir`. |
| `verify_adversarial` (~645) | `L3Verdict` | `verify_l3_<cid>` | `"high"` | Per-composition; same composition-id stage naming. |
| L4 testability call (~717) | `L4Verdict` | `verify_l4_<cid>` | `"high"` | Per-composition; same composition-id stage naming. The function may currently be called `verify_testability` or inline — verify by reading the surrounding 20 lines first. |

Concrete template — apply to each function in the table:

```python
def <function_name>(...existing_args..., run_dir: Path) -> dict:
    prompt = <EXISTING_PROMPT_TEMPLATE>.format(...)
    ok, stdout, stderr, artifact_dir = run_codex_with_artifacts(
        stage=<STAGE>, prompt=prompt, run_dir=run_dir, reasoning=<REASONING>,
    )
    if not ok:
        update_artifact_meta(artifact_dir, parse_status="codex_failed")
        return {"error": stderr, <existing_fallback_key>: []}

    def _repair(p: str) -> tuple[bool, str, str]:
        return run_codex_with_artifacts(
            stage=<STAGE> + "_repair", prompt=p, run_dir=run_dir, reasoning=<REASONING>,
        )[:3]

    result = parse_or_repair(stdout, <SCHEMA_CLASS>, repair_call=_repair)
    update_artifact_meta(
        artifact_dir,
        parse_status=result.status,
        validation_errors=result.errors,
    )
    if result.value is None:
        return {"error": "schema_validation_failed", <existing_fallback_key>: []}
    return result.value.model_dump()
```

For per-composition verifiers (`verify_consistency`, `verify_adversarial`, L4): `<STAGE>` is dynamic per call, so build it inline:

```python
    stage = f"verify_l1_{c.get('id', 'unknown')}"
```

- [ ] **Step 4: Delete the old `_parse_json_block` and the old `run_codex`**

Once all 6 call sites have moved, search for `run_codex(` and `_parse_json_block(` in `pipeline/compose.py`. There should be zero remaining call-site matches (only the function definitions themselves). Delete:

```bash
grep -n "run_codex(\|_parse_json_block(" pipeline/compose.py
```

Expected: matches show only `def run_codex(` and `def _parse_json_block(` — the definitions. If any call-site matches remain, finish migrating those before proceeding.

Then delete the function bodies of `run_codex` (compose.py:51–68) and `_parse_json_block` (compose.py:737–773), plus the now-unused `subprocess` import if no other code in `compose.py` uses it.

- [ ] **Step 5: Run all tests + ruff**

```bash
uv run pytest -v
uv run ruff check .
```

Expected: green.

- [ ] **Step 6: Commit**

```bash
git add pipeline/compose.py
git commit -m "$(cat <<'EOF'
refactor(compose): migrate all LLM stages to wrapper + schemas

Phase 1 items 3 + 4. Every codex call now persists artifacts under
llm_calls/<stage>/<NN>/ and validates output against a Pydantic
schema with one shot at JSON repair. Deletes the old run_codex and
_parse_json_block helpers.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 9: Strict relevance gate

**Files:**
- Modify: `pipeline/corpus_gate.py:274–321` (signature + abort flow)
- Modify: `pipeline/compose.py` (CLI flag + plumb through)
- Modify: `tests/test_corpus_gate.py` (new tests)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_corpus_gate.py`:

```python
def test_validate_corpus_strict_aborts_when_relevance_skipped():
    papers = [{
        "paper_id": "P1",
        "openalex_id": "https://openalex.org/W1",
        "title": "An attention mechanism for sequence modeling",
        "year": 2024,
        "abstract": "We propose a novel attention mechanism. " * 30,
    }]
    report = validate_corpus(
        papers, problem="study attention",
        strict=True, api_key=None,
    )
    assert report.aborted
    assert "relevance" in report.abort_reason.lower()
    assert "relevance" in report.skipped_checks


def test_validate_corpus_non_strict_does_not_abort_on_skipped_relevance():
    papers = [{
        "paper_id": "P1",
        "openalex_id": "https://openalex.org/W1",
        "title": "An attention mechanism for sequence modeling",
        "year": 2024,
        "abstract": "We propose a novel attention mechanism. " * 30,
    }]
    report = validate_corpus(
        papers, problem="study attention",
        strict=False, api_key=None,
    )
    assert not report.aborted
    assert "relevance" in report.skipped_checks
```

- [ ] **Step 2: Run the new tests to verify they fail**

```bash
uv run pytest tests/test_corpus_gate.py::test_validate_corpus_strict_aborts_when_relevance_skipped \
              tests/test_corpus_gate.py::test_validate_corpus_non_strict_does_not_abort_on_skipped_relevance -v
```

Expected: TypeError on unknown `strict` keyword.

- [ ] **Step 3: Add `strict` parameter to `validate_corpus`**

Modify `pipeline/corpus_gate.py:274–321`. Replace the `validate_corpus` signature and body to look like:

```python
def validate_corpus(
    papers: list[dict],
    problem: str,
    relevance_threshold: float = 0.30,
    max_fail_rate: float = 0.30,
    api_key: str | None = None,
    strict: bool = False,
) -> CorpusGateReport:
    """Run all three checks. Returns a report with drop_ids (papers that
    failed any check) and an abort flag if the failure rate exceeds
    max_fail_rate OR (strict and relevance was skipped)."""
    if api_key is None:
        api_key = os.environ.get("OPENAI_API_KEY")

    rel_issues, rel_skipped = check_relevance(papers, problem, relevance_threshold, api_key)
    enc_issues = check_encoding(papers)
    str_issues = check_structure(papers)

    all_issues = rel_issues + enc_issues + str_issues
    drop_ids = {i.paper_id for i in all_issues}
    by_check = {
        "relevance": len(rel_issues),
        "encoding": len(enc_issues),
        "structure": len(str_issues),
    }
    total = len(papers)
    failed = len(drop_ids)
    passed = total - failed
    fail_rate = failed / total if total else 0.0

    report = CorpusGateReport(
        total=total,
        passed=passed,
        failed=failed,
        fail_rate=fail_rate,
        by_check=by_check,
        drop_ids=drop_ids,
        issues=all_issues,
    )
    if rel_skipped:
        report.skipped_checks.append("relevance")

    if strict and rel_skipped:
        report.aborted = True
        report.abort_reason = (
            "relevance check skipped (no API key or transient error) "
            "and --strict-gate is enabled; set OPENAI_API_KEY or pass "
            "--no-strict-gate to proceed"
        )
        return report

    if fail_rate > max_fail_rate:
        report.aborted = True
        report.abort_reason = (
            f"fail rate {fail_rate:.1%} exceeds max_fail_rate {max_fail_rate:.1%}; "
            f"corpus quality is insufficient for reliable composition"
        )
    return report
```

- [ ] **Step 4: Run the new tests to verify they pass**

```bash
uv run pytest tests/test_corpus_gate.py -v
```

Expected: all green.

- [ ] **Step 5: Thread `strict_gate` into `compose_one`**

In `pipeline/compose.py`, add a `strict_gate: bool = True` parameter to `compose_one`. Then at the call to `validate_corpus` (around line 837), pass `strict=strict_gate and not dry_run`:

```python
        report = validate_corpus(
            corpus, problem,
            relevance_threshold=relevance_threshold,
            max_fail_rate=max_fail_rate,
            strict=strict_gate and not dry_run,
        )
```

- [ ] **Step 6: Add the CLI flag**

In `pipeline/compose.py:~1112`, find the argparse setup. Add:

```python
    p.add_argument(
        "--strict-gate",
        dest="strict_gate",
        action="store_true",
        default=True,
        help="Abort if Step 1.5 relevance check is skipped (default: on for non-dry-run)",
    )
    p.add_argument(
        "--no-strict-gate",
        dest="strict_gate",
        action="store_false",
        help="Allow Step 1.5 to skip the relevance check (warning only)",
    )
```

And pass it through to `compose_one(..., strict_gate=args.strict_gate)`.

- [ ] **Step 7: Run all tests + ruff**

```bash
uv run pytest -v
uv run ruff check .
```

Expected: green.

- [ ] **Step 8: Commit**

```bash
git add pipeline/corpus_gate.py pipeline/compose.py tests/test_corpus_gate.py
git commit -m "$(cat <<'EOF'
feat(corpus_gate): add strict mode that aborts on skipped relevance

Phase 1 item 5. validate_corpus(strict=True) aborts when relevance
check is skipped (typically due to missing OPENAI_API_KEY). compose
CLI gains --strict-gate / --no-strict-gate; strict default for
non-dry-run.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 10: Digest renderer badge for skipped relevance

**Files:**
- Modify: wherever the markdown digest is produced (find via `grep -rn "digest\|render_digest\|to_markdown" pipeline/ scripts/`)

> **Engineer note:** the spec assumes a digest renderer exists. If `grep` finds no central renderer (the codebase may produce per-run summaries inline in `compose.py`), simplify this task to: emit a `relevance_skipped: true` field at the top of the run's `state.json` and a single-line warning to stdout when `print_report` prints the report.

- [ ] **Step 1: Locate the renderer**

```bash
grep -rn "digest\|render_digest\|skipped_checks" pipeline/ scripts/
```

If a clear digest renderer exists, modify it. Otherwise, fall back to the simpler change below.

- [ ] **Step 2 (fallback simpler change): Surface in `print_report` and `state.json`**

In `pipeline/corpus_gate.py`, modify `print_report` so the "skipped checks" line is prefixed with a warning marker:

Find:
```python
    if report.skipped_checks:
        print(f"  skipped checks: {', '.join(report.skipped_checks)}")
```

Replace with:
```python
    if report.skipped_checks:
        print(f"  ⚠️ SKIPPED CHECKS: {', '.join(report.skipped_checks)} "
              f"— corpus quality conclusions may be weaker than reported")
```

In `pipeline/compose.py`, where `state["corpus_gate"] = report.as_dict()` is set, also set:

```python
        state["relevance_skipped"] = "relevance" in report.skipped_checks
```

- [ ] **Step 3: Write a test**

Add to `tests/test_corpus_gate.py`:

```python
def test_print_report_warns_on_skipped_relevance(capsys):
    from pipeline.corpus_gate import (
        CorpusGateReport, print_report,
    )
    r = CorpusGateReport(
        total=5, passed=5, failed=0, fail_rate=0.0,
        by_check={"relevance": 0, "encoding": 0, "structure": 0},
        drop_ids=set(),
    )
    r.skipped_checks.append("relevance")
    print_report(r)
    captured = capsys.readouterr()
    assert "SKIPPED" in captured.out
    assert "relevance" in captured.out.lower()
```

- [ ] **Step 4: Run tests + ruff**

```bash
uv run pytest -v
uv run ruff check .
```

Expected: green.

- [ ] **Step 5: Commit**

```bash
git add pipeline/corpus_gate.py pipeline/compose.py tests/test_corpus_gate.py
git commit -m "$(cat <<'EOF'
feat(reports): surface skipped relevance check prominently

Phase 1 item 5 (cont). print_report now prefixes skipped checks with
a warning. state.json gains a top-level relevance_skipped boolean
so downstream tools can render badges without re-parsing.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 11: Final integration sweep + findings update

**Files:**
- Create: `findings/phase1-remediation-20260521.md`
- Modify: track-record memory if convenient

- [ ] **Step 1: Run the full validation set**

```bash
uv run pytest -v
uv run ruff check .
uv run python -m pipeline.compose --help  # sanity: CLI still works
```

Expected: pytest green, ruff `All checks passed!`, --help renders the new flags.

- [ ] **Step 2: Do a dry-run compose end-to-end to confirm artifact layout**

```bash
uv run python -m pipeline.compose \
    --topic-list data/topics/cs-bridges.json \
    --problem "test compose artifact persistence" \
    --max-papers 5 \
    --dry-run
```

> If the project doesn't have a small/cheap topic list to use, substitute one from `data/topics/`. Confirm `corpus_raw.json`, `corpus_validated.json`, and `corpus.json` all appear in the run directory, and every paper has a `paper_uid`.

- [ ] **Step 3: Write findings entry**

Create `findings/phase1-remediation-20260521.md`:

```markdown
# Phase 1 remediation - 2026-05-21

Implementing the Phase 1 plan from `findings/project-deep-analysis-20260521.md`.

## Shipped

- Ruff lint clean (was 20 errors).
- `paper_uid` stable identifier (16-hex SHA-256 prefix) attached to every paper.
- Dual-write: `corpus_raw.json` (pre-gate) and `corpus_validated.json` (post-gate). `corpus.json` retained as a copy for back-compat.
- Per-call LLM artifacts under `<run_dir>/llm_calls/<stage>/<NN>/` (`prompt.txt`, `stdout.txt`, `stderr.txt`, `meta.json`).
- Pydantic v2 schemas for every LLM-emitted shape; one-shot repair retry on validation failure.
- Strict relevance gate (`--strict-gate` default for non-dry-run). Aborts if relevance check skipped.

## Deferred to Phase 2

- Embed-quant top-k fix, calib_db reuse, bit-budget labeling.
- Round-trip tests for compression experiments.
- Stronger Heilbronn baselines + seed sweeps + CIs.
- Real cache traces (SNIA / MSR Cambridge).
- Compose integration tests with fake OpenAlex + fake codex.

## Deferred to Phase 3

- Verification recipes per WEAK+PURELY_TESTABLE composition.
- Post-L3 novelty check stage.
- Calibration dashboard.
- Domain packs.
- Expert hit-rate study.

## Open items inside Phase 1

- Local embedding fallback for the relevance gate (intentionally not added; spec decision 2026-05-21). Revisit if missing API key blocks too many dev runs.
- Backfill paper_uid into pre-existing saved run state directories (out of scope by design).
```

- [ ] **Step 4: Commit**

```bash
git add findings/phase1-remediation-20260521.md
git commit -m "$(cat <<'EOF'
docs(findings): record Phase 1 remediation outcome

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 5: Final verification**

```bash
git log --oneline -15
uv run pytest -v
uv run ruff check .
```

Expected: clean log of focused commits, all tests green, lint clean.

---

## Self-review (already performed by author)

- **Spec coverage:** all 5 spec items map to tasks — item 1 → Task 1; item 2 → Tasks 2–3; item 3 → Tasks 6–8; item 4 → Tasks 4–5 and threaded into Task 8; item 5 → Tasks 9–10.
- **Type consistency:** `paper_uid` field, `run_dir` parameter, `stage` argument names, and Pydantic model class names are used identically across tasks.
- **Placeholders:** none — every step shows the actual code or command.
- **TDD discipline:** every non-config task starts with a failing test before implementation.
- **Frequent commits:** 11 commits, each on a green build.

## Risks & mitigations recap

- **Task 8 is large.** Mitigation: it's structured as 7 internal sub-steps and the engineer can commit between sub-steps if a stage migration is non-trivial.
- **Schema literals might diverge from real prompts.** Mitigation: explicit engineer note in Task 5 tells them to sample saved runs and adjust literals before locking schemas.
- **Strict gate could surprise dev workflows.** Mitigation: error message embeds the opt-out flag and the env-var fix.
