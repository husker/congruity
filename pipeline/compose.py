"""Compositional synthesis — find subsets of papers whose combined contributions
solve a problem better than any single paper.

This is a step beyond equivalence-finding (discover.py). Instead of "concept A in
literature X equals concept B in literature Y," we ask: "given problem Q and N
papers each addressing partial aspects, which subset of 2-5 papers composes into
a stronger solution than any individual paper?"

Pipeline:
    1. Fetch a corpus (post-cutoff papers from a topic list).
    2. Extract problem dimensions (LLM call 1).
    3. Score each paper on each dimension (LLM call 2).
    4. Find candidate compositions (LLM call 3).
    5. Three-level verification:
       L1 internal consistency       (LLM)
       L2 source grounding           (programmatic citation check)
       L3 adversarial cross-check    (second LLM pass with falsification prompt)
    6. Rank by verification status + dimension coverage.

Output: ranked compositions, each labeled with which verification levels it passed.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pipeline.corpus import sample_topic, search_works
from pipeline.corpus_gate import print_report, validate_corpus
from pipeline.identity import paper_uid
from pipeline.llm_io import (aggregate_llm_usage, parse_or_repair,
                             run_codex_with_artifacts, update_artifact_meta)
from pipeline.schemas import (
    CompositionsOutput,
    DimensionsOutput,
    L1Item,
    L3Verdict,
    L4Verdict,
    ScoresOutput,
)

COMPOSE_DIR = Path(__file__).resolve().parent.parent / "data" / "pipeline" / "compositions"
DEFAULT_POST_CUTOFF = 2024  # widened from discovery's 2025; compositional analysis benefits from larger pool
DEFAULT_PAPERS_PER_TOPIC = 4
DEFAULT_PAPERS_PER_QUERY = 25  # paper-direct search mode; results are relevance-ranked
DEFAULT_MAX_PAPERS = 100
DEFAULT_TARGET_COMPOSITIONS = 5
DEFAULT_MIN_GATE_PAPERS = 15
# Recipes go to the top-K priority comps, not an L3/L4 gate. Deliberately
# generous: the gate-precision series showed winners can sit mid-pack (the best
# composition in the project, C2, ranked ~4th of 6 on the available signals), so
# a tight short-list reproduces the very "bury the winner" bug this replaces.
DEFAULT_RECIPE_TOP_K = 5
CODEX_TIMEOUT_S = 1500  # 25 min — large-corpus compose stages can be slow
COMPOSE_ABSTRACT_WORDS = 80  # cap per-abstract words in compose-stage prompt to control cost
COMPOSE_MAX_PAPERS_PER_CLUSTER = 15


# ----------------------------------------------------------------------------
# Corpus assembly
# ----------------------------------------------------------------------------


def assemble_corpus(
    topic_list_path: Path,
    papers_per_topic: int = DEFAULT_PAPERS_PER_TOPIC,
    max_papers: int = DEFAULT_MAX_PAPERS,
    year_min: int = DEFAULT_POST_CUTOFF,
) -> list[dict]:
    """Read topic list → fetch papers from each → return flat list capped at max_papers."""
    topics = json.loads(topic_list_path.read_text())
    print(f"  topic list: {len(topics)} topics")
    all_papers: list[dict] = []
    seen_ids: set[str] = set()
    # Sort topics by works_count descending so most-active topics seed the corpus
    topics_sorted = sorted(topics, key=lambda t: -t.get("works_count", 0))
    for t in topics_sorted:
        if len(all_papers) >= max_papers:
            break
        print(f"  sampling {t['display_name'][:50]}...", end=" ", flush=True)
        papers = sample_topic(
            t["id"],
            year_min=year_min,
            cap=papers_per_topic,
            include_refs=False,
            include_abstract=True,
        )
        added = 0
        for p in papers:
            pid = p.get("openalex_id")
            if pid and pid not in seen_ids:
                seen_ids.add(pid)
                # Tag with source topic for later attribution
                p["source_topic"] = t["display_name"]
                p["source_topic_id"] = t["id"]
                all_papers.append(p)
                added += 1
                if len(all_papers) >= max_papers:
                    break
        print(f"+{added}")
    return all_papers


def assemble_corpus_from_search(
    queries: list[str],
    papers_per_query: int = DEFAULT_PAPERS_PER_QUERY,
    max_papers: int = DEFAULT_MAX_PAPERS,
    year_min: int = DEFAULT_POST_CUTOFF,
) -> list[dict]:
    """Paper-direct corpus: fetch relevance-ranked papers per search query.

    For problems narrower than any OpenAlex topic, where within-topic random
    sampling yields almost no on-problem papers and the Step 1.5 gate aborts.
    Queries are taken in order; duplicates across queries are dropped.
    """
    print(f"  paper search: {len(queries)} queries")
    all_papers: list[dict] = []
    seen_ids: set[str] = set()
    for q in queries:
        if len(all_papers) >= max_papers:
            break
        print(f"  searching {q[:60]!r}...", end=" ", flush=True)
        papers = search_works(
            q,
            year_min=year_min,
            cap=papers_per_query,
            include_abstract=True,
        )
        added = 0
        for p in papers:
            pid = p.get("openalex_id")
            if pid and pid not in seen_ids:
                seen_ids.add(pid)
                p["source_topic"] = f"search: {q}"
                p["source_topic_id"] = ""
                all_papers.append(p)
                added += 1
                if len(all_papers) >= max_papers:
                    break
        print(f"+{added}")
    return all_papers


# ----------------------------------------------------------------------------
# Stage 1: extract problem dimensions
# ----------------------------------------------------------------------------


DIMENSIONS_PROMPT_TEMPLATE = """You are analyzing a corpus of scientific papers to support compositional synthesis.

PROBLEM: {problem}
{pack_context}
CORPUS ({n_papers} papers):
{corpus_summary}

YOUR TASK
Identify 5-8 *problem dimensions* — orthogonal aspects of the problem above that
papers in the corpus address. Each dimension should be:
- Concrete and measurable (not vague like "performance")
- Independent (covering different aspects, not overlapping)
- Actually present in the corpus (at least 2 papers address each dimension)

OUTPUT — JSON ONLY, no prose:
{{
  "dimensions": [
    {{"name": "<short_name>", "description": "<one sentence>", "example_question": "<question this dimension answers>"}},
    ...
  ]
}}

Be specific. For example, in a corpus on direct air capture, "capture mechanism"
is too vague — "CO2 binding chemistry (chemisorption vs physisorption)" is good.
"""


def _abstract_snippet(p: dict, max_words: int = 60) -> str:
    abstract = p.get("abstract", "") or ""
    words = abstract.split()
    if len(words) > max_words:
        return " ".join(words[:max_words]) + "..."
    return abstract


def _corpus_summary(papers: list[dict], with_index: bool = True) -> str:
    """Render a corpus into the per-paper block prompts use.

    Uses p["paper_id"] when available (assigned by compose_one as P1..PN
    over the full corpus). For batched scoring this matters: every batch
    must label its papers with their corpus-level IDs, not local 1..K
    indices, otherwise downstream scores collide across batches.
    """
    lines = []
    for i, p in enumerate(papers, 1):
        pid = p.get("paper_id") or f"P{i}"
        idx = f"{pid}: " if with_index else ""
        topic = p.get("source_topic", "?")
        lines.append(f"{idx}[{p.get('year', '?')}] {p.get('title', '?')} (topic: {topic})")
        lines.append(f"     {_abstract_snippet(p, max_words=50)}")
    return "\n".join(lines)


def extract_dimensions(papers: list[dict], problem: str, run_dir: Path,
                       pack_context: str = "") -> dict:
    prompt = DIMENSIONS_PROMPT_TEMPLATE.format(
        problem=problem,
        pack_context=pack_context,
        n_papers=len(papers),
        corpus_summary=_corpus_summary(papers, with_index=True),
    )
    ok, stdout, stderr, artifact_dir = run_codex_with_artifacts(
        stage="dimensions", prompt=prompt, run_dir=run_dir, reasoning="high",
    )
    if not ok:
        update_artifact_meta(artifact_dir, parse_status="codex_failed")
        return {"error": stderr, "dimensions": []}

    def _repair(p: str) -> tuple[bool, str, str, Path]:
        return run_codex_with_artifacts(
            stage="dimensions_repair", prompt=p, run_dir=run_dir, reasoning="high",
        )

    result = parse_or_repair(stdout, DimensionsOutput, repair_call=_repair)
    update_artifact_meta(
        artifact_dir,
        parse_status=result.status,
        validation_errors=result.errors,
    )
    if result.value is None:
        return {"error": "schema_validation_failed", "dimensions": []}
    return result.value.model_dump()


# ----------------------------------------------------------------------------
# Stage 2: score papers on dimensions
# ----------------------------------------------------------------------------


SCORING_PROMPT_TEMPLATE = """You are scoring papers in a corpus on a set of problem dimensions.

PROBLEM: {problem}

DIMENSIONS:
{dimensions_block}

CORPUS:
{corpus_block}

YOUR TASK
For each paper Pi, give a score 0-5 on each dimension:
  0 = not addressed
  1 = mentioned in passing
  2 = partial contribution
  3 = substantive contribution
  4 = strong, but with explicit limitations
  5 = comprehensive treatment

Then also extract a *one-line summary of what this paper actually contributes* on
each dimension where score >= 2.

OUTPUT — JSON ONLY, no prose:
{{
  "scores": [
    {{
      "paper_id": "P1",
      "title_short": "<first 60 chars of title>",
      "dimensions": {{
        "<dim_name>": {{"score": N, "contribution": "<one line, omit if score < 2>"}},
        ...
      }}
    }},
    ...
  ]
}}
"""


def _dimensions_block(dimensions: list[dict]) -> str:
    return "\n".join(
        f"- {d['name']}: {d['description']}"
        for d in dimensions
    )


SCORING_BATCH_SIZE = 20

DEFAULT_MAX_CONCURRENCY = 4


def _max_concurrency() -> int:
    """Parallel codex calls for batch stages (scoring, verification).

    Codex runs as a subprocess, so concurrent calls are pure I/O overlap
    from our side. Override with CONGRUITY_MAX_CONCURRENCY; set 1 to
    restore fully serial behavior.
    """
    try:
        return max(1, int(os.environ.get("CONGRUITY_MAX_CONCURRENCY",
                                         DEFAULT_MAX_CONCURRENCY)))
    except ValueError:
        return DEFAULT_MAX_CONCURRENCY

SCORING_MIN_COVERAGE = 0.5

# When the first scoring attempt for a batch returns empty or partial output,
# we issue ONE custom-repair codex call that explicitly enumerates the
# paper IDs the model is expected to score. parse_or_repair's generic
# repair prompt has been observed to confuse the model into scoring the
# wrong papers with made-up dimension names.
SCORING_REPAIR_PROMPT_TEMPLATE = """Your previous response to the scoring task was
empty or incomplete. You MUST return scores for the following paper IDs and ONLY
these paper IDs:

  {paper_id_list}

Each paper must have a `paper_id` exactly matching one of the IDs above. The
`dimensions` object must use these dimension names (exactly):

  {dimension_names}

Score each paper on every listed dimension. The score is an integer from 0 to 5
(see scoring rubric in the original prompt below). Empty contribution strings
are OK for low scores.

OUTPUT — JSON ONLY, no prose, no markdown fences:
{{
  "scores": [
    {{"paper_id": "...", "title_short": "...", "dimensions": {{"<dim>": {{"score": N, "contribution": "<short>"}}, ...}}}},
    ...
  ]
}}

ORIGINAL PROMPT (for context):

{original_prompt}
"""


def _score_one_batch(
    batch: list[dict],
    dim_names: list[str],
    problem: str,
    dimensions: list[dict],
    run_dir: Path,
    batch_idx: int,
) -> tuple[list[dict], Path]:
    """Run one scoring batch with an explicit ID-aware repair on empty/partial.

    Returns (scored_papers, artifact_dir) for this batch. Persists primary +
    (if needed) repair artifacts under llm_calls/scoring/NN/ and
    llm_calls/scoring_repair/NN/ respectively.
    """
    prompt = SCORING_PROMPT_TEMPLATE.format(
        problem=problem,
        dimensions_block=_dimensions_block(dimensions),
        corpus_block=_corpus_summary(batch, with_index=True),
    )
    requested_ids = [p["paper_id"] for p in batch if p.get("paper_id")]
    ok, stdout, _stderr, artifact_dir = run_codex_with_artifacts(
        stage="scoring", prompt=prompt, run_dir=run_dir, reasoning="high",
    )
    if not ok:
        update_artifact_meta(
            artifact_dir,
            parse_status="codex_failed",
            batch_idx=batch_idx,
            batch_papers=len(batch),
            requested_ids=requested_ids,
        )
        return [], artifact_dir

    # First parse attempt with no auto-repair — we'll do our own
    # ID-aware repair if anything's missing.
    primary = parse_or_repair(stdout, ScoresOutput, repair_call=None)
    primary_scores: list[dict] = (
        [s.model_dump() for s in primary.value.scores] if primary.value else []
    )
    primary_pids = {s.get("paper_id", "") for s in primary_scores}
    missing = [pid for pid in requested_ids if pid not in primary_pids]

    if not missing:
        update_artifact_meta(
            artifact_dir,
            parse_status=primary.status,
            validation_errors=primary.errors,
            batch_idx=batch_idx,
            batch_papers=len(batch),
            requested_ids=requested_ids,
            scored_ids=list(primary_pids),
        )
        return primary_scores, artifact_dir

    # Custom-repair retry: explicit paper IDs + dimension names.
    repair_prompt = SCORING_REPAIR_PROMPT_TEMPLATE.format(
        paper_id_list=", ".join(missing),
        dimension_names=", ".join(dim_names),
        original_prompt=prompt,
    )
    ok_r, stdout_r, _stderr_r, _ = run_codex_with_artifacts(
        stage="scoring_repair", prompt=repair_prompt, run_dir=run_dir, reasoning="high",
    )
    repair_scores: list[dict] = []
    repair_pids: set[str] = set()
    if ok_r:
        repair_parsed = parse_or_repair(stdout_r, ScoresOutput, repair_call=None)
        if repair_parsed.value is not None:
            requested_set = set(missing)
            for s in repair_parsed.value.scores:
                sd = s.model_dump()
                pid = sd.get("paper_id", "")
                # Only accept scores for paper_ids we actually asked the
                # repair to fix — guards against the observed pattern of the
                # model returning a bogus score for an unrelated paper.
                if pid in requested_set and pid not in primary_pids:
                    repair_scores.append(sd)
                    repair_pids.add(pid)

    combined = primary_scores + repair_scores
    update_artifact_meta(
        artifact_dir,
        parse_status=("repaired_partial" if repair_pids else
                      ("repaired" if primary.status == "failed" and repair_pids
                       else primary.status)),
        validation_errors=primary.errors,
        batch_idx=batch_idx,
        batch_papers=len(batch),
        requested_ids=requested_ids,
        scored_ids_primary=sorted(primary_pids),
        scored_ids_repair=sorted(repair_pids),
        missing_after_repair=sorted(set(missing) - repair_pids),
    )
    return combined, artifact_dir


def score_papers(
    papers: list[dict],
    dimensions: list[dict],
    problem: str,
    run_dir: Path,
    *,
    batch_size: int = SCORING_BATCH_SIZE,
) -> dict:
    """Score papers in batches with ID-aware per-batch repair.

    The all-in-one prompt is a silent-failure mode: a 95-paper × 8-dimension
    request asks the model for ~760 structured entries in one response and
    gpt-5.5 via codex routinely emits ``{"scores": []}`` after burning its
    reasoning budget. We split into batches of ``batch_size`` (default 20)
    and, on partial/empty output, retry once per batch with an
    enumerate-paper-IDs prompt that the model has been less prone to drop.
    """
    if not papers:
        return {
            "scores": [],
            "coverage": 0.0,
            "n_papers": 0,
            "n_batches": 0,
            "empty_batches": 0,
            "per_batch_yield": [],
        }

    all_scores: list[dict] = []
    seen_pids: set[str] = set()
    batches = [
        papers[i:i + batch_size]
        for i in range(0, len(papers), batch_size)
    ]
    dim_names = [d.get("name", "") for d in dimensions]
    per_batch_yield: list[int] = []
    last_artifact_dir: Path | None = None

    # Batches are independent codex calls — run them concurrently.
    # Dedup (seen_pids) is a deterministic post-pass in batch order, so the
    # output is identical to the serial loop.
    workers = min(_max_concurrency(), len(batches)) or 1
    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            batch_results = list(pool.map(
                lambda ib: _score_one_batch(
                    ib[1], dim_names, problem, dimensions, run_dir, ib[0]),
                enumerate(batches)))
    else:
        batch_results = [
            _score_one_batch(batch, dim_names, problem, dimensions,
                             run_dir, batch_idx)
            for batch_idx, batch in enumerate(batches)
        ]

    for batch_idx, (scored, _artifact_dir) in enumerate(batch_results):
        before = len(all_scores)
        for sd in scored:
            pid = sd.get("paper_id", "")
            if pid and pid not in seen_pids:
                all_scores.append(sd)
                seen_pids.add(pid)
        per_batch_yield.append(len(all_scores) - before)
        print(
            f"  scoring batch {batch_idx + 1}/{len(batches)}: "
            f"+{per_batch_yield[-1]} scores "
            f"({len(all_scores)}/{len(papers)} total)",
            flush=True,
        )

    coverage = len(all_scores) / len(papers) if papers else 0.0
    empty_batches = sum(1 for y in per_batch_yield if y == 0)
    # The run summary is stamped on the highest-numbered scoring artifact
    # dir (one scan, after all batches): under parallel execution the last
    # batch in batch order need not own the last NN/ dir.
    stage_dir = run_dir / "llm_calls" / "scoring"
    if stage_dir.exists():
        nn_dirs = sorted(p for p in stage_dir.iterdir() if p.is_dir())
        if nn_dirs:
            last_artifact_dir = nn_dirs[-1]
    if last_artifact_dir is not None:
        update_artifact_meta(
            last_artifact_dir,
            scoring_total_papers=len(papers),
            scoring_n_batches=len(batches),
            scoring_per_batch_yield=per_batch_yield,
            scoring_empty_batches=empty_batches,
            scoring_coverage=round(coverage, 3),
            scoring_coverage_warning=(
                None if (coverage >= SCORING_MIN_COVERAGE and empty_batches == 0)
                else (
                    f"{empty_batches}/{len(batches)} batches returned no scores; "
                    f"overall coverage {coverage:.0%}"
                    if empty_batches > 0
                    else f"only {coverage:.0%} of corpus papers received scores"
                )
            ),
        )
    return {
        "scores": all_scores,
        "coverage": round(coverage, 3),
        "n_papers": len(papers),
        "n_batches": len(batches),
        "empty_batches": empty_batches,
        "per_batch_yield": per_batch_yield,
    }


# ----------------------------------------------------------------------------
# Stage 3: find compositions
# ----------------------------------------------------------------------------


COMPOSE_PROMPT_TEMPLATE = """You are identifying compositional syntheses — subsets of
papers whose combined contributions solve the problem better than any single paper.

PROBLEM: {problem}
{pack_context}
PROBLEM DIMENSIONS:
{dimensions_block}

DIMENSION CLUSTERS (papers grouped by the dimension they score highest on):
{clusters_block}

CORPUS (truncated abstracts for context; full text available per-paper if needed):
{corpus_block}

YOUR TASK
Identify up to {target} compositional syntheses. Each is a subset of 2-4 papers
PULLED FROM DIFFERENT CLUSTERS where combining their contributions yields a result
no single paper provides.

A good composition has:
- Each paper from a DIFFERENT dimension cluster (covers different aspects)
- Each paper's contribution to its dimension is concrete (score >= 3)
- Physical/conceptual compatibility (papers' assumptions and parameters must align)
- A clear synthesis claim: "Paper A's X + Paper B's Y + Paper C's Z → new capability"
- Each claim traceable to a specific paper (provide a verbatim quote)
- A MECHANISM MAP: name the mathematical or algorithmic STRUCTURE that transfers
  (not a topic — the actual transplantable machinery, e.g. "cluster moves on
  Potts configurations" rather than "physics methods"), and which specific
  failure mode of the destination field's SOTA it addresses. Compositions that
  merely concatenate components without a transferring structure are weak.

Cross-disciplinary compositions are especially valuable: if one cluster's papers
come from physics or biology and another cluster's come from CS theory, and they
genuinely compose on the problem, that is the Swanson-shape result we want.

Do NOT invent compositions. If no compatible combinations exist across clusters,
return an empty list. False positives are much worse than no result here.

OUTPUT — JSON ONLY:
{{
  "compositions": [
    {{
      "id": "C1",
      "synthesis_claim": "<one paragraph: what this combination achieves that no single paper does>",
      "papers": [
        {{
          "paper_id": "P3",
          "role": "<what this paper contributes — one phrase>",
          "supporting_dimensions": ["dim_name", ...],
          "supporting_quote": "<verbatim quote from the paper's abstract>"
        }},
        ...
      ],
      "compatibility_argument": "<one paragraph: why these papers' approaches compose without contradiction>",
      "mechanism_map": {{
        "transferring_structure": "<the transplantable mathematical/algorithmic structure>",
        "sota_failure_mode_addressed": "<the specific SOTA failure mode this attacks>"
      }},
      "covered_dimensions": ["dim_name", ...],
      "open_questions": ["<short>", ...],
      "confidence": <1-10 integer>,
      "confidence_reason": "<one line>"
    }},
    ...
  ]
}}
"""


def _scores_block(scores: list[dict]) -> str:
    lines = []
    for s in scores:
        dims = s.get("dimensions", {}) or {}
        scored = sorted(dims.items(), key=lambda kv: -kv[1].get("score", 0))
        top = [f"{name}={d.get('score', 0)}" for name, d in scored if d.get("score", 0) >= 2]
        lines.append(f"{s['paper_id']}: {s.get('title_short', '')}")
        if top:
            lines.append(f"  scores: {', '.join(top)}")
        for name, d in scored:
            contrib = d.get("contribution")
            if contrib and d.get("score", 0) >= 2:
                lines.append(f"  {name}: {contrib}")
    return "\n".join(lines)


def find_compositions(
    papers: list[dict],
    dimensions: list[dict],
    scores: list[dict],
    problem: str,
    *,
    target: int = DEFAULT_TARGET_COMPOSITIONS,
    run_dir: Path,
    pack_context: str = "",
) -> dict:
    """Find candidate compositions. Uses programmatic dimension clusters to
    bound the LLM's search space — critical for large corpora (~100 papers)."""
    # Refusing to generate compositions when scoring produced nothing
    # prevents the silent-failure mode where Stage 3 returns empty (HTTP
    # error, model bail-out, etc.) and Stage 4 still surfaces "candidates"
    # built from random clusters of unscored papers.
    if not scores:
        return {
            "error": "empty_scores",
            "compositions": [],
            "clusters": {},
        }
    papers_by_id = {p["paper_id"]: p for p in papers if p.get("paper_id")}
    scores_by_id = {s["paper_id"]: s for s in scores if s.get("paper_id")}
    clusters = cluster_papers_by_dimension(scores, dimensions)
    candidate_ids = _compose_candidate_ids(
        clusters,
        scores_by_id,
        max_papers_per_cluster=COMPOSE_MAX_PAPERS_PER_CLUSTER,
    )
    candidate_papers = [
        p for p in papers
        if p.get("paper_id") in candidate_ids
    ]
    print(
        f"  composition prompt candidates: {len(candidate_papers)}/{len(papers)} papers "
        f"from {sum(1 for pids in clusters.values() if pids)} clusters",
        flush=True,
    )
    if not candidate_papers:
        return {
            "error": "no_compose_candidates",
            "compositions": [],
            "clusters": clusters,
        }
    prompt = COMPOSE_PROMPT_TEMPLATE.format(
        problem=problem,
        pack_context=pack_context,
        target=target,
        dimensions_block=_dimensions_block(dimensions),
        clusters_block=_clusters_block(
            clusters,
            papers_by_id,
            scores_by_id,
            max_papers_per_cluster=COMPOSE_MAX_PAPERS_PER_CLUSTER,
        ),
        corpus_block=_corpus_full(candidate_papers, abstract_words=COMPOSE_ABSTRACT_WORDS),
    )
    ok, stdout, stderr, artifact_dir = run_codex_with_artifacts(
        stage="compositions", prompt=prompt, run_dir=run_dir, reasoning="high",
    )
    if not ok:
        update_artifact_meta(artifact_dir, parse_status="codex_failed")
        return {"error": stderr, "compositions": [], "clusters": clusters}

    def _repair(p: str) -> tuple[bool, str, str, Path]:
        return run_codex_with_artifacts(
            stage="compositions_repair", prompt=p, run_dir=run_dir, reasoning="high",
        )

    result = parse_or_repair(stdout, CompositionsOutput, repair_call=_repair)
    update_artifact_meta(
        artifact_dir,
        parse_status=result.status,
        validation_errors=result.errors,
    )
    if result.value is None:
        return {"error": "schema_validation_failed", "compositions": [], "clusters": clusters}
    out = result.value.model_dump()
    out["clusters"] = clusters
    return out


def _corpus_full(papers: list[dict], abstract_words: int | None = None) -> str:
    """Render full corpus for the compose prompt. With `abstract_words` set, trims
    each abstract to first N words to control prompt size on large corpora."""
    lines = []
    for i, p in enumerate(papers, 1):
        pid = p.get("paper_id", f"P{i}")
        lines.append(f"{pid}: [{p.get('year', '?')}] {p.get('title', '?')}")
        topic = p.get("source_topic", "?")
        lines.append(f"   topic: {topic}")
        abstract = p.get("abstract", "") or ""
        if abstract_words is not None:
            words = abstract.split()
            if len(words) > abstract_words:
                abstract = " ".join(words[:abstract_words]) + "..."
        lines.append(f"   abstract: {abstract}")
        lines.append("")
    return "\n".join(lines)


def _compose_candidate_ids(
    clusters: dict[str, list[str]],
    scores_by_id: dict[str, dict],
    max_papers_per_cluster: int = COMPOSE_MAX_PAPERS_PER_CLUSTER,
) -> set[str]:
    """Return the bounded paper set that Stage 4 should inspect in detail.

    The pipeline can ingest and score broad corpora, but the composition prompt
    should only include top cluster candidates. Large runs previously sent the
    full validated corpus here, which caused Step 4 timeouts even though the
    scored cluster structure was already bounded.
    """
    candidate_ids: set[str] = set()
    for dim_name, pids in clusters.items():
        if not pids:
            continue

        def cluster_score(pid):
            return (scores_by_id.get(pid, {}).get("dimensions", {}) or {}).get(
                dim_name, {}
            ).get("score", 0) or 0

        ranked = sorted(pids, key=lambda pid: -cluster_score(pid))
        candidate_ids.update(ranked[:max_papers_per_cluster])
    return candidate_ids


def cluster_papers_by_dimension(scores: list[dict],
                                dimensions: list[dict],
                                min_dim_score: int = 3) -> dict[str, list[str]]:
    """Group papers by their primary dimension (the one they score highest on,
    score >= min_dim_score). A paper can appear in multiple clusters if it ties.
    Returns {dimension_name: [paper_id, ...]}.

    This is the bounded-search trick for large corpora: instead of asking the LLM
    to consider all C(n, k) subsets, we provide a clustered structure and ask for
    compositions spanning clusters.
    """
    clusters: dict[str, list[str]] = {d["name"]: [] for d in dimensions}
    for s in scores:
        pid = s.get("paper_id")
        if not pid:
            continue
        dims = s.get("dimensions", {}) or {}
        # Find this paper's top dimension(s)
        if not dims:
            continue
        max_score = max((d.get("score", 0) or 0) for d in dims.values())
        if max_score < min_dim_score:
            continue
        for dim_name, d in dims.items():
            if (d.get("score", 0) or 0) == max_score and dim_name in clusters:
                clusters[dim_name].append(pid)
    return clusters


def _clusters_block(clusters: dict[str, list[str]],
                    papers_by_id: dict[str, dict],
                    scores_by_id: dict[str, dict],
                    max_papers_per_cluster: int = 15) -> str:
    """Render the dimension clusters as a prompt block. Show paper id, title,
    score on the cluster's dimension, and top contribution line."""
    lines = []
    for dim_name, pids in clusters.items():
        if not pids:
            continue
        lines.append(f"=== CLUSTER: {dim_name} ({len(pids)} papers) ===")
        # Sort by score on this dimension, descending
        def cluster_score(pid):
            return (scores_by_id.get(pid, {}).get("dimensions", {}) or {}).get(
                dim_name, {}
            ).get("score", 0) or 0
        ranked = sorted(pids, key=lambda p: -cluster_score(p))[:max_papers_per_cluster]
        for pid in ranked:
            paper = papers_by_id.get(pid, {})
            score = cluster_score(pid)
            score_entry = (scores_by_id.get(pid, {}).get("dimensions", {}) or {}).get(
                dim_name, {}
            )
            contrib = score_entry.get("contribution", "")
            lines.append(f"  {pid} [{dim_name}={score}] {paper.get('title', '')[:80]}")
            if contrib:
                lines.append(f"      → {contrib}")
        if len(pids) > max_papers_per_cluster:
            lines.append(f"  ... and {len(pids) - max_papers_per_cluster} more")
        lines.append("")
    return "\n".join(lines)


# ----------------------------------------------------------------------------
# Verification — Levels 1, 2, 3
# ----------------------------------------------------------------------------


CONSISTENCY_PROMPT_TEMPLATE = """You are auditing a proposed compositional synthesis
for internal contradictions.

COMPOSITION CLAIM:
{claim}

COMPATIBILITY ARGUMENT:
{compat}

CONSTITUENT PAPERS (verbatim source material):
{papers_block}

YOUR TASK
Check whether the composition is internally consistent. Identify any direct contradictions:
- Parameter conflicts (e.g. paper A operates at 80°C, paper B requires 200°C)
- Assumption clashes (e.g. paper A assumes pure CO2, paper B assumes flue gas)
- Methodological incompatibilities

Be conservative — only flag GENUINE contradictions, not stylistic differences or
gaps in the abstracts.

OUTPUT — JSON ONLY:
{{
  "consistent": <true|false>,
  "issues": [
    {{"severity": "<critical|moderate|minor>", "detail": "<one line>", "paper_refs": ["P3", ...]}},
    ...
  ]
}}
"""


def verify_consistency(composition: dict, papers_by_id: dict[str, dict], run_dir: Path) -> dict:
    papers_block = []
    for p in composition.get("papers", []):
        pid = p.get("paper_id")
        paper = papers_by_id.get(pid)
        if not paper:
            continue
        papers_block.append(f"{pid}: {paper.get('title', '')}")
        papers_block.append(f"   {paper.get('abstract', '')}")
        papers_block.append("")
    prompt = CONSISTENCY_PROMPT_TEMPLATE.format(
        claim=composition.get("synthesis_claim", ""),
        compat=composition.get("compatibility_argument", ""),
        papers_block="\n".join(papers_block),
    )
    cid = composition.get("id", "unknown")
    stage = f"verify_l1_{cid}"
    ok, stdout, stderr, artifact_dir = run_codex_with_artifacts(
        stage=stage, prompt=prompt, run_dir=run_dir, reasoning="medium",
    )
    if not ok:
        update_artifact_meta(artifact_dir, parse_status="codex_failed")
        return {"level": 1, "passed": False, "error": stderr}

    def _repair(p: str) -> tuple[bool, str, str, Path]:
        return run_codex_with_artifacts(
            stage=stage + "_repair", prompt=p, run_dir=run_dir, reasoning="medium",
        )

    result = parse_or_repair(stdout, L1Item, repair_call=_repair)
    update_artifact_meta(
        artifact_dir,
        parse_status=result.status,
        validation_errors=result.errors,
    )
    if result.value is None:
        return {"level": 1, "passed": False, "error": "schema_validation_failed"}
    return {
        "level": 1,
        "level_name": "internal_consistency",
        "passed": result.value.consistent,
        "issues": result.value.issues,
    }


def verify_source_grounding(composition: dict, papers_by_id: dict[str, dict]) -> dict:
    """Each cited paper must be in the corpus, and each supporting_quote must be
    a substring of that paper's abstract (after normalization)."""
    def normalize(s: str) -> str:
        s = s.lower()
        s = re.sub(r"[^a-z0-9\s]", " ", s)
        s = re.sub(r"\s+", " ", s).strip()
        return s

    results = []
    all_grounded = True
    for p in composition.get("papers", []):
        pid = p.get("paper_id")
        # Prefer the canonical prompt-shaped field; fall back to the legacy
        # short name so historical state files reprocess cleanly.
        quote = (p.get("supporting_quote") or p.get("quote") or "").strip()
        paper = papers_by_id.get(pid)
        if not paper:
            results.append({"paper_id": pid, "status": "PAPER_NOT_IN_CORPUS"})
            all_grounded = False
            continue
        abstract_norm = normalize(paper.get("abstract", ""))
        quote_norm = normalize(quote)
        if not quote_norm:
            results.append({"paper_id": pid, "status": "NO_QUOTE_PROVIDED"})
            all_grounded = False
        elif quote_norm in abstract_norm:
            results.append({"paper_id": pid, "status": "VERBATIM_MATCH"})
        else:
            # Token overlap fallback (catches lightly paraphrased quotes)
            quote_tokens = set(quote_norm.split())
            abs_tokens = set(abstract_norm.split())
            if quote_tokens:
                overlap = len(quote_tokens & abs_tokens) / len(quote_tokens)
            else:
                overlap = 0.0
            if overlap >= 0.7:
                results.append({"paper_id": pid, "status": "PARAPHRASED_MATCH",
                                "overlap": round(overlap, 2)})
            else:
                results.append({"paper_id": pid, "status": "QUOTE_NOT_IN_ABSTRACT",
                                "overlap": round(overlap, 2)})
                all_grounded = False
    return {
        "level": 2,
        "level_name": "source_grounding",
        "passed": all_grounded,
        "per_paper": results,
    }


PREDICTIONS_PROMPT_TEMPLATE = """You are reclassifying the objections raised by an
adversarial reviewer of a compositional synthesis. The goal of the work is to find
compositions that, when prototyped, BEAT THE RELEVANT BASELINES on the realistic
inputs the problem actually cares about. So the only objections that matter are
ones that would stop that from happening. Classify each objection as:

  - FUNDAMENTAL (disqualifying): even after charitably narrowing any overbroad or
    "universal/arbitrary/always" phrasing down to the realistic scope the problem
    targets, the composition STILL cannot beat the relevant baselines — because it
    is logically/physically impossible, contradicts a theorem, or violates a
    conservation/counting law in a way that bites on the realistic inputs. The bar
    is: "no experiment on realistic inputs could make this competitive."

  - EMPIRICAL (not disqualifying): the objection identifies an untested prediction,
    an engineering risk, or an overbroad claim that can be NARROWED and then tested.
    It does not, by itself, prevent the composition from beating baselines on
    realistic inputs.

TWO HARD RULES — apply them before calling anything FUNDAMENTAL:

  1. BASELINE-RELATIVE. If the SAME objection applies equally to the incumbent
     baselines/state-of-the-art, it is NOT fundamental — it cannot explain why this
     composition would lose to them. (Example: "a lossless compressor cannot shrink
     every arbitrary input — pigeonhole" is TRUE but applies to gzip/zstd/brotli
     too; every compressor expands on random data. It says nothing about beating
     baselines on typical inputs, so it is EMPIRICAL/non-disqualifying. Mark the
     universal CLAIM as needing narrowing, not the composition as refuted.)

  2. CHARITABLE NARROWING. If the objection only refutes an overbroad phrasing
     ("works for ALL inputs", "universally optimal"), narrow the claim to the
     realistic distribution the problem names and re-judge. If the narrowed claim
     could still beat baselines, the objection is EMPIRICAL.

When in doubt, mark EMPIRICAL. FUNDAMENTAL is reserved for objections that survive
both rules above.

PROBLEM: {problem}

COMPOSITION CLAIM:
{claim}

ADVERSARIAL CRITIQUE (from previous L3 pass — each item is an objection):
{l3_block}

YOUR TASK
For EACH objection above:
  1. Determine `applies_to_baselines_too` (would incumbents fail this identically?).
  2. Determine `narrowed_claim_survives` (after narrowing any universal phrasing to
     the realistic scope, could the composition still beat baselines?).
  3. Classify as FUNDAMENTAL only if BOTH are false; otherwise EMPIRICAL.
  4. If EMPIRICAL: state the specific test that would settle it (method, tooling,
     effort, predicted_outcome_under_composition).
  5. If FUNDAMENTAL: state why no experiment on realistic inputs can rescue it, and
     confirm it does not also sink the baselines.
  6. Confidence in classification (1-10).

OUTPUT — JSON ONLY:
{{
  "objection_classifications": [
    {{
      "objection_summary": "<one line — restate the objection>",
      "type": "FUNDAMENTAL" | "EMPIRICAL",
      "applies_to_baselines_too": <true|false>,
      "narrowed_claim_survives": <true|false>,
      "test": {{
        "method": "<short>",
        "tooling": "<specific>",
        "effort": "<estimate>",
        "predicted_outcome_under_composition": "<one line>"
      }},
      "fundamental_reason": "<if FUNDAMENTAL: why no realistic-input test can rescue it AND why it does not also sink the baselines>",
      "confidence": <1-10>
    }},
    ...
  ],
  "summary": {{
    "total_objections": N,
    "fundamental_count": <count of FUNDAMENTAL (disqualifying) objections only>,
    "empirical_count": N,
    "net_testability_score": <fundamental_count == 0 ? "PURELY_TESTABLE" : empirical_count / total_objections>,
    "verdict_for_unsolved_problem": "<one sentence: is this composition worth prototyping to see if it beats baselines on realistic inputs?>"
  }}
}}
"""


def extract_testable_predictions(composition: dict,
                                 papers_by_id: dict[str, dict],
                                 problem: str,
                                 *,
                                 run_dir: Path) -> dict:
    """L4 pass: classify each L3 objection as FUNDAMENTAL or EMPIRICAL.

    FUNDAMENTAL means *disqualifying for the practical metric*: even after
    charitably narrowing any universal/overbroad claim, the composition still
    cannot beat the relevant baselines on realistic inputs. Objections that apply
    equally to the baselines (e.g. the pigeonhole "can't compress all inputs"
    bound, which hits gzip/zstd/brotli too) or that only refute an overbroad
    phrasing are EMPIRICAL — not disqualifying. This bar was tightened after the
    gate-precision series found L4 mis-flagging C2 (the project's best
    composition) FUNDAMENTAL on a pigeonhole objection
    (findings/short-string-c2-rejected-verification-20260529.md).
    """
    l3 = composition.get("verification", {}).get("L3", {})
    attempts = l3.get("falsification_attempts", []) or []
    if not attempts:
        return {"level": 4, "level_name": "predictions_and_tests",
                "skipped": "no L3 objections to classify",
                "objection_classifications": [],
                "summary": {"total_objections": 0,
                            "fundamental_count": 0,
                            "empirical_count": 0}}

    # Build a compact L3 block from the falsification_attempts
    l3_lines = []
    for i, a in enumerate(attempts, 1):
        angle = a.get("angle", "(unnamed)")
        arg = a.get("argument", "")
        would_invalidate = a.get("would_invalidate", False)
        marker = "[WOULD INVALIDATE]" if would_invalidate else "[challenge]"
        l3_lines.append(f"{i}. {marker} {angle}: {arg}")

    prompt = PREDICTIONS_PROMPT_TEMPLATE.format(
        problem=problem,
        claim=composition.get("synthesis_claim", ""),
        l3_block="\n".join(l3_lines),
    )
    cid = composition.get("id", "unknown")
    stage = f"verify_l4_{cid}"
    ok, stdout, stderr, artifact_dir = run_codex_with_artifacts(
        stage=stage, prompt=prompt, run_dir=run_dir, reasoning="high",
    )
    if not ok:
        update_artifact_meta(artifact_dir, parse_status="codex_failed")
        return {"level": 4, "level_name": "predictions_and_tests",
                "error": stderr,
                "objection_classifications": [],
                "summary": {}}

    def _repair(p: str) -> tuple[bool, str, str, Path]:
        return run_codex_with_artifacts(
            stage=stage + "_repair", prompt=p, run_dir=run_dir, reasoning="high",
        )

    result = parse_or_repair(stdout, L4Verdict, repair_call=_repair)
    update_artifact_meta(
        artifact_dir,
        parse_status=result.status,
        validation_errors=result.errors,
    )
    if result.value is None:
        return {"level": 4, "level_name": "predictions_and_tests",
                "error": "schema_validation_failed",
                "objection_classifications": [],
                "summary": {}}
    return result.value.model_dump()


ADVERSARIAL_PROMPT_TEMPLATE = """You are an adversarial reviewer trying to FALSIFY a
proposed compositional synthesis. Your job is to find the strongest reason this
composition does NOT work, not to be agreeable.

PROBLEM: {problem}

PROPOSED COMPOSITION:
{claim}

COMPATIBILITY ARGUMENT BY ORIGINAL AUTHOR:
{compat}

CONSTITUENT PAPERS:
{papers_block}

YOUR TASK
Attempt to falsify this composition. Specifically look for:
- Trivial / well-known combinations (this is not a discovery, it's folklore)
- Hidden incompatibilities the original argument glossed over
- Missing components — does the composition actually solve the stated problem, or does
  it leave a critical step unaddressed?
- Cited paper claims that don't hold up (the paper says something weaker than the
  composition implies)
- Generalization errors — paper conclusions held in specific conditions not in the composition

VERDICT DEFINITIONS — these match L4's FUNDAMENTAL vs EMPIRICAL axis so the two
stages stay calibrated:

  - "sound"   : after honest scrutiny no falsification angle survives. Rare and
                informative when it happens.
  - "weak"    : the composition has gaps, untested premises, partial coverage, or
                "this hasn't been demonstrated end-to-end" issues — i.e. any
                objection that a definitive experiment, computation, or
                measurement could in principle settle. "Not novel," "incremental
                over prior work," "boundary handling is unclear," "needs an
                ablation" are all WEAK, not invalid.
  - "invalid" : the composition is logically or physically impossible. The
                constituent papers' assumptions directly contradict each other,
                the synthesis claim violates a theorem the cited papers prove,
                conservation laws forbid it, or one of the papers explicitly
                refutes what the composition is asserting. No experiment can
                rescue this; pursuing it cannot succeed under known math/physics.

CALIBRATION RULE — before returning "invalid", pause and ask:

    "Would a definitive experiment change my mind?"

If yes (e.g. a benchmark could plausibly show this combination beats baselines,
or a calculation could resolve the disputed regime), the verdict is WEAK.
"Invalid" is reserved for objections no test can rescue.

Be ruthless about identifying weaknesses, but accurate about the verdict — most
critique-worthy compositions are WEAK, not INVALID, and the bar for INVALID is
"no experiment can rescue this," not "this looks weak."

OUTPUT — JSON ONLY:
{{
  "verdict": "<sound|weak|invalid>",
  "verdict_reason": "<one paragraph; if 'invalid', name the theorem, contradiction, or impossibility>",
  "falsification_attempts": [
    {{"angle": "<short label>", "argument": "<one paragraph>", "would_invalidate": <true|false>}},
    ...
  ]
}}
"""


def verify_adversarial(composition: dict,
                       papers_by_id: dict[str, dict],
                       problem: str,
                       *,
                       run_dir: Path) -> dict:
    papers_block = []
    for p in composition.get("papers", []):
        pid = p.get("paper_id")
        paper = papers_by_id.get(pid)
        if not paper:
            continue
        papers_block.append(f"{pid}: {paper.get('title', '')}")
        papers_block.append(f"   {paper.get('abstract', '')}")
        papers_block.append("")
    prompt = ADVERSARIAL_PROMPT_TEMPLATE.format(
        problem=problem,
        claim=composition.get("synthesis_claim", ""),
        compat=composition.get("compatibility_argument", ""),
        papers_block="\n".join(papers_block),
    )
    cid = composition.get("id", "unknown")
    stage = f"verify_l3_{cid}"
    ok, stdout, stderr, artifact_dir = run_codex_with_artifacts(
        stage=stage, prompt=prompt, run_dir=run_dir, reasoning="high",
    )
    if not ok:
        update_artifact_meta(artifact_dir, parse_status="codex_failed")
        return {"level": 3, "passed": False, "error": stderr}

    def _repair(p: str) -> tuple[bool, str, str, Path]:
        return run_codex_with_artifacts(
            stage=stage + "_repair", prompt=p, run_dir=run_dir, reasoning="high",
        )

    result = parse_or_repair(stdout, L3Verdict, repair_call=_repair)
    update_artifact_meta(
        artifact_dir,
        parse_status=result.status,
        validation_errors=result.errors,
    )
    if result.value is None:
        return {"level": 3, "passed": False, "error": "schema_validation_failed"}
    verdict = result.value.verdict
    return {
        "level": 3,
        "level_name": "adversarial_check",
        "passed": verdict == "sound",
        "verdict": verdict,
        "verdict_reason": result.value.verdict_reason,
        "falsification_attempts": getattr(result.value, "falsification_attempts", []),
    }


# ----------------------------------------------------------------------------
# Corpus persistence helpers
# ----------------------------------------------------------------------------


def _reconcile_verdict(v3: dict | None, v4: dict | None) -> dict:
    """Combine independent L3 (adversarial) and L4 (testability) outputs into a
    single combined verdict per composition.

    L3 and L4 are independent prompts on different framings of the same
    composition. They can disagree:
      - L3=invalid + L4=PURELY_TESTABLE      (observed on TSP C3)
      - L3=sound   + L4 reports fundamentals (L4 found a refutation L3 missed)
      - either stage missing/errored

    The reconciled status keys downstream behavior:
      ``actionable``      — L3=weak  + L4=PURELY_TESTABLE (recipe gate fires)
      ``sound``           — L3=sound (with any L4; sound is rare and informative)
      ``weak_untestable`` — L3=weak  + L4 has fundamental objections
      ``invalid``         — L3=invalid (L3's refutation wins; L4 ignored for gating)
      ``incomplete``     — either stage missing or errored

    ``l3_l4_consistent`` is False when L3 and L4 tell different stories — the
    dashboard surfaces these so the user can audit by hand.
    """
    l3 = v3 or {}
    l4 = v4 or {}
    l3_verdict = (l3.get("verdict") or "").lower()
    l4_summary = l4.get("summary") or {}
    l4_score = l4_summary.get("net_testability_score")
    fundamental = l4_summary.get("fundamental_count", 0) or 0

    # Detect missing stages (either errored upstream).
    if not l3_verdict or "error" in l3:
        return {
            "status": "incomplete",
            "l3_l4_consistent": False,
            "reason": "L3 stage did not produce a verdict",
        }
    if not l4_summary or "error" in l4:
        # We can still emit a status from L3 alone, but downstream gating
        # requires L4 to decide actionability.
        if l3_verdict == "invalid":
            return {
                "status": "invalid",
                "l3_l4_consistent": True,
                "reason": "L3=invalid; L4 missing",
            }
        return {
            "status": "incomplete",
            "l3_l4_consistent": False,
            "reason": "L4 stage did not produce a summary",
        }

    if l3_verdict == "invalid":
        # L3's refutation wins. Inconsistency surfaces when L4 marked
        # everything testable despite L3 finding a fatal flaw.
        consistent = (l4_score != "PURELY_TESTABLE") or fundamental > 0
        return {
            "status": "invalid",
            "l3_l4_consistent": consistent,
            "reason": (
                "L3 found a fundamental refutation"
                + ("" if consistent
                   else "; L4 disagreed by labelling every objection empirical")
            ),
        }

    if l3_verdict == "sound":
        consistent = fundamental == 0
        return {
            "status": "sound",
            "l3_l4_consistent": consistent,
            "reason": (
                "L3=sound"
                + ("" if consistent
                   else f"; L4 found {fundamental} fundamental objection(s) L3 may have missed")
            ),
        }

    # L3 == "weak" — actionability depends on L4's testability assessment.
    if l4_score == "PURELY_TESTABLE":
        return {
            "status": "actionable",
            "l3_l4_consistent": True,
            "reason": "L3=weak and L4=PURELY_TESTABLE; recipe-eligible",
        }
    # Any other L4 outcome under weak → weak but not testable as-is.
    return {
        "status": "weak_untestable",
        "l3_l4_consistent": True,
        "reason": (
            f"L3=weak; L4 reports {fundamental} fundamental objection(s) "
            f"(net_testability_score={l4_score})"
        ),
    }


def _priority_score(c: dict) -> float:
    """Rank a composition for *verification effort* — prioritize, do not veto.

    Evidence (findings/outcome-calibration-2026-05-29.md + the gate-precision
    series): among the pre-registration signals, LLM ``confidence`` is the only
    one that tracked empirical outcome. L3's verdict AND L4's ``purely_testable``
    both over-fire — 2 of 3 rejected compositions verified were winners, including
    the project's best (C2, which was conf=6 AND purely_testable=False). So the
    score deliberately uses NEITHER L3 verdict nor purely_testable: including
    purely_testable actively buried C2. We rank by confidence (the only weakly
    predictive signal), L1/L2 (internal consistency + source grounding — honest
    checks that do not over-fire), and dimension coverage (completeness). Because
    these signals are weak, the real protection is a generous top-K, not a clever
    score — prioritize, then test broadly.
    """
    ver = c.get("verification") or {}
    conf = (c.get("confidence") or 0) / 10.0
    dims = len(c.get("covered_dimensions") or [])
    l1 = 1.0 if (ver.get("L1") or {}).get("passed") else 0.0
    l2 = 1.0 if (ver.get("L2") or {}).get("passed") else 0.0
    return 3.0 * conf + 0.5 * min(dims, 6) + 1.0 * l2 + 0.5 * l1


def _assign_verify_priority(
    compositions: list[dict], top_k: int = DEFAULT_RECIPE_TOP_K
) -> None:
    """Score each composition and flag the top_k as ``verify_recommended``.

    Implements *prioritize, don't veto*: verification recipes and the
    recommended-to-test flag go to the highest-priority compositions regardless
    of L3/L4 verdict, instead of only the L3=weak+PURELY_TESTABLE "actionable"
    set — which buried winners (see the gate-precision findings). The descriptive
    ``reconciled.status`` is left untouched (the dashboard still shows it).
    """
    scored: list[tuple[float, dict]] = []
    for c in compositions:
        rec = c.setdefault("verification", {}).setdefault("reconciled", {})
        score = _priority_score(c)
        rec["priority_score"] = round(score, 4)
        rec["verify_recommended"] = False
        scored.append((score, c))
    # Rank by score desc; deterministic tie-break on confidence then id.
    scored.sort(key=lambda sc: (-sc[0], -(sc[1].get("confidence") or 0), sc[1].get("id", "")))
    for _, c in scored[: max(0, top_k)]:
        c["verification"]["reconciled"]["verify_recommended"] = True


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


def _resume_load(out_dir: Path, name: str) -> Any:
    """Load a persisted per-stage artifact for ``--resume``.

    Returns None when the artifact is absent or corrupt, in which case the
    stage runs fresh. Per-stage JSON files double as resume checkpoints —
    a crashed run re-pays only the stages that never finished.
    """
    p = out_dir / name
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError:
        return None


# ----------------------------------------------------------------------------
# Orchestrator
# ----------------------------------------------------------------------------


def _pack_context(pack_id: str) -> str:
    """Render a computable pack as prompt context for mechanism-first composition.

    Injecting the pack's declared failure modes and gotchas into the dimension
    and composition prompts lets the model target what the destination SOTA
    actually lacks, instead of concatenating plausible components.
    """
    from pipeline.computable import load_pack
    pack = load_pack(pack_id)
    shape = pack.pack.problem_shape
    strong = next(
        (b.comparator for b in pack.sota.bar_levels
         if b.level == "strong_reproducible" and b.available), "")
    lines = [
        "",
        "DOMAIN PACK CONTEXT (the executable verification contract candidates will face):",
        f"  objective: {shape.objective}",
        f"  home-field SOTA bar: {strong}",
        "  constraints:",
        *[f"    - {c}" for c in shape.constraints],
        "  KNOWN FAILURE MODES of candidates in this domain — a strong",
        "  composition names which of these its transferring structure addresses:",
        *[f"    - {f}" for f in shape.failure_modes],
    ]
    if pack.gotchas.strip():
        lines.append("  domain gotchas:")
        lines.extend(f"    {ln}" for ln in pack.gotchas.strip().splitlines())
    lines.append("")
    return "\n".join(lines)


def compose_one(
    topic_list_path: Path | None,
    problem: str,
    papers_per_topic: int = DEFAULT_PAPERS_PER_TOPIC,
    max_papers: int = DEFAULT_MAX_PAPERS,
    year_min: int = DEFAULT_POST_CUTOFF,
    target_compositions: int = DEFAULT_TARGET_COMPOSITIONS,
    dry_run: bool = False,
    relevance_threshold: float = 0.30,
    max_fail_rate: float = 0.30,
    min_gate_papers: int = DEFAULT_MIN_GATE_PAPERS,
    recipe_top_k: int = DEFAULT_RECIPE_TOP_K,
    skip_gate: bool = False,
    strict_gate: bool = True,
    with_recipes: bool = False,
    with_novelty: bool = False,
    relevance_mode: str = "auto",
    strict_scoring: bool = True,
    search_queries: list[str] | None = None,
    papers_per_query: int = DEFAULT_PAPERS_PER_QUERY,
    computable_pack: str | None = None,
    resume_run_id: str | None = None,
) -> dict:
    """Run the full compositional pipeline for a single problem statement.

    The corpus comes from either a topic list (topic-sampled mode) or
    `search_queries` (paper-direct mode for problems narrower than any
    OpenAlex topic). Exactly one source must be provided.

    ``resume_run_id`` resumes a crashed run: per-stage artifacts already on
    disk (validated corpus, dimensions.json, scores.json,
    compositions_raw.json, verify_progress.json) are reused instead of
    re-paying network and codex spend; only unfinished stages run.
    """
    resume = bool(resume_run_id)
    if not resume and (topic_list_path is None) == (not search_queries):
        raise ValueError("provide exactly one of topic_list_path or search_queries")
    if resume and dry_run:
        raise ValueError("resume_run_id and dry_run are mutually exclusive")
    # Fail fast on a bad pack id, before any network or codex spend.
    pack_context = _pack_context(computable_pack) if computable_pack else ""
    if resume:
        run_id = resume_run_id
        out_dir = COMPOSE_DIR / run_id
        if not out_dir.is_dir():
            raise ValueError(f"cannot resume {run_id!r}: {out_dir} does not exist")
    else:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        out_dir = COMPOSE_DIR / run_id
        out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== compose run {run_id}{' (resume)' if resume else ''} ===")
    print(f"problem: {problem}")
    if search_queries:
        print(f"paper_search: {len(search_queries)} queries")
    elif topic_list_path:
        print(f"topic_list: {topic_list_path}")
    print(f"dry_run: {dry_run}\n")

    # Step 1: corpus
    print("Step 1: assembling corpus")
    resumed_corpus = False
    corpus: list[dict] = []
    if resume:
        persisted = _resume_load(out_dir, "corpus_validated.json") \
            or _resume_load(out_dir, "corpus.json")
        if persisted:
            corpus = persisted
            resumed_corpus = True
            print(f"  resume: reusing validated corpus ({len(corpus)} papers)")
        elif (topic_list_path is None) == (not search_queries):
            raise ValueError(
                f"cannot resume {run_id!r}: no persisted corpus in {out_dir} "
                "and no corpus source given")
    if not resumed_corpus:
        if search_queries:
            corpus = assemble_corpus_from_search(search_queries,
                                                 papers_per_query=papers_per_query,
                                                 max_papers=max_papers,
                                                 year_min=year_min)
        else:
            assert topic_list_path is not None
            corpus = assemble_corpus(topic_list_path,
                                     papers_per_topic=papers_per_topic,
                                     max_papers=max_papers,
                                     year_min=year_min)
        # Re-index papers as P1..PN
        for i, p in enumerate(corpus, 1):
            p["paper_id"] = f"P{i}"
    papers_by_id = {p["paper_id"]: p for p in corpus}
    print(f"  corpus: {len(corpus)} papers from {len({p['source_topic'] for p in corpus})} topics")
    # Snapshot the raw corpus now; validated/legacy copies are written
    # after the Step 1.5 gate has run (or here, in dry-run mode).
    raw_corpus_snapshot = [dict(p) for p in corpus]  # shallow copy preserves pre-drop state

    state: dict[str, Any] = {
        "run_id": run_id,
        "problem": problem,
        "corpus_size": len(corpus),
        "topics_covered": sorted({p["source_topic"] for p in corpus}),
        "dry_run": dry_run,
    }

    if dry_run:
        _persist_corpus_artifacts(out_dir, raw_corpus=raw_corpus_snapshot, validated_corpus=corpus)
        # Emit a fake compositions block so the digest renderer can shape itself
        state["status"] = "dry_run_complete"
        (out_dir / "state.json").write_text(json.dumps(state, indent=2))
        print(f"\nDRY RUN — corpus saved to {out_dir}/corpus.json")
        print(f"  topics: {len(state['topics_covered'])}")
        for t in state["topics_covered"]:
            print(f"    - {t}")
        return state

    # Step 1.5: corpus quality gate
    if resumed_corpus:
        # The persisted corpus is post-gate; re-validating would re-pay the
        # relevance backend. Restore the saved gate report for the digest.
        print("\nStep 1.5: skipped (resume — corpus already validated)")
        saved_gate = _resume_load(out_dir, "corpus_gate.json")
        if saved_gate:
            state["corpus_gate"] = saved_gate
    elif not skip_gate:
        print("\nStep 1.5: corpus quality gate")
        report = validate_corpus(
            corpus, problem,
            relevance_threshold=relevance_threshold,
            max_fail_rate=max_fail_rate,
            strict=strict_gate and not dry_run,
            relevance_mode=relevance_mode,
            min_passed=min_gate_papers,
        )
        (out_dir / "corpus_gate.json").write_text(json.dumps(report.as_dict(), indent=2))
        print_report(report)
        if report.aborted:
            state["status"] = "aborted_corpus_gate"
            state["corpus_gate"] = report.as_dict()
            (out_dir / "state.json").write_text(json.dumps(state, indent=2))
            print(f"\nAborted at Step 1.5. Report saved to {out_dir}/corpus_gate.json")
            return state
        if report.drop_ids:
            corpus = [p for p in corpus if p["paper_id"] not in report.drop_ids]
            # Re-index survivors as P1..PN
            for i, p in enumerate(corpus, 1):
                p["paper_id"] = f"P{i}"
            papers_by_id = {p["paper_id"]: p for p in corpus}
            state["corpus_size"] = len(corpus)
            state["topics_covered"] = sorted({p["source_topic"] for p in corpus})
            print(f"  dropped {len(report.drop_ids)} papers; "
                  f"continuing with {len(corpus)}")
        state["corpus_gate"] = report.as_dict()
        state["relevance_skipped"] = "relevance" in report.skipped_checks
        _persist_corpus_artifacts(out_dir, raw_corpus=raw_corpus_snapshot, validated_corpus=corpus)
    else:
        print("\nStep 1.5: corpus quality gate SKIPPED (--skip-gate)")
        _persist_corpus_artifacts(out_dir, raw_corpus=raw_corpus_snapshot, validated_corpus=corpus)

    # Step 2: dimensions
    print("\nStep 2: extracting problem dimensions (Codex)")
    dims_result = _resume_load(out_dir, "dimensions.json") if resume else None
    if dims_result and dims_result.get("dimensions"):
        print("  resume: reusing dimensions.json")
    else:
        dims_result = extract_dimensions(corpus, problem, run_dir=out_dir,
                                         pack_context=pack_context)
    dimensions = dims_result.get("dimensions", [])
    (out_dir / "dimensions.json").write_text(json.dumps(dims_result, indent=2))
    print(f"  {len(dimensions)} dimensions extracted")
    for d in dimensions:
        print(f"    - {d.get('name', '?')}: {d.get('description', '')[:80]}")
    if not dimensions:
        state["status"] = "no_dimensions_extracted"
        (out_dir / "state.json").write_text(json.dumps(state, indent=2))
        return state

    # Step 3: scoring
    print("\nStep 3: scoring papers on dimensions (Codex)")
    scores_result = _resume_load(out_dir, "scores.json") if resume else None
    if scores_result and scores_result.get("scores"):
        print("  resume: reusing scores.json")
    else:
        scores_result = score_papers(corpus, dimensions, problem, run_dir=out_dir)
    scores = scores_result.get("scores", [])
    (out_dir / "scores.json").write_text(json.dumps(scores_result, indent=2))
    coverage = scores_result.get("coverage", 0.0)
    empty_batches = scores_result.get("empty_batches", 0)
    n_batches = scores_result.get("n_batches", 0)
    print(f"  scored {len(scores)} papers (coverage {coverage:.0%}, "
          f"{empty_batches}/{n_batches} empty batches)")

    # Gate before Stage 4: a Stage 3 that returned no scores or fell below
    # the coverage floor must not silently feed Stage 4. Prior to this
    # gate, run 20260522-062501 generated 7 "compositions" from 0/95
    # scored papers. Under strict mode the run aborts cleanly so the
    # operator can investigate; --no-strict-scoring re-enables the legacy
    # behavior for debugging only.
    state["scoring_coverage"] = coverage
    state["scoring_empty_batches"] = empty_batches
    state["scoring_n_batches"] = n_batches
    if (not scores) or coverage < SCORING_MIN_COVERAGE or empty_batches > 0:
        reason = (
            "no_scores" if not scores else
            f"coverage_below_floor ({coverage:.0%} < {SCORING_MIN_COVERAGE:.0%})"
            if coverage < SCORING_MIN_COVERAGE else
            f"{empty_batches}/{n_batches} empty batches"
        )
        if strict_scoring:
            state["status"] = "insufficient_scoring_coverage"
            state["scoring_abort_reason"] = reason
            (out_dir / "state.json").write_text(json.dumps(state, indent=2))
            print(f"  ⚠️  scoring gate: aborting before Stage 4 ({reason})")
            print("  rerun with --no-strict-scoring to bypass for debugging")
            return state
        else:
            print(f"  ⚠️  scoring gate would have fired ({reason}); continuing "
                  "because --no-strict-scoring")

    # Step 4: composition finding
    print("\nStep 4: finding compositions (Codex)")
    comp_result = _resume_load(out_dir, "compositions_raw.json") if resume else None
    if comp_result and comp_result.get("compositions"):
        print("  resume: reusing compositions_raw.json")
    else:
        comp_result = find_compositions(corpus, dimensions, scores, problem,
                                        target=target_compositions, run_dir=out_dir,
                                        pack_context=pack_context)
    compositions = comp_result.get("compositions", [])
    (out_dir / "compositions_raw.json").write_text(json.dumps(comp_result, indent=2))
    print(f"  {len(compositions)} candidate compositions surfaced")

    # Step 5: verification
    print("\nStep 5: verifying compositions")
    # Per-composition checkpoint: the verification fan-out is the most
    # expensive phase (3 codex calls per composition), so each finished
    # composition is persisted immediately and skipped on resume.
    progress_path = out_dir / "verify_progress.json"
    verify_done: dict[str, dict] = ({} if not resume
                                    else _resume_load(out_dir, "verify_progress.json") or {})
    # Give every composition a unique id up front: verification results are
    # keyed by id in the progress checkpoint, and the LLM occasionally emits
    # duplicates or omits the field.
    _seen_ids: set[str] = set()
    for _i, c in enumerate(compositions, 1):
        if not c.get("id") or c["id"] in _seen_ids:
            c["id"] = f"C{_i}-auto"
        _seen_ids.add(c["id"])

    def _verify_one_comp(c: dict) -> dict:
        cid = c.get("id", "C?")
        print(f"  {cid}: {c.get('synthesis_claim', '')[:80]}...")
        v1 = verify_consistency(c, papers_by_id, run_dir=out_dir)
        v2 = verify_source_grounding(c, papers_by_id)
        v3 = verify_adversarial(c, papers_by_id, problem, run_dir=out_dir)
        c["verification"] = {"L1": v1, "L2": v2, "L3": v3}
        # L4: classify L3 objections as FUNDAMENTAL or EMPIRICAL
        v4 = extract_testable_predictions(c, papers_by_id, problem, run_dir=out_dir)
        c["verification"]["L4"] = v4
        # Reconcile L3 + L4 into a single combined verdict. L3 and L4 are
        # independent prompts that can disagree (e.g. L3=invalid while L4
        # marks every objection EMPIRICAL → PURELY_TESTABLE). The reconciled
        # status is what downstream code (the recipe gate, the dashboard)
        # should key off; raw L3/L4 stay in the verification dict for audit.
        c["verification"]["reconciled"] = _reconcile_verdict(v3, v4)
        # Stage 6 recipes are generated in a post-loop pass below: they go to the
        # top-K compositions by priority (prioritize, don't veto on L3/L4).
        # Stage 7 (Phase 3): novelty check (opt-in; hits external APIs).
        if with_novelty and (v3 or {}).get("verdict") != "invalid":
            from pipeline.novelty import check_novelty
            c["verification"]["novelty"] = check_novelty(
                c, state, run_dir=out_dir, with_search=True,
            )
        passed = [v["level_name"] for v in (v1, v2, v3) if v.get("passed")]
        # L4 doesn't have a binary pass/fail — it's a classifier. But we summarize.
        l4_summary = v4.get("summary") or {}
        c["verification_summary"] = {
            "levels_passed": passed,
            "all_passed": len(passed) == 3,
            "objections_fundamental": l4_summary.get("fundamental_count", 0),
            "objections_empirical": l4_summary.get("empirical_count", 0),
            "purely_testable": (l4_summary.get("fundamental_count", 0) == 0
                                and l4_summary.get("empirical_count", 0) > 0),
        }
        l4_msg = ""
        if l4_summary:
            f = l4_summary.get("fundamental_count", 0)
            e = l4_summary.get("empirical_count", 0)
            l4_msg = f"  L4 objections: {f} fundamental, {e} empirical"
        print(f"    L1 consistency: {'PASS' if v1.get('passed') else 'FAIL'}"
              f"  L2 grounding: {'PASS' if v2.get('passed') else 'FAIL'}"
              f"  L3 adversarial: {'PASS' if v3.get('passed') else 'FAIL'}"
              f"  ({v3.get('verdict', '?')})"
              f"{l4_msg}")
        return c

    for c in compositions:
        if c["id"] in verify_done:
            print(f"  {c['id']}: resume — already verified")
    pending = [c for c in compositions if c["id"] not in verify_done]
    progress_lock = threading.Lock()

    def _record_verified(c: dict) -> None:
        with progress_lock:
            verify_done[c["id"]] = c
            progress_path.write_text(json.dumps(verify_done, indent=2))

    # Compositions verify independently (3 codex calls each) — fan out.
    # The novelty check hits rate-limited external APIs, so it forces the
    # serial path. Results are re-ordered to match the composition list, so
    # output is identical to the serial loop.
    workers = 1 if with_novelty else min(_max_concurrency(), max(1, len(pending)))
    if pending and workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_verify_one_comp, c) for c in pending]
            for fut in as_completed(futures):
                _record_verified(fut.result())
    else:
        for c in pending:
            _record_verified(_verify_one_comp(c))
    verified_compositions = [verify_done[c["id"]] for c in compositions
                             if c["id"] in verify_done]

    # Stage 6: prioritize, don't veto. Score every composition and flag the
    # top-K for verification; recipes go to those — regardless of L3/L4 verdict,
    # because the gate-precision series showed L3=invalid / fund>0 bury winners.
    _assign_verify_priority(verified_compositions, top_k=recipe_top_k)
    if with_recipes:
        from pipeline.recipe import generate_recipe
        for c in verified_compositions:
            if c["verification"]["reconciled"].get("verify_recommended"):
                c["verification"]["recipe"] = generate_recipe(c, state, run_dir=out_dir)

    state["dimensions"] = dimensions
    state["compositions"] = verified_compositions
    state["status"] = "complete"
    usage = aggregate_llm_usage(out_dir)
    (out_dir / "llm_usage.json").write_text(json.dumps(usage, indent=2))
    state["llm_usage"] = {k: usage[k] for k in
                          ("calls", "duration_s",
                           "est_prompt_tokens", "est_output_tokens")}
    print(f"  llm usage: {usage['calls']} codex calls, "
          f"{usage['duration_s'] / 60:.1f} codex-min, "
          f"~{(usage['est_prompt_tokens'] + usage['est_output_tokens']) / 1000:.0f}k est tokens")
    (out_dir / "state.json").write_text(json.dumps(state, indent=2))
    print(f"\nSaved: {out_dir}")
    return state


def write_compose_digest(state: dict, out_path: Path) -> Path:
    problem = state.get("problem", "")
    compositions = state.get("compositions", [])
    lines = [
        f"# Compositional synthesis digest — {state.get('run_id', '?')}",
        "",
        "Generated by congruity v1 compose pipeline.",
        "",
        f"**Problem:** {problem}",
        "",
        f"- Corpus size: {state.get('corpus_size', '?')}",
        f"- Topics covered: {len(state.get('topics_covered', []))}",
        f"- Compositions surfaced: {len(compositions)}",
        f"- Compositions passing all 3 verification levels: "
        f"{sum(1 for c in compositions if (c.get('verification_summary') or {}).get('all_passed'))}",
        "",
        "## Compositions",
        "",
    ]
    # Sort by verification priority (prioritize, don't veto): the priority score
    # does not demote L3=invalid / fund>0, because those over-fire (see the
    # gate-precision findings). Falls back to computing the score if absent.
    def sort_key(c):
        rec = (c.get("verification") or {}).get("reconciled") or {}
        score = rec.get("priority_score")
        if score is None:
            score = _priority_score(c)
        return (-score, -(c.get("confidence") or 0), c.get("id", ""))
    for c in sorted(compositions, key=sort_key):
        cid = c.get("id", "?")
        ver = c.get("verification") or {}
        vs = c.get("verification_summary") or {}
        lines.extend([
            f"### {cid} — {len(c.get('papers', []))} papers — verified at: "
            f"{', '.join(vs.get('levels_passed', [])) or 'none'}",
            "",
            f"**Synthesis claim:** {c.get('synthesis_claim', '')}",
            "",
            f"**Compatibility argument:** {c.get('compatibility_argument', '')}",
            "",
            "**Constituent papers:**",
            "",
        ])
        for p in c.get("papers", []):
            lines.append(f"- **{p.get('paper_id')}** — role: {p.get('role')}")
            lines.append(f"  - dimensions: {', '.join(p.get('supporting_dimensions', []))}")
            quote = p.get("supporting_quote", "")
            lines.append(f"  - quote: _\"{quote[:200]}{'...' if len(quote) > 200 else ''}\"_")
        lines.extend([
            "",
            f"**Covered dimensions:** {', '.join(c.get('covered_dimensions', []))}",
            "",
            f"**Open questions:** {'; '.join(c.get('open_questions', [])) or 'none stated'}",
            "",
            f"**Codex confidence:** {c.get('confidence', '?')}/10 — _{c.get('confidence_reason', '')}_",
            "",
            "**Verification:**",
            "",
        ])
        l1 = ver.get("L1") or {}
        l2 = ver.get("L2") or {}
        l3 = ver.get("L3") or {}
        lines.append(f"- L1 internal consistency: **{'PASS' if l1.get('passed') else 'FAIL'}**")
        if l1.get("issues"):
            for iss in l1.get("issues", []):
                lines.append(f"  - {iss.get('severity', '?')}: {iss.get('detail', '')}")
        lines.append(f"- L2 source grounding: **{'PASS' if l2.get('passed') else 'FAIL'}**")
        for pr in l2.get("per_paper", []):
            lines.append(f"  - {pr.get('paper_id')}: {pr.get('status')}")
        lines.append(f"- L3 adversarial check: **{l3.get('verdict', '?').upper()}** — {l3.get('verdict_reason', '')}")
        for fa in l3.get("falsification_attempts", []):
            wi = "WOULD INVALIDATE" if fa.get("would_invalidate") else "challenge"
            lines.append(f"  - [{wi}] {fa.get('angle', '')}: {fa.get('argument', '')}")

        # L4 testability summary
        l4 = ver.get("L4") or {}
        l4s = l4.get("summary") or {}
        if l4s:
            f_count = l4s.get("fundamental_count", 0)
            e_count = l4s.get("empirical_count", 0)
            total = l4s.get("total_objections", f_count + e_count)
            verdict = l4s.get("verdict_for_unsolved_problem", "")
            badge = "**PURELY TESTABLE**" if (f_count == 0 and e_count > 0) else (
                "MIXED" if e_count > 0 else "MOSTLY FUNDAMENTAL"
            )
            lines.append("")
            lines.append(f"**L4 testability classification: {badge}** "
                         f"({e_count} empirical / {f_count} fundamental / {total} total)")
            if verdict:
                lines.append(f"_Investment recommendation: {verdict}_")
            lines.append("")
            # Show each classified objection
            lines.append("**Testable predictions implied by this composition:**")
            lines.append("")
            for oc in l4.get("objection_classifications", []):
                otype = oc.get("type", "?")
                summary = oc.get("objection_summary", "")
                if otype == "EMPIRICAL":
                    test = oc.get("test") or {}
                    method = test.get("method", "?")
                    tooling = test.get("tooling", "?")
                    effort = test.get("effort", "?")
                    predicted = test.get("predicted_outcome_under_composition", "")
                    lines.append(f"- **EMPIRICAL** — {summary}")
                    lines.append(f"  - test: {method} ({tooling}) — effort {effort}")
                    if predicted:
                        lines.append(f"  - predicted outcome if composition holds: _{predicted}_")
                elif otype == "FUNDAMENTAL":
                    reason = oc.get("fundamental_reason", "")
                    lines.append(f"- **FUNDAMENTAL** — {summary}")
                    if reason:
                        lines.append(f"  - why no test would settle it: {reason}")
                else:
                    lines.append(f"- (unclassified) {summary}")
            lines.append("")
        lines.append("---")
        lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))
    return out_path


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------


def reprocess_run_with_l3(
    run_dir: Path, *, with_recipes: bool = False,
    recipe_top_k: int = DEFAULT_RECIPE_TOP_K,
) -> dict:
    """Re-run L3 (adversarial) on an existing compose run with the current
    ADVERSARIAL_PROMPT_TEMPLATE.

    Use when the L3 prompt has been recalibrated and historical compositions
    need re-classification (e.g. the 2026-05-22 tightening that defines
    `invalid` as logically/physically impossible rather than "not novel").
    L1, L2, L4, and novelty results are preserved; only L3 + reconciled
    status + verification_summary fields driven by L3 are updated.

    Pass ``with_recipes=True`` to (re)generate recipes for compositions that
    newly clear to ``actionable``. Compositions that move OFF actionable
    have their stale recipe stripped regardless. Recipes are also kept up
    to date when at least one existing composition already carries one
    (back-compat for runs whose original ``--with-recipes`` produced one).
    """
    state_path = run_dir / "state.json"
    if not state_path.exists():
        raise FileNotFoundError(f"no state.json at {run_dir}")
    state = json.loads(state_path.read_text())
    compositions = state.get("compositions") or []
    problem = state.get("problem", "")
    corpus_path = run_dir / "corpus_validated.json"
    if not corpus_path.exists():
        corpus_path = run_dir / "corpus.json"
    corpus = json.loads(corpus_path.read_text()) if corpus_path.exists() else []
    papers_by_id = {p["paper_id"]: p for p in corpus if p.get("paper_id")}

    had_existing_recipe = any(
        "recipe" in (c.get("verification") or {}) for c in compositions
    )
    recipe_mode = with_recipes or had_existing_recipe

    print(f"Reprocessing L3 on {run_dir.name}: {len(compositions)} compositions")
    print(f"  with_recipes: {with_recipes} "
          f"(existing recipes detected: {had_existing_recipe})")
    transitions: list[str] = []
    for c in compositions:
        cid = c.get("id", "C?")
        ver = c.get("verification") or {}
        old_l3_verdict = ((ver.get("L3") or {}).get("verdict") or "?")
        old_reconciled = ((ver.get("reconciled") or {}).get("status") or "?")

        print(f"  {cid}: re-running L3...")
        v3_new = verify_adversarial(c, papers_by_id, problem, run_dir=run_dir)
        ver["L3"] = v3_new
        # L4 stays as-is. Recompute the reconciled verdict against the new L3.
        v4 = ver.get("L4")
        ver["reconciled"] = _reconcile_verdict(v3_new, v4)
        c["verification"] = ver

        new_l3_verdict = v3_new.get("verdict", "?")
        new_reconciled = ver["reconciled"]["status"]
        transitions.append(f"{cid}: L3 {old_l3_verdict}→{new_l3_verdict}, "
                           f"reconciled {old_reconciled}→{new_reconciled}")

    # Recipe re-evaluation (prioritize, don't veto): recipes go to the top-K by
    # priority, regardless of L3/L4 verdict; comps that fall out lose theirs.
    _assign_verify_priority(compositions, top_k=recipe_top_k)
    if recipe_mode:
        from pipeline.recipe import generate_recipe
        for c in compositions:
            ver = c["verification"]
            recommended = ver["reconciled"].get("verify_recommended")
            if recommended and "recipe" not in ver:
                ver["recipe"] = generate_recipe(c, state, run_dir=run_dir)
                print(f"    + recipe generated ({c.get('id', '?')}, top-{recipe_top_k})")
            elif not recommended and "recipe" in ver:
                ver.pop("recipe", None)
                print(f"    - recipe removed ({c.get('id', '?')}, below top-{recipe_top_k})")

    state["compositions"] = compositions
    state_path.write_text(json.dumps(state, indent=2))

    findings_dir = Path(__file__).resolve().parent.parent / "findings"
    digest_path = findings_dir / f"compose-{state['run_id']}-l3rerun.md"
    write_compose_digest(state, digest_path)
    print()
    print("Transitions:")
    for t in transitions:
        print(f"  {t}")
    print(f"\nUpdated state: {state_path}")
    print(f"New digest:    {digest_path}")
    return state


def reprocess_run_with_l4(run_dir: Path) -> dict:
    """Add L4 testability classification to an existing compose run that pre-dates
    the L4 stage. Loads compositions from state.json, runs extract_testable_predictions
    on each, persists updated state and re-renders the digest."""
    state_path = run_dir / "state.json"
    if not state_path.exists():
        raise FileNotFoundError(f"no state.json at {run_dir}")
    state = json.loads(state_path.read_text())
    compositions = state.get("compositions") or []
    problem = state.get("problem", "")
    # Prefer the post-gate corpus; fall back to legacy single-file layout.
    corpus_path = run_dir / "corpus_validated.json"
    if not corpus_path.exists():
        corpus_path = run_dir / "corpus.json"
    corpus = json.loads(corpus_path.read_text()) if corpus_path.exists() else []
    papers_by_id = {p["paper_id"]: p for p in corpus if p.get("paper_id")}

    print(f"Reprocessing {run_dir.name}: {len(compositions)} compositions")
    for c in compositions:
        cid = c.get("id", "C?")
        ver = c.get("verification") or {}
        if "L4" in ver and ver["L4"].get("objection_classifications"):
            print(f"  {cid}: L4 already present, skipping")
            continue
        print(f"  {cid}: running L4 testability classification...")
        v4 = extract_testable_predictions(c, papers_by_id, problem, run_dir=run_dir)
        ver["L4"] = v4
        c["verification"] = ver
        l4s = v4.get("summary") or {}
        f_count = l4s.get("fundamental_count", 0)
        e_count = l4s.get("empirical_count", 0)
        vs = c.get("verification_summary") or {}
        vs["objections_fundamental"] = f_count
        vs["objections_empirical"] = e_count
        vs["purely_testable"] = (f_count == 0 and e_count > 0)
        c["verification_summary"] = vs
        print(f"    {e_count} empirical / {f_count} fundamental objections")

    state["compositions"] = compositions
    state_path.write_text(json.dumps(state, indent=2))

    # Re-render digest
    findings_dir = Path(__file__).resolve().parent.parent / "findings"
    digest_path = findings_dir / f"compose-{state['run_id']}-with-l4.md"
    write_compose_digest(state, digest_path)
    print(f"\nUpdated state: {state_path}")
    print(f"New digest: {digest_path}")
    return state


FINDINGS_DIR = Path(__file__).resolve().parent.parent / "findings"


def add_compose_args(parser: argparse.ArgumentParser, *, required: bool = True) -> None:
    """Attach the canonical compose argument surface to ``parser``.

    Both ``pipeline.compose`` and ``pipeline.runner`` use this helper so the
    documented ``python -m pipeline compose`` and the module form
    ``python -m pipeline.compose`` share one source of truth. Without this
    unification, runner-level callers silently missed the recipe / novelty /
    relevance / scoring-strictness flags.

    ``required=False`` is for the module entry point, which routes through a
    subparser for the ``reprocess`` subcommand and validates ``--topic-list``
    / ``--problem`` only when no subcommand is given.
    """
    parser.add_argument(
        "--topic-list", type=Path, required=False,
        help="Path to a topic list JSON "
             "(e.g. data/pipeline/topic_lists/carbon_capture.json); "
             "either this or --paper-search is required",
    )
    parser.add_argument(
        "--problem", required=required,
        help="Problem statement / synthesis target",
    )
    parser.add_argument(
        "--paper-search", default=None,
        help="Paper-direct corpus mode: semicolon-separated OpenAlex search "
             "queries used instead of --topic-list. For problems narrower "
             "than any OpenAlex topic (results are relevance-ranked).",
    )
    parser.add_argument(
        "--computable-pack", default=None,
        help="Computable pack id whose problem shape, failure modes, and "
             "gotchas are injected into the dimension and composition "
             "prompts (mechanism-first composition).",
    )
    parser.add_argument("--papers-per-topic", type=int,
                        default=DEFAULT_PAPERS_PER_TOPIC)
    parser.add_argument("--papers-per-query", type=int,
                        default=DEFAULT_PAPERS_PER_QUERY,
                        help="Papers fetched per --paper-search query "
                             f"(default {DEFAULT_PAPERS_PER_QUERY})")
    parser.add_argument("--max-papers", type=int,
                        default=DEFAULT_MAX_PAPERS)
    parser.add_argument("--year-min", type=int,
                        default=DEFAULT_POST_CUTOFF)
    parser.add_argument(
        "--target", type=int, default=DEFAULT_TARGET_COMPOSITIONS,
        help="Number of compositions to surface",
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Skip Codex calls; validate corpus assembly only")
    parser.add_argument(
        "--resume", default=None, metavar="RUN_ID",
        help="Resume a crashed run: reuse per-stage artifacts already in "
             "data/pipeline/compositions/<RUN_ID>/ (validated corpus, "
             "dimensions, scores, compositions, per-composition verification) "
             "and run only the unfinished stages. Pass the same flags as the "
             "original invocation.",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--relevance-threshold", type=float, default=0.30,
        help="Cosine-similarity floor for Step 1.5 relevance check (default 0.30)",
    )
    parser.add_argument(
        "--max-fail-rate", type=float, default=0.30,
        help="Abort if Step 1.5 data-quality fail rate exceeds this fraction "
             "(default 0.30); relevance drops alone do not trigger this abort",
    )
    parser.add_argument(
        "--min-gate-papers", type=int, default=DEFAULT_MIN_GATE_PAPERS,
        help="Minimum number of papers that must survive Step 1.5 filtering "
             f"before downstream composition starts (default {DEFAULT_MIN_GATE_PAPERS})",
    )
    parser.add_argument(
        "--recipe-top-k", type=int, default=DEFAULT_RECIPE_TOP_K,
        help="Generate verification recipes for the top-K compositions by priority "
             f"score, not by L3/L4 verdict (default {DEFAULT_RECIPE_TOP_K}). "
             "Prioritize, don't veto: L3=invalid / fundamental objections do not "
             "exclude a composition from being recommended for testing.",
    )
    parser.add_argument(
        "--relevance-mode",
        choices=["auto", "openai", "anthropic", "codex", "none"],
        default="auto",
        help="Relevance backend for Step 1.5 (auto picks codex > anthropic > "
             "openai > none based on local auth / env keys)",
    )
    parser.add_argument(
        "--with-recipes", action="store_true", default=False,
        help="emit a verification recipe for each WEAK+PURELY_TESTABLE composition",
    )
    parser.add_argument(
        "--with-novelty", action="store_true", default=False,
        help="run post-L3 novelty checks (hits S2/OpenAlex; opt-in to avoid rate limits)",
    )
    parser.add_argument(
        "--skip-gate", action="store_true",
        help="Bypass Step 1.5 corpus quality gate entirely",
    )
    parser.add_argument(
        "--strict-gate", dest="strict_gate", action="store_true", default=True,
        help="Abort if Step 1.5 relevance check is skipped (default: on for non-dry-run)",
    )
    parser.add_argument(
        "--no-strict-gate", dest="strict_gate", action="store_false",
        help="Allow Step 1.5 to skip the relevance check (warning only)",
    )
    parser.add_argument(
        "--strict-scoring", dest="strict_scoring",
        action="store_true", default=True,
        help="Abort before Stage 4 if scoring coverage is below floor "
             "or any batch returned empty (default: on)",
    )
    parser.add_argument(
        "--no-strict-scoring", dest="strict_scoring", action="store_false",
        help="Continue past Stage 3 even with degraded scoring (debug only)",
    )


def run_compose_from_args(args: argparse.Namespace) -> int:
    """Run ``compose_one`` with kwargs derived from ``args`` and write the digest.

    Returns 0 on success (status ``complete`` or ``dry_run_complete``);
    nonzero when the run aborted at a quality gate (corpus or scoring) so
    scripted callers can detect failure without parsing state.json.
    """
    queries = None
    if getattr(args, "paper_search", None):
        queries = [q.strip() for q in args.paper_search.split(";") if q.strip()]
    resume_run_id = getattr(args, "resume", None)
    if not queries and not args.topic_list and not resume_run_id:
        raise SystemExit("compose: one of --topic-list / --paper-search is required")
    state = compose_one(
        topic_list_path=None if queries else args.topic_list,
        search_queries=queries,
        papers_per_query=getattr(args, "papers_per_query", DEFAULT_PAPERS_PER_QUERY),
        computable_pack=getattr(args, "computable_pack", None),
        problem=args.problem,
        papers_per_topic=args.papers_per_topic,
        max_papers=args.max_papers,
        year_min=args.year_min,
        target_compositions=args.target,
        dry_run=args.dry_run,
        relevance_threshold=args.relevance_threshold,
        max_fail_rate=args.max_fail_rate,
        min_gate_papers=args.min_gate_papers,
        recipe_top_k=args.recipe_top_k,
        skip_gate=args.skip_gate,
        with_recipes=args.with_recipes,
        with_novelty=args.with_novelty,
        relevance_mode=args.relevance_mode,
        strict_gate=args.strict_gate,
        strict_scoring=args.strict_scoring,
        resume_run_id=resume_run_id,
    )

    if not args.dry_run and state.get("status") == "complete":
        out_path = args.output if args.output else (
            FINDINGS_DIR / f"compose-{state['run_id']}.md"
        )
        write_compose_digest(state, out_path)
        print(f"\nDigest: {out_path}")

    status = state.get("status", "")
    return 0 if status in {"complete", "dry_run_complete"} else 1


def main() -> int:
    # Flush print() output line-by-line so `tail -f` of a redirected log
    # actually moves. Python defaults to block-buffered stdout when not
    # attached to a TTY (which is exactly the long-running nohup case),
    # making the log look frozen for minutes between flushes.
    try:
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    except AttributeError:
        # Python <3.7 or stdout already replaced; ignore.
        pass

    # Load .env at the very top so subsequent os.environ.get calls (the
    # relevance backends read API keys directly from the environment) pick
    # up keys the user keeps in a repo-root .env file. Shell `export` still
    # wins because load_env defaults to override=False.
    from pipeline.env import load_env
    load_env()

    p = argparse.ArgumentParser(description="Compositional synthesis pipeline")
    sub = p.add_subparsers(dest="cmd", required=False)

    # Reprocess subcommand: add L4 to an existing run
    pr = sub.add_parser("reprocess", help="Add L4 testability classification to an existing run")
    pr.add_argument("--run-dir", type=Path, required=True,
                    help="Path to data/pipeline/compositions/<timestamp>/")

    # Reprocess-l3 subcommand: re-run L3 (adversarial) with current prompt
    pr3 = sub.add_parser(
        "reprocess-l3",
        help="Re-run L3 adversarial verification on an existing run (use after L3 prompt edits)",
    )
    pr3.add_argument("--run-dir", type=Path, required=True,
                     help="Path to data/pipeline/compositions/<timestamp>/")
    pr3.add_argument(
        "--with-recipes", action="store_true", default=False,
        help="Generate recipes for compositions that newly clear to actionable",
    )

    # Default run (no subcommand) — full compose surface
    add_compose_args(p, required=False)
    args = p.parse_args()

    if args.cmd == "reprocess":
        reprocess_run_with_l4(args.run_dir)
        return 0
    if args.cmd == "reprocess-l3":
        reprocess_run_with_l3(args.run_dir, with_recipes=args.with_recipes)
        return 0

    if not args.problem or not (args.topic_list or args.paper_search):
        p.error("--problem plus one of --topic-list / --paper-search are "
                "required (or use a `reprocess*` subcommand)")

    return run_compose_from_args(args)


if __name__ == "__main__":
    sys.exit(main())
