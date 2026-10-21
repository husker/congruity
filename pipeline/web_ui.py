"""Static web UI for explaining Congruity compose runs.

The calibration report is useful for operators. This module builds the product
view: a self-contained dashboard plus one drill-down page per run, written from
the saved ``data/pipeline/compositions/<run_id>/state.json`` artifacts and the
verified-outcome ledger in ``findings/INDEX.md``.

Usage:

    uv run python -m pipeline.web_ui
    uv run python -m pipeline web
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from typing import Any

from pipeline.calibration import (
    INDEX_PATH,
    _parse_index_outcomes,
    _summarize_l3,
    _summarize_l4,
    _summarize_reconciled,
    _total_llm_time,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ROOT = REPO_ROOT / "data" / "pipeline" / "compositions"
DEFAULT_OUT = REPO_ROOT / "findings" / "web"


OUTCOME_LABELS = {
    "clean-positive": "Clear win",
    "partial-positive": "Promising partial win",
    "clean-negative": "Falsified",
    "partial-negative": "Close but not a win",
}

STATUS_LABELS = {
    "actionable": "Ready to test",
    "sound": "Strong claim",
    "weak_untestable": "Needs reformulation",
    "invalid": "Rejected by review",
    "incomplete": "Incomplete review",
}

DOMAIN_RULES: tuple[tuple[tuple[str, ...], str, str, str], ...] = (
    (
        (
            "tsp",
            "travelling salesman",
            "traveling salesman",
            "traveling-salesman",
            "travelling-salesman",
            "hamiltonian",
            "tsplib",
            "routing",
        ),
        "Hard route-and-cycle solver experiment",
        "Can cross-paper ideas make very hard route-planning puzzles easier to solve?",
        (
            "This is about problems where a solver must visit many points exactly once. "
            "The traveling-salesman version asks for the shortest such route; the "
            "Hamiltonian-cycle version asks whether such a loop exists at all."
        ),
    ),
    (
        ("heilbronn", "triangle"),
        "Point-placement geometry test",
        "Can optimization ideas produce better point layouts for a hard geometry problem?",
        (
            "This asks where to place points so that the tiniest triangle formed by any "
            "three points is as large as possible."
        ),
    ),
    (
        ("embedding", "quantization", "recall@10", "mteb", "beir"),
        "Search-vector compression test",
        "Can vector compression keep search quality while using fewer bits?",
        (
            "This is about shrinking the numeric fingerprints used by search and AI "
            "retrieval systems without losing the right answers."
        ),
    ),
    (
        ("cache", "sieve", "eviction"),
        "Memory-cache decision test",
        "Can a new eviction policy beat strong cache baselines?",
        (
            "This tests whether software can make better choices about what data to keep "
            "nearby and what data to throw away."
        ),
    ),
    (
        ("cerny", "synchronizing", "reset word", "complete dfa"),
        "Reset-word automata discovery test",
        "Can the system find better ways to reset a machine-like model with a short instruction string?",
        (
            "A synchronizing automaton is a tiny rule-based machine that can be forced "
            "into one known state no matter where it started. This run looks for better "
            "ways to find hard examples and prove how short the reset instructions can be."
        ),
    ),
    (
        ("compression", "brotli", "zstd", "hpack", "string"),
        "Tiny text compression test",
        "Can a composition shrink tiny strings better than production compressors?",
        (
            "This checks whether very short pieces of text can be packed smaller than "
            "standard compression tools manage today."
        ),
    ),
    (
        ("audio", "stem", "multitrack", "musdb"),
        "Separated-audio prediction test",
        "Can cross-stem prediction make audio stems more compressible or predictable?",
        (
            "This tests whether separated music tracks, such as vocals and drums, can "
            "predict each other well enough to improve compression or reconstruction."
        ),
    ),
)

EXPERIMENT_EXPLAINERS: tuple[tuple[tuple[str, ...], str, str, str], ...] = (
    (
        ("tsp delaunay", "delaunay candidate"),
        "Route-planning geometry shortcut test",
        "Can a geometry shortcut help a route solver avoid checking every possible road?",
        (
            "A Delaunay graph is a way to keep mostly nearby connections between points. "
            "For route planning, the hope is that ignoring unlikely long jumps makes the "
            "search faster without missing a good route."
        ),
    ),
    (
        ("heilbronn", "small-points"),
        "Point-placement geometry test",
        "Can a new optimizer find better point layouts for a famous geometry puzzle?",
        (
            "The Heilbronn problem asks how to place points so the smallest triangle made "
            "by any three of them is as large as possible. Bigger smallest triangle means "
            "a better layout."
        ),
    ),
    (
        ("short-string", "compression"),
        "Tiny text compression test",
        "Can the system find a better way to shrink very short strings?",
        (
            "Short strings are hard to compress because normal compressors spend many "
            "bytes just setting up their machinery. This run tested whether a generated "
            "idea could beat practical baselines."
        ),
    ),
    (
        ("embedding", "quantization"),
        "Search-vector compression test",
        "Can AI search vectors be made smaller without losing search accuracy?",
        (
            "Embedding vectors are long lists of numbers that help search systems find "
            "related text. Quantization means storing those numbers with fewer bits."
        ),
    ),
    (
        ("cache", "replacement"),
        "Memory-cache decision test",
        "Can a new rule make better keep-or-discard choices in a cache?",
        (
            "A cache is a small fast memory. The hard part is deciding which item to evict "
            "when it fills up, because the wrong choice slows future requests."
        ),
    ),
    (
        ("multitrack", "stems"),
        "Separated-audio prediction test",
        "Can one separated music track help predict another?",
        (
            "Music stems are separate tracks such as vocals, bass, and drums. This tested "
            "whether relationships between stems were strong enough to be useful."
        ),
    ),
)


@dataclass
class Outcome:
    outcome: str = ""
    experiment: str = ""
    doc: str = ""

    @property
    def label(self) -> str:
        if not self.outcome:
            return "Not verified yet"
        return OUTCOME_LABELS.get(self.outcome, self.outcome.replace("-", " ").title())

    @property
    def polarity(self) -> str:
        if "positive" in self.outcome:
            return "positive"
        if "negative" in self.outcome:
            return "negative"
        return "neutral"


@dataclass
class CompositionView:
    comp_id: str
    claim: str
    confidence: int | float | None
    l3_verdict: str
    l3_reason: str
    l4_score: str
    reconciled: str
    reconciled_reason: str
    l3_l4_consistent: bool | None
    empirical_objections: int
    fundamental_objections: int
    source_grounding: str
    baselines: list[str]
    metric_primary: str
    benchmark_dataset: str
    stop_rule: str
    likely_failure_mode: str
    open_questions: list[str]
    raw: dict[str, Any]

    @property
    def status_label(self) -> str:
        return STATUS_LABELS.get(self.reconciled, self.reconciled.replace("_", " ").title())

    @property
    def is_actionable(self) -> bool:
        return self.reconciled in {"actionable", "sound"}


@dataclass
class RunView:
    run_id: str
    path: Path
    state: dict[str, Any]
    title: str
    plain_goal: str
    plain_context: str
    technical_name: str
    status: str
    outcome: Outcome
    corpus_size: int
    dimension_count: int
    composition_count: int
    topics_covered: int
    scoring_coverage: float | None
    relevance_skipped: bool
    llm_seconds: float
    l3_counts: Counter[str]
    l4_counts: Counter[str]
    reconciled_counts: Counter[str]
    inconsistent_count: int
    compositions: list[CompositionView]
    gate: dict[str, Any]

    @property
    def complete(self) -> bool:
        return self.status in {"complete", "dry_run_complete"}

    @property
    def verified(self) -> bool:
        return bool(self.outcome.outcome)

    @property
    def actionable_count(self) -> int:
        return int(self.reconciled_counts.get("actionable", 0) + self.reconciled_counts.get("sound", 0))

    @property
    def invalid_count(self) -> int:
        return int(self.reconciled_counts.get("invalid", 0))

    @property
    def untestable_count(self) -> int:
        return int(self.reconciled_counts.get("weak_untestable", 0))

    @property
    def top_status(self) -> str:
        if self.outcome.outcome:
            return self.outcome.label
        if "aborted" in self.status:
            return "Stopped early"
        if self.composition_count == 0:
            return "No candidate survived"
        if self.actionable_count:
            return f"{self.actionable_count} testable candidate(s)"
        if self.invalid_count == self.composition_count:
            return "Rejected all candidates"
        return self.status.replace("_", " ")

    @property
    def css_group(self) -> str:
        if self.outcome.outcome:
            return self.outcome.polarity
        if self.actionable_count:
            return "actionable"
        if "aborted" in self.status or self.invalid_count:
            return "negative"
        return "neutral"


def build_site(root: Path = DEFAULT_ROOT, out_dir: Path = DEFAULT_OUT,
               index_path: Path = INDEX_PATH) -> dict[str, Any]:
    """Build the static dashboard and one detail page per run.

    Returns a small manifest for tests and CLI output.
    """
    runs = load_runs(root, index_path)
    generated = datetime.now(timezone.utc)
    out_dir.mkdir(parents=True, exist_ok=True)
    runs_dir = out_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    for run in runs:
        (runs_dir / f"{run.run_id}.html").write_text(
            render_run_page(run, generated),
            encoding="utf-8",
        )

    index_html = render_dashboard(runs, root, index_path, generated)
    (out_dir / "index.html").write_text(index_html, encoding="utf-8")
    return {
        "out_dir": out_dir,
        "index": out_dir / "index.html",
        "run_pages": [runs_dir / f"{r.run_id}.html" for r in runs],
        "run_count": len(runs),
    }


def load_runs(root: Path = DEFAULT_ROOT, index_path: Path = INDEX_PATH) -> list[RunView]:
    if not root.exists():
        return []

    previous_index = INDEX_PATH
    if index_path != INDEX_PATH:
        # _parse_index_outcomes is intentionally shared with calibration. It
        # reads the module-level path, so patch it only for this call.
        import pipeline.calibration as calibration

        calibration.INDEX_PATH = index_path
        try:
            index = _parse_index_outcomes()
        finally:
            calibration.INDEX_PATH = previous_index
    else:
        index = _parse_index_outcomes()

    out: list[RunView] = []
    for run_dir in sorted((p for p in root.iterdir() if p.is_dir()), reverse=True):
        state_path = run_dir / "state.json"
        if not state_path.exists():
            continue
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        run_id = str(state.get("run_id") or run_dir.name)
        outcome_info = index.get(run_id) or {}
        out.append(_make_run_view(run_dir, state, Outcome(**outcome_info)))
    return out


def _make_run_view(run_dir: Path, state: dict[str, Any], outcome: Outcome) -> RunView:
    compositions = [_make_composition_view(c) for c in state.get("compositions") or []]
    raw_compositions = state.get("compositions") or []
    rec_counts, inconsistent = _summarize_reconciled(raw_compositions)
    title, plain_goal, plain_context, technical_name = _infer_run_language(
        str(state.get("problem") or ""),
        outcome,
    )
    gate = _load_json(run_dir / "corpus_gate.json") or {}
    scoring_coverage = state.get("scoring_coverage")
    if not isinstance(scoring_coverage, (int, float)):
        scoring_coverage = _latest_scoring_coverage(run_dir)
    topics_covered = state.get("topics_covered")
    if not isinstance(topics_covered, int):
        topics_covered = _infer_topic_count(run_dir)
    return RunView(
        run_id=str(state.get("run_id") or run_dir.name),
        path=run_dir,
        state=state,
        title=title,
        plain_goal=plain_goal,
        plain_context=plain_context,
        technical_name=technical_name,
        status=str(state.get("status") or "unknown"),
        outcome=outcome,
        corpus_size=int(state.get("corpus_size") or 0),
        dimension_count=len(state.get("dimensions") or []),
        composition_count=len(raw_compositions),
        topics_covered=topics_covered,
        scoring_coverage=float(scoring_coverage) if isinstance(scoring_coverage, (int, float)) else None,
        relevance_skipped=bool(state.get("relevance_skipped") or "relevance" in (gate.get("skipped_checks") or [])),
        llm_seconds=round(_total_llm_time(run_dir), 1),
        l3_counts=_summarize_l3(raw_compositions),
        l4_counts=_summarize_l4(raw_compositions),
        reconciled_counts=rec_counts,
        inconsistent_count=inconsistent,
        compositions=compositions,
        gate=gate,
    )


def _make_composition_view(comp: dict[str, Any]) -> CompositionView:
    verification = comp.get("verification") or {}
    l3 = verification.get("L3") or {}
    l4 = verification.get("L4") or {}
    l4_summary = l4.get("summary") or {}
    rec = verification.get("reconciled") or {}
    if not rec:
        from pipeline.compose import _reconcile_verdict

        rec = _reconcile_verdict(l3, l4)

    l4_score = l4_summary.get("net_testability_score")
    if isinstance(l4_score, (int, float)):
        l4_score = f"{l4_score:.2f}"
    elif not isinstance(l4_score, str):
        l4_score = "missing"

    source_grounding = "not checked"
    l2 = verification.get("L2") or {}
    if l2:
        source_grounding = "grounded" if l2.get("passed") else "needs audit"

    recipe = verification.get("recipe") or {}
    verification_summary = comp.get("verification_summary") or {}
    return CompositionView(
        comp_id=str(comp.get("id") or "candidate"),
        claim=str(comp.get("synthesis_claim") or ""),
        confidence=comp.get("confidence"),
        l3_verdict=str(l3.get("verdict") or "missing"),
        l3_reason=str(l3.get("verdict_reason") or ""),
        l4_score=str(l4_score),
        reconciled=str(rec.get("status") or "incomplete"),
        reconciled_reason=str(rec.get("reason") or ""),
        l3_l4_consistent=rec.get("l3_l4_consistent"),
        empirical_objections=int(
            l4_summary.get("empirical_count")
            or verification_summary.get("objections_empirical")
            or 0
        ),
        fundamental_objections=int(
            l4_summary.get("fundamental_count")
            or verification_summary.get("objections_fundamental")
            or 0
        ),
        source_grounding=source_grounding,
        baselines=[str(x) for x in recipe.get("named_baselines") or []],
        metric_primary=str(recipe.get("metric_primary") or ""),
        benchmark_dataset=str(recipe.get("benchmark_dataset") or ""),
        stop_rule=str(recipe.get("stop_rule") or ""),
        likely_failure_mode=str(recipe.get("likely_failure_mode") or ""),
        open_questions=[str(x) for x in comp.get("open_questions") or []],
        raw=comp,
    )


def render_dashboard(runs: list[RunView], root: Path, index_path: Path,
                     generated: datetime) -> str:
    stats = _dashboard_stats(runs)
    generated_label = generated.strftime("%Y-%m-%d %H:%M UTC")
    rows = "\n".join(_run_row(run) for run in runs)
    run_cards = "\n".join(_spotlight_card(run) for run in _spotlight_runs(runs))
    outcome_bars = _outcome_bars(stats)
    funnel = _funnel_svg(stats)
    recent_runs = _recent_run_tiles(runs[:8])

    body = f"""
<header class="app-shell">
  <nav class="topbar" aria-label="Primary">
    <a class="brand" href="index.html">
      <span class="brand-mark" aria-hidden="true"></span>
      <span>Congruity Evidence Board</span>
    </a>
    <div class="topbar-meta">Generated {escape(generated_label)}</div>
  </nav>

  <section class="hero-grid">
    <div class="hero-copy">
      <p class="eyebrow">Compose -> attack -> test</p>
      <h1>Can an AI system turn papers into discoveries you can actually check?</h1>
      <p class="lede">Congruity is a research-direction finder that proposes cross-paper ideas, has them attacked by an adversarial reviewer, then turns the survivors into concrete experiments with baselines, metrics, and stop rules.</p>
      <div class="answer-strip">
        <div>
          <span class="answer-label">Current answer</span>
          <strong>{escape(stats["capability_answer"])}</strong>
        </div>
        <p>{escape(stats["capability_explanation"])}</p>
      </div>
    </div>
    <div class="score-panel" aria-label="Project proof summary">
      <div class="score-main">
        <span class="score-kicker">Verified signal</span>
        <strong>{stats["positive_verified"]}/{stats["verified_runs"]}</strong>
        <span>verified experiments produced a positive or partial-positive result</span>
      </div>
      <div class="score-grid">
        {_metric_tile("Runs", stats["runs"], "Compose attempts with saved state")}
        {_metric_tile("Testable candidates", stats["actionable"], "L3 weak or sound plus L4 purely testable")}
        {_metric_tile("Rejected candidates", stats["invalid"], "Ideas discarded by adversarial review")}
        {_metric_tile("Clean negatives", stats["clean_negative"], "Experiments that falsified a candidate")}
      </div>
    </div>
  </section>
</header>

<main>
  <section class="section-band proof-band" aria-labelledby="proof-title">
    <div class="section-head">
      <div>
        <p class="eyebrow">What the evidence says</p>
        <h2 id="proof-title">The system is useful when it is calibrated, not when every idea wins.</h2>
      </div>
      <p>Positive runs show discovery potential. Negative runs show the loop can falsify its own ideas instead of only producing plausible prose.</p>
    </div>
    <div class="proof-layout">
      <div class="funnel-card">
        {funnel}
      </div>
      <div class="proof-copy">
        <div class="finding positive">
          <strong>It can make testable hypotheses.</strong>
          <span>{stats["actionable"]} candidate(s) are marked ready for bounded verification, with named baselines and metrics when recipes exist.</span>
        </div>
        <div class="finding mixed">
          <strong>It has produced empirical wins, but the proof is still early.</strong>
          <span>{stats["positive_verified"]} positive or partial-positive result(s) across {stats["verified_runs"]} verified experiment(s). That is a signal, not a finished benchmark suite.</span>
        </div>
        <div class="finding negative">
          <strong>It rejects and falsifies many ideas.</strong>
          <span>{stats["invalid"]} candidate(s) were rejected by review and {stats["clean_negative"]} verified experiment(s) were clean negatives. That is what keeps the product honest.</span>
        </div>
      </div>
    </div>
  </section>

  <section class="section-band" aria-labelledby="track-title">
    <div class="section-head">
      <div>
        <p class="eyebrow">Verified track record</p>
        <h2 id="track-title">Runs with code-backed verdicts</h2>
      </div>
      <p>Each verified row connects a generated claim to a benchmark result. Use these as the highest-confidence evidence.</p>
    </div>
    <div class="track-layout">
      {outcome_bars}
      <div class="spotlight-grid">{run_cards}</div>
    </div>
  </section>

  <section class="section-band run-browser" aria-labelledby="runs-title">
    <div class="section-head">
      <div>
        <p class="eyebrow">Run browser</p>
        <h2 id="runs-title">Every compose run, in plain English</h2>
      </div>
      <div class="controls">
        <input id="run-search" type="search" placeholder="Search runs, goals, outcomes" aria-label="Search runs">
        <select id="run-filter" aria-label="Filter runs">
          <option value="all">All runs</option>
          <option value="verified">Verified only</option>
          <option value="actionable">Has testable candidates</option>
          <option value="positive">Positive signal</option>
          <option value="negative">Rejected or falsified</option>
          <option value="stopped">Stopped early</option>
        </select>
      </div>
    </div>
    <div class="recent-strip" aria-label="Most recent runs">{recent_runs}</div>
    <div class="table-wrap">
      <table class="run-table">
        <thead>
          <tr>
            <th>Run</th>
            <th>What it was for</th>
            <th>Result</th>
            <th>Evidence</th>
            <th></th>
          </tr>
        </thead>
        <tbody>{rows}</tbody>
      </table>
    </div>
  </section>
</main>

<footer class="site-footer">
  <span>Source data: {escape(str(root))}</span>
  <span>Verified ledger: {escape(str(index_path))}</span>
</footer>
"""
    return _document("Congruity Evidence Board", body, script=_DASHBOARD_JS)


def render_run_page(run: RunView, generated: datetime) -> str:
    generated_label = generated.strftime("%Y-%m-%d %H:%M UTC")
    cards = "\n".join(_composition_card(c) for c in run.compositions)
    if not cards:
        cards = """
<div class="empty-state">
  <strong>No compositions were produced for this run.</strong>
  <p>The useful conclusion is that the pipeline did not find a candidate strong enough to advance past the current gates.</p>
</div>
"""
    evidence = _run_evidence_summary(run)
    detail_json = escape(json.dumps(_run_export(run), indent=2, default=str))
    outcome_class = run.outcome.polarity if run.outcome.outcome else run.css_group
    body = f"""
<header class="detail-hero {escape(outcome_class)}">
  <nav class="topbar detail">
    <a class="brand" href="../index.html">
      <span class="brand-mark" aria-hidden="true"></span>
      <span>Congruity Evidence Board</span>
    </a>
    <a class="back-link" href="../index.html">Back to dashboard</a>
  </nav>
  <div class="detail-grid">
    <div>
      <p class="eyebrow">Run {escape(run.run_id)}</p>
      <h1>{escape(run.title)}</h1>
      <p class="lede">{escape(run.plain_goal)}</p>
      <div class="translation-box">
        <span>What this means</span>
        <p>{escape(run.plain_context)}</p>
        {_technical_label(run)}
      </div>
      <div class="plain-answer">
        <span>Did this prove Congruity works?</span>
        <strong>{escape(_proof_answer(run))}</strong>
        <p>{escape(_proof_reason(run))}</p>
      </div>
    </div>
    <aside class="run-facts" aria-label="Run facts">
      {_metric_tile("Outcome", run.outcome.label, "Empirical verdict when a verification doc exists")}
      {_metric_tile("Candidates", run.composition_count, "Generated compositions")}
      {_metric_tile("Ready to test", run.actionable_count, "Actionable L3/L4 survivors")}
      {_metric_tile("Corpus", run.corpus_size, "Papers after corpus validation")}
    </aside>
  </div>
</header>

<main>
  <section class="section-band" aria-labelledby="tested-title">
    <div class="section-head">
      <div>
        <p class="eyebrow">Layperson summary</p>
        <h2 id="tested-title">What was being tested?</h2>
      </div>
    </div>
    <div class="summary-grid">
      <div class="summary-item">
        <span>Goal</span>
        <p>{escape(run.plain_goal)}</p>
      </div>
      <div class="summary-item">
        <span>Plain-English translation</span>
        <p>{escape(run.plain_context)}</p>
      </div>
      <div class="summary-item">
        <span>Pipeline result</span>
        <p>{escape(_pipeline_result(run))}</p>
      </div>
      <div class="summary-item">
        <span>Evidence level</span>
        <p>{escape(evidence)}</p>
      </div>
    </div>
  </section>

  <section class="section-band" aria-labelledby="metrics-title">
    <div class="section-head compact">
      <div>
        <p class="eyebrow">Run health</p>
        <h2 id="metrics-title">How much evidence did the run produce?</h2>
      </div>
    </div>
    {_run_meter_grid(run)}
  </section>

  <section class="section-band" aria-labelledby="candidates-title">
    <div class="section-head">
      <div>
        <p class="eyebrow">Candidate hypotheses</p>
        <h2 id="candidates-title">What ideas came out of the run?</h2>
      </div>
      <p>Advanced users can open each candidate for baselines, metrics, reviewer objections, and raw JSON.</p>
    </div>
    <div class="candidate-list">{cards}</div>
  </section>

  <section class="section-band advanced" aria-labelledby="advanced-title">
    <div class="section-head compact">
      <div>
        <p class="eyebrow">Advanced audit</p>
        <h2 id="advanced-title">Run artifacts</h2>
      </div>
    </div>
    <details>
      <summary>Raw run summary JSON</summary>
      <pre>{detail_json}</pre>
    </details>
    <p class="artifact-path">Artifacts live in <code>{escape(str(run.path))}</code>. Page generated {escape(generated_label)}.</p>
  </section>
</main>
"""
    return _document(f"{run.run_id} - {run.title}", body)


def _dashboard_stats(runs: list[RunView]) -> dict[str, Any]:
    verified = [r for r in runs if r.verified]
    positive_verified = sum(1 for r in verified if "positive" in r.outcome.outcome)
    clean_negative = sum(1 for r in verified if r.outcome.outcome == "clean-negative")
    actionable = sum(r.actionable_count for r in runs)
    invalid = sum(r.invalid_count for r in runs)
    total_comps = sum(r.composition_count for r in runs)
    complete = sum(1 for r in runs if r.complete)
    stopped = sum(1 for r in runs if "aborted" in r.status)
    if positive_verified:
        answer = "Yes, with early but real evidence"
        explanation = (
            "Congruity has generated at least one verified positive result and several clean "
            "negatives. The product value is calibrated testable discovery, not guaranteed wins."
        )
    elif verified:
        answer = "Not yet proven"
        explanation = (
            "The verified loop is working, but the saved experiments have not yet produced a "
            "positive result. That is still useful falsification evidence."
        )
    else:
        answer = "Needs verification"
        explanation = (
            "The dashboard has compose artifacts, but no code-backed verification rows yet."
        )
    return {
        "runs": len(runs),
        "complete": complete,
        "stopped": stopped,
        "verified_runs": len(verified),
        "positive_verified": positive_verified,
        "clean_negative": clean_negative,
        "actionable": actionable,
        "invalid": invalid,
        "total_comps": total_comps,
        "capability_answer": answer,
        "capability_explanation": explanation,
    }


def _infer_run_language(problem: str, outcome: Outcome) -> tuple[str, str, str, str]:
    problem_clean = _compact_text(problem)
    technical_name = outcome.experiment or _headline_from_problem(problem_clean)
    technical_lower = technical_name.lower()
    for needles, title, goal, context in EXPERIMENT_EXPLAINERS:
        if any(n in technical_lower for n in needles):
            return title, goal, context, technical_name

    lower = problem_clean.lower()
    for needles, title, goal, context in DOMAIN_RULES:
        if any(n in lower for n in needles):
            return title, goal, context, technical_name

    if outcome.experiment:
        return (
            _plain_title_from_technical(outcome.experiment),
            _first_sentence(problem_clean, 170),
            (
                "This run has a technical experiment label in the ledger. The page keeps "
                "that label for traceability and explains the evidence in simpler terms."
            ),
            technical_name,
        )
    if problem_clean:
        return (
            _headline_from_problem(problem_clean),
            _first_sentence(problem_clean, 190),
            "No domain-specific translation was available, so this page uses the original problem statement.",
            technical_name,
        )
    return (
        "Untitled compose run",
        "No problem statement was recorded for this run.",
        "The saved run did not include enough context to translate its purpose.",
        technical_name,
    )


def _plain_title_from_technical(label: str) -> str:
    cleaned = re.sub(r"[-_]+", " ", label).strip()
    cleaned = re.sub(r"\bC\d+\b", "", cleaned).strip()
    if not cleaned:
        return "Verified experiment"
    return cleaned[:1].upper() + cleaned[1:]


def _headline_from_problem(problem: str) -> str:
    first = _first_sentence(problem, 80)
    first = re.sub(r"^(find|develop|create|build)\s+", "", first, flags=re.IGNORECASE)
    return first[:1].upper() + first[1:]


def _first_sentence(text: str, limit: int) -> str:
    text = _compact_text(text)
    if not text:
        return ""
    match = re.search(r"(?<=[.!?])\s", text)
    sentence = text[: match.start()].strip() if match else text
    if len(sentence) <= limit:
        return sentence
    return sentence[: limit - 1].rstrip() + "..."


def _compact_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _load_json(path: Path) -> dict[str, Any] | list[Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _latest_scoring_coverage(run_dir: Path) -> float | None:
    meta_paths = sorted(run_dir.glob("llm_calls/scoring*/**/meta.json"), reverse=True)
    for path in meta_paths:
        meta = _load_json(path)
        if isinstance(meta, dict) and isinstance(meta.get("scoring_coverage"), (int, float)):
            return float(meta["scoring_coverage"])
    return None


def _infer_topic_count(run_dir: Path) -> int:
    corpus = _load_json(run_dir / "corpus.json")
    if isinstance(corpus, list):
        return len({p.get("topic_id") or p.get("topic") for p in corpus if isinstance(p, dict)})
    return 0


def _run_row(run: RunView) -> str:
    result = run.top_status
    group = run.css_group
    evidence = _evidence_chip(run)
    data = " ".join(
        [
            f'data-group="{escape(group)}"',
            f'data-verified="{str(run.verified).lower()}"',
            f'data-actionable="{str(run.actionable_count > 0).lower()}"',
            f'data-stopped="{str("aborted" in run.status).lower()}"',
            f'data-search="{escape((run.run_id + " " + run.title + " " + run.plain_goal + " " + result).lower())}"',
        ]
    )
    return f"""
<tr {data}>
  <td><a class="run-link" href="runs/{escape(run.run_id)}.html">{escape(run.run_id)}</a></td>
  <td>
    <strong>{escape(run.title)}</strong>
    <span>{escape(run.plain_goal)}</span>
    <em>{escape(run.plain_context)}</em>
  </td>
  <td><span class="status-pill {escape(group)}">{escape(result)}</span></td>
  <td>{evidence}</td>
  <td><a class="text-action" href="runs/{escape(run.run_id)}.html">Open</a></td>
</tr>
"""


def _evidence_chip(run: RunView) -> str:
    if run.verified:
        doc = '<span>code-backed verdict</span>'
        return f'<span class="evidence-chip {escape(run.outcome.polarity)}">Verified</span>{doc}'
    if run.actionable_count:
        return f'<span class="evidence-chip actionable">{run.actionable_count} recipe-ready</span>'
    if run.composition_count:
        return f'<span class="evidence-chip neutral">{run.composition_count} candidate(s)</span>'
    return '<span class="evidence-chip neutral">No candidates</span>'


def _spotlight_runs(runs: list[RunView]) -> list[RunView]:
    verified = [r for r in runs if r.verified]
    if verified:
        return verified[:6]
    actionable = [r for r in runs if r.actionable_count]
    return actionable[:6]


def _spotlight_card(run: RunView) -> str:
    label = run.outcome.label if run.verified else run.top_status
    group = run.outcome.polarity if run.verified else run.css_group
    return f"""
<a class="spotlight-card {escape(group)}" href="runs/{escape(run.run_id)}.html">
  <span>{escape(label)}</span>
  <strong>{escape(run.title)}</strong>
  <p>{escape(run.plain_context)}</p>
  <small>{escape(_pipeline_result(run))}</small>
</a>
"""


def _outcome_bars(stats: dict[str, Any]) -> str:
    verified = max(1, int(stats["verified_runs"]))
    rows = [
        ("positive", "Positive or partial-positive", int(stats["positive_verified"])),
        ("negative", "Clean negatives", int(stats["clean_negative"])),
        ("actionable", "Recipe-ready candidates", int(stats["actionable"])),
        ("invalid", "Rejected by review", int(stats["invalid"])),
    ]
    html = ['<div class="outcome-panel">']
    for cls, label, value in rows:
        denom = verified if cls in {"positive", "negative"} else max(1, int(stats["total_comps"]))
        width = max(3, min(100, round(100 * value / denom)))
        html.append(
            f"""
<div class="outcome-row">
  <div><span>{escape(label)}</span><strong>{value}</strong></div>
  <div class="bar-track"><span class="{escape(cls)}" style="width:{width}%"></span></div>
</div>
"""
        )
    html.append("</div>")
    return "\n".join(html)


def _funnel_svg(stats: dict[str, Any]) -> str:
    stages = [
        ("Runs", stats["runs"], "Saved compose attempts"),
        ("Complete", stats["complete"], "Runs that reached an end state"),
        ("Candidates", stats["total_comps"], "Generated hypotheses"),
        ("Ready", stats["actionable"], "Survivors worth testing"),
        ("Verified wins", stats["positive_verified"], "Empirical positives"),
    ]
    max_v = max(1, max(v for _, v, _ in stages))
    rows = []
    for i, (label, value, note) in enumerate(stages):
        width = max(60, 330 * math.sqrt(value / max_v)) if value else 32
        y = 24 + i * 58
        rows.append(
            f"""
<g>
  <rect x="{(360 - width) / 2:.1f}" y="{y}" width="{width:.1f}" height="34" rx="6" class="funnel-{i}"></rect>
  <text x="180" y="{y + 22}" text-anchor="middle">{escape(label)}: {value}</text>
  <title>{escape(note)}</title>
</g>
"""
        )
    return f"""
<svg class="funnel" viewBox="0 0 360 330" role="img" aria-label="Congruity evidence funnel">
  <text x="180" y="15" text-anchor="middle" class="funnel-title">Evidence funnel</text>
  {''.join(rows)}
</svg>
"""


def _recent_run_tiles(runs: list[RunView]) -> str:
    return "\n".join(
        f"""
<a class="recent-tile {escape(run.css_group)}" href="runs/{escape(run.run_id)}.html">
  <span>{escape(run.run_id[-6:])}</span>
  <strong>{escape(run.title)}</strong>
  <em>{escape(run.plain_goal)}</em>
</a>
"""
        for run in runs
    )


def _proof_answer(run: RunView) -> str:
    if run.outcome.outcome in {"clean-positive", "partial-positive"}:
        return "It is positive evidence, because this run led to a benchmarked win."
    if run.outcome.outcome in {"clean-negative", "partial-negative"}:
        return "It proves the loop can falsify ideas, but this run was not a discovery win."
    if run.actionable_count:
        return "It produced testable hypotheses, but they still need code verification."
    if run.composition_count == 0:
        return "No. This run did not produce a candidate to test."
    if run.invalid_count:
        return "No discovery here. The reviewer rejected the candidates."
    return "Not enough evidence was produced by this run."


def _technical_label(run: RunView) -> str:
    if not run.technical_name or run.technical_name == run.title:
        return ""
    return (
        '<p class="technical-label">Technical label: '
        f"<code>{escape(run.technical_name)}</code></p>"
    )


def _proof_reason(run: RunView) -> str:
    if run.verified:
        doc = f" See {run.outcome.doc}." if run.outcome.doc else ""
        return (
            f"The verification ledger marks this run as {run.outcome.outcome}. "
            f"That means a code experiment has already tested the candidate against baselines.{doc}"
        )
    if run.actionable_count:
        return (
            "The run produced at least one L3 weak or sound, L4 purely testable candidate. "
            "That is exactly the handoff point for building a minimal benchmark."
        )
    if "aborted" in run.status:
        return "The corpus gate stopped the run before it could responsibly synthesize claims."
    return "There is no verified result attached to this run yet."


def _pipeline_result(run: RunView) -> str:
    if run.verified:
        return f"Verified outcome: {run.outcome.label.lower()}."
    if "aborted" in run.status:
        reason = run.gate.get("abort_reason") if isinstance(run.gate, dict) else ""
        if reason:
            return f"Stopped before synthesis because {reason}."
        return "Stopped before synthesis because the input corpus did not pass the quality gate."
    if run.composition_count == 0:
        return "No candidate composition survived the generation and scoring process."
    if run.actionable_count:
        return (
            f"Produced {run.actionable_count} testable candidate(s) out of "
            f"{run.composition_count}; next step is benchmark verification."
        )
    if run.invalid_count == run.composition_count:
        return "Every candidate was rejected by adversarial review."
    return (
        f"Produced {run.composition_count} candidate(s), but none are both strong enough "
        "and testable enough to be a clean next experiment."
    )


def _run_evidence_summary(run: RunView) -> str:
    if run.verified:
        return (
            f"Code-backed verification exists and labels the result {run.outcome.outcome}. "
            "The original technical experiment label is preserved on this page for audit."
        )
    if run.actionable_count:
        return (
            f"{run.actionable_count} candidate(s) are actionable. The dashboard can explain "
            "what to test, what baselines to use, and when to stop."
        )
    if "aborted" in run.status:
        return "The run stopped at the corpus quality gate, so no claim was generated."
    return "No external verification is attached yet."


def _run_meter_grid(run: RunView) -> str:
    coverage = "n/a" if run.scoring_coverage is None else f"{run.scoring_coverage:.0%}"
    gate = run.gate if isinstance(run.gate, dict) else {}
    gate_passed = gate.get("passed")
    if isinstance(gate_passed, int):
        gate_summary = f"{gate_passed}/{gate.get('total', '?')} passed"
    else:
        gate_summary = "not recorded"
    l3 = _count_string(run.l3_counts)
    l4 = _count_string(run.l4_counts)
    return f"""
<div class="meter-grid">
  {_metric_tile("Corpus gate", gate_summary, "Input-paper quality checks")}
  {_metric_tile("Scoring coverage", coverage, "Fraction of corpus scored by the ranking stage")}
  {_metric_tile("L3 review", l3 or "none", "Adversarial verdict mix")}
  {_metric_tile("L4 testability", l4 or "none", "Can objections be settled by experiments?")}
  {_metric_tile("LLM wall time", f"{run.llm_seconds:.1f}s", "Recorded duration across LLM calls")}
  {_metric_tile("Disagreements", run.inconsistent_count, "L3 and L4 verdict inconsistencies")}
</div>
"""


def _count_string(counter: Counter[str]) -> str:
    return ", ".join(f"{k}: {v}" for k, v in counter.most_common())


def _composition_card(comp: CompositionView) -> str:
    group = "actionable" if comp.is_actionable else "negative" if comp.reconciled == "invalid" else "neutral"
    baselines = "".join(f"<li>{escape(b)}</li>" for b in comp.baselines[:6])
    if not baselines:
        baselines = "<li>No recipe baselines recorded.</li>"
    questions = "".join(f"<li>{escape(q)}</li>" for q in comp.open_questions[:5])
    if not questions:
        questions = "<li>No open questions recorded.</li>"
    raw = escape(json.dumps(_composition_export(comp), indent=2, default=str))
    confidence = "n/a" if comp.confidence is None else str(comp.confidence)
    return f"""
<article class="candidate {escape(group)}">
  <div class="candidate-head">
    <div>
      <span class="candidate-id">{escape(comp.comp_id)}</span>
      <h3>{escape(_first_sentence(comp.claim, 180))}</h3>
    </div>
    <span class="status-pill {escape(group)}">{escape(comp.status_label)}</span>
  </div>
  <p>{escape(comp.claim)}</p>
  <div class="candidate-stats">
    <span>Confidence <strong>{escape(confidence)}</strong></span>
    <span>L3 <strong>{escape(comp.l3_verdict)}</strong></span>
    <span>L4 <strong>{escape(comp.l4_score)}</strong></span>
    <span>Empirical objections <strong>{comp.empirical_objections}</strong></span>
    <span>Fundamental objections <strong>{comp.fundamental_objections}</strong></span>
  </div>
  <details>
    <summary>Baselines, metrics, and stop rule</summary>
    <div class="detail-columns">
      <div>
        <h4>Named baselines</h4>
        <ul>{baselines}</ul>
      </div>
      <div>
        <h4>Primary metric</h4>
        <p>{escape(comp.metric_primary or "No metric recorded.")}</p>
        <h4>Benchmark data</h4>
        <p>{escape(comp.benchmark_dataset or "No benchmark dataset recorded.")}</p>
      </div>
    </div>
    <h4>Stop rule</h4>
    <p>{escape(comp.stop_rule or "No stop rule recorded.")}</p>
    <h4>Likely failure mode</h4>
    <p>{escape(comp.likely_failure_mode or "No failure mode recorded.")}</p>
  </details>
  <details>
    <summary>Reviewer rationale and open questions</summary>
    <h4>L3 adversarial reason</h4>
    <p>{escape(comp.l3_reason or "No L3 reason recorded.")}</p>
    <h4>Open questions</h4>
    <ul>{questions}</ul>
  </details>
  <details>
    <summary>Raw candidate JSON</summary>
    <pre>{raw}</pre>
  </details>
</article>
"""


def _metric_tile(label: str, value: object, help_text: str) -> str:
    return f"""
<div class="metric-tile" title="{escape(help_text)}">
  <span>{escape(label)}</span>
  <strong>{escape(str(value))}</strong>
</div>
"""


def _run_export(run: RunView) -> dict[str, Any]:
    return {
        "run_id": run.run_id,
        "title": run.title,
        "plain_goal": run.plain_goal,
        "plain_context": run.plain_context,
        "technical_name": run.technical_name,
        "status": run.status,
        "outcome": run.outcome.__dict__,
        "corpus_size": run.corpus_size,
        "dimensions": run.dimension_count,
        "compositions": run.composition_count,
        "actionable": run.actionable_count,
        "invalid": run.invalid_count,
        "l3_counts": dict(run.l3_counts),
        "l4_counts": dict(run.l4_counts),
        "reconciled_counts": dict(run.reconciled_counts),
        "problem": run.state.get("problem"),
    }


def _composition_export(comp: CompositionView) -> dict[str, Any]:
    return {
        "id": comp.comp_id,
        "claim": comp.claim,
        "confidence": comp.confidence,
        "l3_verdict": comp.l3_verdict,
        "l3_reason": comp.l3_reason,
        "l4_score": comp.l4_score,
        "reconciled": comp.reconciled,
        "baselines": comp.baselines,
        "metric_primary": comp.metric_primary,
        "benchmark_dataset": comp.benchmark_dataset,
        "stop_rule": comp.stop_rule,
    }


def _document(title: str, body: str, script: str = "") -> str:
    script_tag = f"<script>{script}</script>" if script else ""
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escape(title)}</title>
  <link rel="icon" href="data:,">
  <style>{_CSS}</style>
</head>
<body>
{body}
{script_tag}
</body>
</html>
"""


_DASHBOARD_JS = """
const search = document.getElementById('run-search');
const filter = document.getElementById('run-filter');
function applyRunFilters() {
  const q = (search?.value || '').trim().toLowerCase();
  const f = filter?.value || 'all';
  document.querySelectorAll('.run-table tbody tr').forEach((row) => {
    const matchesText = !q || row.dataset.search.includes(q);
    const matchesFilter =
      f === 'all' ||
      (f === 'verified' && row.dataset.verified === 'true') ||
      (f === 'actionable' && row.dataset.actionable === 'true') ||
      (f === 'positive' && row.dataset.group === 'positive') ||
      (f === 'negative' && row.dataset.group === 'negative') ||
      (f === 'stopped' && row.dataset.stopped === 'true');
    row.hidden = !(matchesText && matchesFilter);
  });
}
search?.addEventListener('input', applyRunFilters);
filter?.addEventListener('change', applyRunFilters);
"""


_CSS = """
:root {
  color-scheme: light;
  --paper: #f8f4eb;
  --surface: #fffdf7;
  --surface-strong: #ffffff;
  --ink: #191714;
  --muted: #6f675c;
  --line: #ded5c7;
  --green: #0f7c63;
  --green-soft: #d9eee8;
  --red: #a94237;
  --red-soft: #f3ddd8;
  --gold: #9a6a16;
  --gold-soft: #f4e8c7;
  --blue: #285c86;
  --blue-soft: #dce9f2;
  --violet: #654c8f;
  --shadow: 0 18px 50px rgba(42, 33, 22, 0.08);
}
* { box-sizing: border-box; }
html { background: var(--paper); }
body {
  margin: 0;
  min-width: 320px;
  color: var(--ink);
  background:
    linear-gradient(180deg, rgba(255, 253, 247, 0.94), rgba(248, 244, 235, 0.98)),
    var(--paper);
  font-family: "Avenir Next", "Segoe UI", system-ui, sans-serif;
  line-height: 1.5;
}
a { color: inherit; text-decoration: none; }
code, pre {
  font-family: "SF Mono", Consolas, monospace;
  font-size: 0.88em;
}
pre {
  overflow: auto;
  max-height: 520px;
  padding: 16px;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: #15130f;
  color: #f5ead9;
}
h1, h2, h3, h4, p { margin-top: 0; }
h1, h2, h3 {
  font-family: Georgia, "Times New Roman", serif;
  letter-spacing: 0;
}
h1 {
  max-width: 820px;
  margin-bottom: 18px;
  font-size: clamp(2.2rem, 5vw, 5.8rem);
  line-height: 0.94;
}
h2 {
  margin-bottom: 0;
  font-size: clamp(1.45rem, 2.8vw, 2.6rem);
  line-height: 1.04;
}
h3 { font-size: 1.25rem; line-height: 1.2; }
h4 {
  margin: 16px 0 6px;
  color: var(--muted);
  font-size: 0.76rem;
  letter-spacing: 0.08em;
  text-transform: uppercase;
}
.app-shell, .detail-hero {
  padding: 24px clamp(18px, 4vw, 52px) 42px;
}
.topbar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
  margin-bottom: 54px;
}
.topbar.detail { margin-bottom: 36px; }
.brand {
  display: inline-flex;
  align-items: center;
  gap: 10px;
  min-height: 40px;
  font-weight: 700;
}
.brand-mark {
  width: 26px;
  height: 26px;
  border: 2px solid var(--ink);
  border-radius: 50%;
  background:
    linear-gradient(90deg, transparent 46%, var(--ink) 46% 54%, transparent 54%),
    linear-gradient(0deg, transparent 46%, var(--ink) 46% 54%, transparent 54%),
    var(--gold-soft);
}
.topbar-meta, .back-link {
  color: var(--muted);
  font-size: 0.9rem;
}
.back-link {
  padding: 8px 12px;
  border: 1px solid var(--line);
  border-radius: 6px;
  background: rgba(255, 255, 255, 0.55);
}
.hero-grid, .detail-grid {
  display: grid;
  grid-template-columns: minmax(0, 1.35fr) minmax(320px, 0.65fr);
  gap: clamp(24px, 6vw, 72px);
  align-items: end;
}
.detail-grid { align-items: stretch; }
.eyebrow {
  margin-bottom: 12px;
  color: var(--gold);
  font-size: 0.78rem;
  font-weight: 800;
  letter-spacing: 0.12em;
  text-transform: uppercase;
}
.lede {
  max-width: 760px;
  color: #3f3932;
  font-size: clamp(1.05rem, 1.9vw, 1.35rem);
}
.answer-strip, .plain-answer {
  display: grid;
  grid-template-columns: minmax(220px, 0.55fr) minmax(0, 1fr);
  gap: 20px;
  max-width: 900px;
  margin-top: 32px;
  padding: 18px;
  border: 1px solid var(--line);
  border-left: 5px solid var(--green);
  border-radius: 8px;
  background: rgba(255, 253, 247, 0.72);
}
.plain-answer { grid-template-columns: 1fr; }
.answer-strip p, .plain-answer p { margin-bottom: 0; color: var(--muted); }
.answer-label, .plain-answer span {
  display: block;
  margin-bottom: 6px;
  color: var(--muted);
  font-size: 0.78rem;
  font-weight: 800;
  letter-spacing: 0.08em;
  text-transform: uppercase;
}
.answer-strip strong, .plain-answer strong {
  display: block;
  font-family: Georgia, "Times New Roman", serif;
  font-size: 1.55rem;
  line-height: 1.1;
}
.score-panel, .run-facts {
  padding: 20px;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: rgba(255, 253, 247, 0.82);
  box-shadow: var(--shadow);
}
.score-main {
  display: grid;
  gap: 8px;
  padding-bottom: 18px;
  border-bottom: 1px solid var(--line);
}
.score-kicker {
  color: var(--muted);
  font-size: 0.78rem;
  font-weight: 800;
  letter-spacing: 0.08em;
  text-transform: uppercase;
}
.score-main strong {
  font-family: Georgia, "Times New Roman", serif;
  font-size: 4.2rem;
  line-height: 0.95;
}
.score-main span:last-child { color: var(--muted); }
.score-grid, .run-facts, .meter-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 12px;
  margin-top: 14px;
}
.metric-tile {
  min-height: 92px;
  padding: 12px;
  border: 1px solid var(--line);
  border-radius: 7px;
  background: var(--surface-strong);
}
.metric-tile span {
  display: block;
  margin-bottom: 12px;
  color: var(--muted);
  font-size: 0.76rem;
  font-weight: 800;
  letter-spacing: 0.08em;
  text-transform: uppercase;
}
.metric-tile strong {
  display: block;
  overflow-wrap: anywhere;
  font-family: Georgia, "Times New Roman", serif;
  font-size: clamp(1.25rem, 2.2vw, 1.75rem);
  line-height: 1.05;
}
main { padding-bottom: 40px; }
.section-band {
  padding: 46px clamp(18px, 4vw, 52px);
  border-top: 1px solid var(--line);
  background: rgba(255, 253, 247, 0.48);
}
.section-band:nth-of-type(even) { background: rgba(246, 239, 224, 0.58); }
.section-head {
  display: flex;
  align-items: end;
  justify-content: space-between;
  gap: 28px;
  margin-bottom: 26px;
}
.section-head.compact { margin-bottom: 18px; }
.section-head > p {
  max-width: 520px;
  margin-bottom: 0;
  color: var(--muted);
}
.proof-layout, .track-layout {
  display: grid;
  grid-template-columns: minmax(280px, 0.9fr) minmax(0, 1.1fr);
  gap: 28px;
  align-items: stretch;
}
.funnel-card, .outcome-panel {
  padding: 18px;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: var(--surface);
}
.funnel { width: 100%; min-height: 300px; }
.funnel text {
  fill: var(--ink);
  font-family: "Avenir Next", "Segoe UI", system-ui, sans-serif;
  font-size: 13px;
  font-weight: 700;
}
.funnel .funnel-title {
  fill: var(--muted);
  font-size: 12px;
  letter-spacing: 0.08em;
  text-transform: uppercase;
}
.funnel-0 { fill: var(--blue-soft); }
.funnel-1 { fill: var(--gold-soft); }
.funnel-2 { fill: #eadfd5; }
.funnel-3 { fill: var(--green-soft); }
.funnel-4 { fill: var(--green); }
.funnel-4 + text { fill: #fffdf7; }
.proof-copy {
  display: grid;
  gap: 14px;
}
.finding {
  padding: 18px;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: var(--surface);
}
.finding strong {
  display: block;
  margin-bottom: 6px;
  font-family: Georgia, "Times New Roman", serif;
  font-size: 1.35rem;
}
.finding span { color: var(--muted); }
.finding.positive { border-left: 5px solid var(--green); }
.finding.mixed { border-left: 5px solid var(--gold); }
.finding.negative { border-left: 5px solid var(--red); }
.outcome-panel { display: grid; gap: 20px; }
.outcome-row {
  display: grid;
  gap: 8px;
}
.outcome-row div:first-child {
  display: flex;
  justify-content: space-between;
  gap: 12px;
  color: var(--muted);
  font-size: 0.92rem;
}
.outcome-row strong {
  color: var(--ink);
  font-family: Georgia, "Times New Roman", serif;
}
.bar-track {
  height: 12px;
  overflow: hidden;
  border-radius: 999px;
  background: #ece3d6;
}
.bar-track span {
  display: block;
  height: 100%;
  border-radius: inherit;
}
.bar-track .positive { background: var(--green); }
.bar-track .negative { background: var(--red); }
.bar-track .actionable { background: var(--blue); }
.bar-track .invalid { background: var(--gold); }
.spotlight-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 14px;
}
.spotlight-card {
  display: grid;
  gap: 8px;
  min-height: 170px;
  padding: 18px;
  border: 1px solid var(--line);
  border-top: 5px solid var(--blue);
  border-radius: 8px;
  background: var(--surface);
  transition: transform 160ms ease, box-shadow 160ms ease;
}
.spotlight-card:hover {
  transform: translateY(-2px);
  box-shadow: var(--shadow);
}
.spotlight-card.positive { border-top-color: var(--green); }
.spotlight-card.negative { border-top-color: var(--red); }
.spotlight-card.actionable { border-top-color: var(--blue); }
.spotlight-card span {
  color: var(--muted);
  font-size: 0.76rem;
  font-weight: 800;
  letter-spacing: 0.08em;
  text-transform: uppercase;
}
.spotlight-card strong {
  font-family: Georgia, "Times New Roman", serif;
  font-size: 1.22rem;
}
.spotlight-card p { margin-bottom: 0; color: var(--muted); }
.spotlight-card small {
  display: block;
  color: #3f3932;
  font-size: 0.88rem;
  line-height: 1.45;
}
.controls {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
  justify-content: flex-end;
}
.controls input, .controls select {
  min-height: 42px;
  border: 1px solid var(--line);
  border-radius: 7px;
  background: var(--surface-strong);
  color: var(--ink);
  font: inherit;
}
.controls input {
  width: min(320px, 100%);
  padding: 0 14px;
}
.controls select { padding: 0 38px 0 12px; }
.recent-strip {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 10px;
  margin-bottom: 18px;
}
.recent-tile {
  min-height: 88px;
  padding: 12px;
  border: 1px solid var(--line);
  border-left: 4px solid var(--blue);
  border-radius: 7px;
  background: var(--surface);
}
.recent-tile.positive { border-left-color: var(--green); }
.recent-tile.negative { border-left-color: var(--red); }
.recent-tile.actionable { border-left-color: var(--blue); }
.recent-tile span {
  display: block;
  color: var(--muted);
  font-size: 0.75rem;
  font-weight: 800;
}
.recent-tile strong {
  display: block;
  margin-top: 6px;
  overflow-wrap: anywhere;
  font-family: Georgia, "Times New Roman", serif;
  font-size: 1rem;
  line-height: 1.16;
}
.recent-tile em {
  display: block;
  margin-top: 6px;
  color: var(--muted);
  font-size: 0.78rem;
  font-style: normal;
  line-height: 1.25;
}
.table-wrap {
  overflow-x: auto;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: var(--surface);
}
.run-table {
  width: 100%;
  min-width: 820px;
  border-collapse: collapse;
}
.run-table th, .run-table td {
  padding: 14px;
  border-bottom: 1px solid var(--line);
  vertical-align: middle;
  text-align: left;
}
.run-table th {
  color: var(--muted);
  font-size: 0.76rem;
  letter-spacing: 0.08em;
  text-transform: uppercase;
}
.run-table tbody tr:last-child td { border-bottom: 0; }
.run-table td:nth-child(2) {
  min-width: 340px;
}
.run-table td:nth-child(2) strong,
.run-table td:nth-child(2) span,
.run-table td:nth-child(2) em {
  display: block;
}
.run-table td:nth-child(2) span,
.run-table td:nth-child(2) em {
  max-width: 620px;
  color: var(--muted);
}
.run-table td:nth-child(2) em {
  margin-top: 5px;
  color: #3f3932;
  font-size: 0.88rem;
  font-style: normal;
}
.run-link {
  font-family: "SF Mono", Consolas, monospace;
  color: var(--blue);
}
.status-pill, .evidence-chip {
  display: inline-flex;
  align-items: center;
  min-height: 28px;
  padding: 4px 9px;
  border-radius: 999px;
  font-size: 0.78rem;
  font-weight: 800;
  white-space: nowrap;
}
.status-pill.positive, .evidence-chip.positive { background: var(--green-soft); color: var(--green); }
.status-pill.negative, .evidence-chip.negative { background: var(--red-soft); color: var(--red); }
.status-pill.actionable, .evidence-chip.actionable { background: var(--blue-soft); color: var(--blue); }
.status-pill.neutral, .evidence-chip.neutral { background: #ede7db; color: var(--muted); }
.status-pill.mixed { background: var(--gold-soft); color: var(--gold); }
.text-action {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  min-height: 34px;
  padding: 0 12px;
  border: 1px solid var(--line);
  border-radius: 6px;
  background: var(--surface-strong);
  color: var(--blue);
  font-weight: 700;
}
.detail-hero {
  border-bottom: 1px solid var(--line);
}
.detail-hero.positive { background: linear-gradient(180deg, #eef7f2, var(--paper)); }
.detail-hero.negative { background: linear-gradient(180deg, #f8ece8, var(--paper)); }
.detail-hero.actionable { background: linear-gradient(180deg, #eef5fa, var(--paper)); }
.detail-hero.neutral { background: linear-gradient(180deg, #fbf7ee, var(--paper)); }
.run-facts {
  align-self: stretch;
  margin-top: 0;
}
.translation-box {
  max-width: 780px;
  margin-top: 18px;
  padding: 16px 18px;
  border: 1px solid var(--line);
  border-left: 5px solid var(--blue);
  border-radius: 8px;
  background: rgba(255, 253, 247, 0.76);
}
.translation-box span {
  display: block;
  margin-bottom: 7px;
  color: var(--muted);
  font-size: 0.76rem;
  font-weight: 800;
  letter-spacing: 0.08em;
  text-transform: uppercase;
}
.translation-box p { margin-bottom: 0; color: #3f3932; }
.technical-label {
  margin-top: 10px;
  color: var(--muted);
  font-size: 0.88rem;
}
.summary-grid {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 14px;
}
.summary-item {
  min-height: 190px;
  padding: 18px;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: var(--surface);
}
.summary-item span {
  display: block;
  margin-bottom: 10px;
  color: var(--muted);
  font-size: 0.76rem;
  font-weight: 800;
  letter-spacing: 0.08em;
  text-transform: uppercase;
}
.summary-item p { color: #3f3932; }
.meter-grid {
  margin-top: 0;
  grid-template-columns: repeat(3, minmax(0, 1fr));
}
.candidate-list {
  display: grid;
  gap: 16px;
}
.candidate {
  padding: 18px;
  border: 1px solid var(--line);
  border-left: 5px solid var(--blue);
  border-radius: 8px;
  background: var(--surface);
}
.candidate.actionable { border-left-color: var(--blue); }
.candidate.negative { border-left-color: var(--red); }
.candidate.neutral { border-left-color: var(--gold); }
.candidate-head {
  display: flex;
  align-items: start;
  justify-content: space-between;
  gap: 14px;
  margin-bottom: 10px;
}
.candidate-id {
  display: inline-flex;
  min-height: 26px;
  align-items: center;
  margin-bottom: 8px;
  padding: 0 8px;
  border-radius: 999px;
  background: #ede7db;
  color: var(--muted);
  font-size: 0.75rem;
  font-weight: 800;
}
.candidate h3 {
  margin-bottom: 0;
  max-width: 900px;
}
.candidate > p { color: #3f3932; }
.candidate-stats {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin: 16px 0;
}
.candidate-stats span {
  padding: 8px 10px;
  border: 1px solid var(--line);
  border-radius: 6px;
  background: var(--surface-strong);
  color: var(--muted);
}
.candidate-stats strong {
  margin-left: 4px;
  color: var(--ink);
}
details {
  margin-top: 10px;
  border-top: 1px solid var(--line);
  padding-top: 10px;
}
summary {
  cursor: pointer;
  color: var(--blue);
  font-weight: 800;
}
.detail-columns {
  display: grid;
  grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
  gap: 22px;
}
ul { padding-left: 20px; }
.empty-state {
  padding: 24px;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: var(--surface);
}
.empty-state strong {
  display: block;
  margin-bottom: 8px;
  font-family: Georgia, "Times New Roman", serif;
  font-size: 1.3rem;
}
.artifact-path {
  margin-top: 14px;
  color: var(--muted);
}
.site-footer {
  display: flex;
  flex-wrap: wrap;
  gap: 16px;
  justify-content: space-between;
  padding: 22px clamp(18px, 4vw, 52px);
  border-top: 1px solid var(--line);
  color: var(--muted);
  font-size: 0.86rem;
}
@media (max-width: 980px) {
  .hero-grid, .detail-grid, .proof-layout, .track-layout {
    grid-template-columns: 1fr;
  }
  .score-panel { max-width: 720px; }
  .summary-grid, .meter-grid {
    grid-template-columns: 1fr 1fr;
  }
  .recent-strip, .spotlight-grid {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }
}
@media (max-width: 680px) {
  .app-shell, .detail-hero {
    padding-top: 16px;
  }
  .topbar {
    align-items: flex-start;
    flex-direction: column;
    margin-bottom: 34px;
  }
  h1 {
    font-size: clamp(2rem, 13vw, 3.6rem);
  }
  .answer-strip, .plain-answer, .section-head, .candidate-head {
    display: block;
  }
  .answer-strip > * + *, .plain-answer > * + * {
    margin-top: 12px;
  }
  .score-grid, .run-facts, .summary-grid, .meter-grid, .recent-strip,
  .spotlight-grid, .detail-columns {
    grid-template-columns: 1fr;
  }
  .controls {
    justify-content: stretch;
    width: 100%;
  }
  .controls input, .controls select {
    width: 100%;
  }
  .section-band {
    padding-top: 34px;
    padding-bottom: 34px;
  }
}
"""


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate the Congruity static web UI")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--index", type=Path, default=INDEX_PATH)
    args = parser.parse_args()
    manifest = build_site(args.root, args.out, args.index)
    print(f"wrote {manifest['index']}")
    print(f"wrote {len(manifest['run_pages'])} run page(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
