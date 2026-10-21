"""Calibration dashboard: walk saved compose runs and emit a report.

The track record is the product's trust engine. This module aggregates every
compose run under ``data/pipeline/compositions/<run_id>/state.json`` plus the
manually-maintained ``findings/INDEX.md`` and produces a report covering:

  - Run summary table
  - L3 verdict mix (sound / weak / invalid)
  - L4 testability mix (PURELY_TESTABLE / numeric_ratio / missing)
  - Skipped relevance gates
  - LLM wall time per run
  - Verified-outcomes section (rendered from findings/INDEX.md if present)

Two output formats:

  --format html  (default) — self-contained HTML with sortable tables and
                              inline-SVG verdict bar charts. No external
                              dependencies; opens in any browser.
  --format md            — markdown report; same data, plain text.

Usage::

    uv run python -m pipeline.calibration                       # → findings/calibration-<today>.html
    uv run python -m pipeline.calibration --format md           # → findings/calibration-<today>.md
    uv run python -m pipeline.calibration --root <dir> --out <path>
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ROOT = REPO_ROOT / "data" / "pipeline" / "compositions"
INDEX_PATH = REPO_ROOT / "findings" / "INDEX.md"


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


def _summarize_l3(compositions: list[dict]) -> Counter:
    counts: Counter[str] = Counter()
    for c in compositions:
        v = (c.get("verification") or {}).get("L3") or {}
        verdict = v.get("verdict") or "missing"
        counts[verdict] += 1
    return counts


def _summarize_l4(compositions: list[dict]) -> Counter:
    counts: Counter[str] = Counter()
    for c in compositions:
        l4_summary = ((c.get("verification") or {}).get("L4") or {}).get("summary") or {}
        score = l4_summary.get("net_testability_score")
        if isinstance(score, str):
            counts[score] += 1
        elif isinstance(score, (int, float)):
            counts["numeric_ratio"] += 1
        else:
            counts["missing"] += 1
    return counts


def _summarize_reconciled(compositions: list[dict]) -> tuple[Counter, int]:
    """Return (status_counts, n_l3_l4_inconsistent).

    Falls back to deriving the reconciliation on-the-fly for runs that
    predate the reconciler (state.json from before the field landed).
    """
    counts: Counter[str] = Counter()
    inconsistent = 0
    for c in compositions:
        v = c.get("verification") or {}
        rec = v.get("reconciled")
        if not rec:
            # On-the-fly fallback for older runs.
            from pipeline.compose import _reconcile_verdict
            rec = _reconcile_verdict(v.get("L3"), v.get("L4"))
        status = rec.get("status", "incomplete")
        counts[status] += 1
        if rec.get("l3_l4_consistent") is False:
            inconsistent += 1
    return counts, inconsistent


def generate_report(root: Path) -> str:
    if not root.exists():
        return "# Calibration report\n\nNo compose runs found at this root.\n"

    runs: list[tuple[Path, dict]] = []
    for run_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        state = _load_run(run_dir)
        if state is None:
            continue
        runs.append((run_dir, state))

    if not runs:
        return "# Calibration report\n\nNo compose runs found at this root.\n"

    lines: list[str] = ["# Calibration report", ""]
    lines.append(f"Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}.")
    lines.append(f"Aggregating {len(runs)} compose run(s) under `{root}`.")
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

    # Aggregate L3 / L4 / gates / LLM time
    l3_counts: Counter[str] = Counter()
    l4_counts: Counter[str] = Counter()
    skipped_relevance = 0
    total_llm_seconds = 0.0
    for run_dir, state in runs:
        if state.get("relevance_skipped"):
            skipped_relevance += 1
        total_llm_seconds += _total_llm_time(run_dir)
        compositions = state.get("compositions") or []
        l3_counts += _summarize_l3(compositions)
        l4_counts += _summarize_l4(compositions)

    lines.append("## L3 verdict mix")
    lines.append("")
    if l3_counts:
        for k, v in l3_counts.most_common():
            lines.append(f"- **{k}**: {v}")
    else:
        lines.append("- (no compositions in any run)")
    lines.append("")

    lines.append("## L4 testability mix")
    lines.append("")
    if l4_counts:
        for k, v in l4_counts.most_common():
            lines.append(f"- **{k}**: {v}")
    else:
        lines.append("- (no compositions in any run)")
    lines.append("")

    lines.append("## Skipped gates")
    lines.append("")
    lines.append(
        f"- runs with relevance check skipped: **{skipped_relevance}** / {len(runs)}"
    )
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

    if INDEX_PATH.exists():
        lines.append("## Verified track record")
        lines.append("")
        lines.append(INDEX_PATH.read_text().rstrip())
        lines.append("")
    else:
        lines.append("## Verified track record")
        lines.append("")
        lines.append(
            "_findings/INDEX.md not found; create it to enable per-experiment outcome counts._"
        )
        lines.append("")

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Tooltip vocabulary
#
# Every term that appears in the dashboard with a non-obvious meaning lives
# here so the explanation is in one place. Used by both SVG <title> elements
# (chart labels) and HTML title= attributes (metrics, table headers,
# outcome badges).
# ---------------------------------------------------------------------------

TIPS_L3 = {
    "sound": "Adversarial reviewer found no falsification attempt that survived scrutiny.",
    "weak": "Adversarial reviewer found objections, but none rise to a fundamental refutation.",
    "invalid": "Adversarial reviewer identified a fatal flaw; composition is rejected.",
    "missing": "No L3 stage output recorded for this composition.",
}

TIPS_L4 = {
    "PURELY_TESTABLE": (
        "Every L3 objection was classified EMPIRICAL — the composition can be settled by a "
        "finite computational/experimental test."
    ),
    "numeric_ratio": (
        "Some L3 objections were FUNDAMENTAL. Score is empirical_count / total_objections "
        "(0.0 = nothing testable; 1.0 = fully testable)."
    ),
    "NOT_TESTABLE": "All L3 objections were FUNDAMENTAL; no experiment can rescue the composition.",
    "PARTIALLY_TESTABLE": "Mix of fundamental and empirical objections.",
    "missing": "No L4 stage summary recorded for this composition.",
}

TIPS_METRIC = {
    "runs": "Compose runs aggregated under data/pipeline/compositions/.",
    "compositions": "Total candidate compositions surfaced across all runs.",
    "relevance skipped": (
        "Runs where the Step 1.5 corpus quality gate ran without a relevance "
        "backend (no Codex/API auth, transient error, or --no-strict-gate)."
    ),
    "total LLM wall time": (
        "Sum of duration_s across every llm_calls/<stage>/<NN>/meta.json for every run. "
        "Only available for Phase 1+ runs."
    ),
    "avg per run": "Total LLM wall time divided by run count.",
}

TIPS_COL = {
    "run_id": "UTC timestamp the compose run started.",
    "problem": "Problem statement passed to compose_one. Hover the cell for the full text.",
    "corpus": "Number of papers in the validated corpus after the Step 1.5 quality gate.",
    "dims": "Number of problem dimensions extracted in Stage 2.",
    "comps": "Number of candidate compositions surfaced in Stage 4.",
    "status": (
        "Final state.status: complete | aborted_corpus_gate | dry_run_complete | "
        "no_dimensions_extracted | ..."
    ),
    "llm_s": "Sum of LLM wall time for this run.",
    "rel.skip": "Whether the Step 1.5 relevance check was skipped on this run.",
}

TIPS_OUTCOME = {
    "clean-positive": "Composition beat the relevant baselines with a clear margin.",
    "partial-positive": "Composition beat some baselines but lost to the strongest.",
    "clean-negative": "Composition was empirically falsified against the baselines.",
    "partial-negative": "Composition was close to the baselines but didn't beat any.",
}

TIPS_RECONCILED = {
    "actionable": "L3=weak and L4=PURELY_TESTABLE. Recipe-eligible: a small bounded test could land this as a positive or clean-negative.",
    "sound": "L3 found no surviving falsification. Rare; the strongest positive signal the pipeline emits.",
    "weak_untestable": "L3=weak but L4 found fundamental objections. Testable in principle but not as currently posed; needs reformulation.",
    "invalid": "L3 found a fatal flaw. Rejected regardless of L4.",
    "incomplete": "L3 or L4 stage failed or returned no output.",
}


def _parse_index_outcomes() -> dict[str, dict[str, str]]:
    """Parse findings/INDEX.md and return run_id -> {outcome, experiment, doc}.

    Expected row shape (whitespace flexible):
        | <run_id> | <experiment> | <composition> | **<outcome>** | `<doc>` |

    Rows that don't match are silently skipped — INDEX.md may also contain
    legacy rows without a run_id, prose, etc.
    """
    out: dict[str, dict[str, str]] = {}
    if not INDEX_PATH.exists():
        return out
    valid_outcomes = set(TIPS_OUTCOME.keys())
    for line in INDEX_PATH.read_text().splitlines():
        line = line.strip()
        if not line.startswith("|") or not line.endswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) < 4:
            continue
        run_id = cells[0]
        # Skip header + divider rows.
        if not run_id or run_id == "run_id" or set(run_id) <= set("- :"):
            continue
        # Outcome is wrapped in ** for emphasis in markdown.
        outcome_cell = cells[3]
        if outcome_cell.startswith("**") and outcome_cell.endswith("**"):
            outcome = outcome_cell[2:-2].strip()
        else:
            outcome = outcome_cell.strip()
        if outcome not in valid_outcomes:
            continue
        doc = cells[4].strip("` ") if len(cells) > 4 else ""
        out[run_id] = {
            "outcome": outcome,
            "experiment": cells[1],
            "doc": doc,
        }
    return out


def _aggregate(runs: list[tuple[Path, dict]]) -> dict:
    """Pure summarization step shared by markdown + HTML renderers."""
    l3_counts: Counter[str] = Counter()
    l4_counts: Counter[str] = Counter()
    reconciled_counts: Counter[str] = Counter()
    outcome_counts: Counter[str] = Counter()
    skipped_relevance = 0
    total_llm_seconds = 0.0
    l3_l4_inconsistent_total = 0
    rows: list[dict] = []
    index = _parse_index_outcomes()
    for run_dir, state in runs:
        if state.get("relevance_skipped"):
            skipped_relevance += 1
        llm_time = _total_llm_time(run_dir)
        total_llm_seconds += llm_time
        compositions = state.get("compositions") or []
        l3_counts += _summarize_l3(compositions)
        l4_counts += _summarize_l4(compositions)
        rec_counts, run_inconsistent = _summarize_reconciled(compositions)
        reconciled_counts += rec_counts
        l3_l4_inconsistent_total += run_inconsistent
        run_id = state.get("run_id", run_dir.name)
        entry = index.get(run_id)
        outcome = entry["outcome"] if entry else ""
        if outcome:
            outcome_counts[outcome] += 1
        rows.append({
            "run_id": run_id,
            "problem": (state.get("problem") or "")[:120],
            "corpus_size": state.get("corpus_size", 0),
            "dimensions": len(state.get("dimensions") or []),
            "compositions": len(compositions),
            "status": state.get("status", "?"),
            "llm_seconds": round(llm_time, 1),
            "relevance_skipped": bool(state.get("relevance_skipped")),
            "outcome": outcome,
            "outcome_doc": entry["doc"] if entry else "",
            "outcome_experiment": entry["experiment"] if entry else "",
            "l3_l4_inconsistent": run_inconsistent,
        })
    return {
        "rows": rows,
        "l3_counts": l3_counts,
        "l4_counts": l4_counts,
        "reconciled_counts": reconciled_counts,
        "outcome_counts": outcome_counts,
        "skipped_relevance": skipped_relevance,
        "total_llm_seconds": total_llm_seconds,
        "l3_l4_inconsistent": l3_l4_inconsistent_total,
        "n_runs": len(runs),
    }


def _bar_svg(counts: Counter, tooltips: dict[str, str] | None = None,
             width: int = 360, bar_height: int = 22) -> str:
    """Inline-SVG horizontal bar chart with optional per-label SVG <title>
    tooltips (browser-native; appears on hover). Empty input renders a
    placeholder string."""
    if not counts:
        return '<p class="muted">(no data)</p>'
    items = counts.most_common()
    max_v = max(v for _, v in items)
    gap = 6
    total_h = len(items) * (bar_height + gap)
    bar_max_w = width - 130
    parts = [f'<svg viewBox="0 0 {width} {total_h}" '
             f'class="chart" role="img" aria-label="verdict counts">']
    for i, (label, value) in enumerate(items):
        y = i * (bar_height + gap)
        bw = max(2, int(bar_max_w * value / max_v))
        tip = (tooltips or {}).get(label, "")
        title_xml = f"<title>{_html_escape(tip)}</title>" if tip else ""
        # Group label + bar + count so the tooltip applies to all three.
        parts.append('<g class="row">')
        if title_xml:
            parts.append(title_xml)
        parts.append(
            f'<text x="0" y="{y + bar_height * 0.7:.0f}" class="lbl">{_html_escape(label)}</text>'
        )
        parts.append(
            f'<rect x="110" y="{y}" width="{bw}" height="{bar_height}" '
            f'class="bar" rx="3"/>'
        )
        parts.append(
            f'<text x="{110 + bw + 6}" y="{y + bar_height * 0.7:.0f}" '
            f'class="num">{value}</text>'
        )
        parts.append("</g>")
    parts.append("</svg>")
    return "".join(parts)


_HTML_CSS = """
:root {
  --fg: #1a1a1a;
  --muted: #6b6b6b;
  --accent: #2563eb;
  --bar: #2563eb;
  --bg: #ffffff;
  --row-alt: #f5f5f5;
  --border: #e5e5e5;
}
@media (prefers-color-scheme: dark) {
  :root {
    --fg: #e5e5e5;
    --muted: #9ca3af;
    --accent: #60a5fa;
    --bar: #60a5fa;
    --bg: #0f0f10;
    --row-alt: #1a1a1c;
    --border: #2a2a2c;
  }
}
* { box-sizing: border-box; }
body {
  margin: 0;
  padding: 32px 40px 64px;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif;
  font-size: 14px;
  line-height: 1.5;
  color: var(--fg);
  background: var(--bg);
  max-width: 1100px;
}
h1 { font-size: 22px; margin: 0 0 4px; }
h2 { font-size: 16px; margin: 28px 0 12px; padding-bottom: 4px; border-bottom: 1px solid var(--border); }
.muted { color: var(--muted); }
.metric-row { display: flex; gap: 32px; flex-wrap: wrap; margin: 12px 0; }
.metric { padding: 8px 16px; border: 1px solid var(--border); border-radius: 6px; }
.metric .v { font-size: 22px; font-weight: 600; }
.metric .k { color: var(--muted); font-size: 11px; text-transform: uppercase; letter-spacing: 0.04em; }
.charts { display: grid; grid-template-columns: 1fr 1fr; gap: 24px; margin: 12px 0; }
.chart { width: 100%; }
.chart .lbl { fill: var(--fg); font-size: 12px; font-family: ui-monospace, "SF Mono", monospace; }
.chart .num { fill: var(--muted); font-size: 12px; font-family: ui-monospace, "SF Mono", monospace; }
.chart .bar { fill: var(--bar); }
.chart g.row { cursor: help; }
.chart g.row:hover .bar { opacity: 0.85; }
th[title], .metric[title] { cursor: help; }
th[title] { border-bottom: 1px dotted var(--muted); }
.metric[title] .k { border-bottom: 1px dotted var(--muted); display: inline-block; padding-bottom: 1px; }
.outcome { border-bottom: 1px dotted currentColor; cursor: help; }
.badge {
  display: inline-block;
  padding: 1px 8px;
  border-radius: 9999px;
  font-size: 11px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.03em;
  cursor: help;
}
.badge.positive { background: rgba(16, 185, 129, 0.15); color: #047857; }
.badge.negative { background: rgba(239, 68, 68, 0.15); color: #b91c1c; }
.badge.neutral { background: var(--row-alt); color: var(--muted); }
@media (prefers-color-scheme: dark) {
  .badge.positive { background: rgba(16, 185, 129, 0.20); color: #34d399; }
  .badge.negative { background: rgba(239, 68, 68, 0.20); color: #fca5a5; }
}
tr.row-positive td:first-child { box-shadow: inset 3px 0 0 #10b981; }
tr.row-negative td:first-child { box-shadow: inset 3px 0 0 #ef4444; }
.filter-bar { display: flex; gap: 16px; align-items: center; margin: 8px 0; font-size: 13px; }
.filter-bar label { cursor: pointer; user-select: none; }
.filter-bar input { cursor: pointer; vertical-align: middle; }
table { border-collapse: collapse; width: 100%; font-size: 13px; margin: 8px 0; }
th, td {
  padding: 6px 10px;
  text-align: left;
  border-bottom: 1px solid var(--border);
}
th {
  cursor: pointer;
  user-select: none;
  white-space: nowrap;
  color: var(--muted);
  font-weight: 500;
}
th:hover { color: var(--accent); }
th.sorted::after { content: " ▾"; color: var(--accent); }
th.sorted.asc::after { content: " ▴"; color: var(--accent); }
tbody tr:nth-child(even) { background: var(--row-alt); }
td.num { text-align: right; font-variant-numeric: tabular-nums; }
td.problem { max-width: 460px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
td.warn { color: #b45309; }
.footer { color: var(--muted); font-size: 12px; margin-top: 32px; }
.track-record { font-size: 13px; }
.track-record table th, .track-record table td { padding: 4px 8px; }
.track-record code { background: var(--row-alt); padding: 1px 4px; border-radius: 3px; font-size: 12px; }
"""


_HTML_JS = """
// Tiny vanilla table-sort. Headers click to toggle.
document.querySelectorAll('table.sortable').forEach(table => {
  const headers = table.querySelectorAll('th');
  headers.forEach((th, idx) => {
    th.addEventListener('click', () => {
      const tbody = table.tBodies[0];
      const rows = Array.from(tbody.rows);
      const asc = !th.classList.contains('sorted') || th.classList.contains('asc');
      headers.forEach(h => h.classList.remove('sorted', 'asc'));
      th.classList.add('sorted');
      if (asc) th.classList.add('asc');
      rows.sort((a, b) => {
        const av = a.cells[idx].dataset.sort ?? a.cells[idx].textContent;
        const bv = b.cells[idx].dataset.sort ?? b.cells[idx].textContent;
        const an = parseFloat(av), bn = parseFloat(bv);
        const cmp = !isNaN(an) && !isNaN(bn) ? an - bn : av.localeCompare(bv);
        return asc ? cmp : -cmp;
      });
      rows.forEach(r => tbody.appendChild(r));
    });
  });
});

// Verified-only filter
const verifiedToggle = document.getElementById('only-verified');
if (verifiedToggle) {
  verifiedToggle.addEventListener('change', () => {
    const onlyVerified = verifiedToggle.checked;
    document.querySelectorAll('#runs-table tbody tr').forEach(tr => {
      const outcome = tr.dataset.outcome;
      tr.style.display = (onlyVerified && (!outcome || outcome === 'none')) ? 'none' : '';
    });
  });
}
"""


def _html_escape(s: str) -> str:
    return (
        s.replace("&", "&amp;")
         .replace("<", "&lt;")
         .replace(">", "&gt;")
         .replace('"', "&quot;")
    )


def _render_index_md_as_html() -> str:
    """Render findings/INDEX.md inline. The index file is a small markdown
    document with a single table; we do a minimal best-effort render rather
    than pull in a real markdown library."""
    if not INDEX_PATH.exists():
        return (
            '<p class="muted">findings/INDEX.md not found; create it to enable '
            "per-experiment outcome counts.</p>"
        )
    text = INDEX_PATH.read_text()
    out_lines = []
    in_table = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("# "):
            out_lines.append(f"<h3>{_html_escape(stripped[2:])}</h3>")
        elif stripped.startswith("|") and stripped.endswith("|"):
            if not in_table:
                out_lines.append("<table>")
                in_table = True
            cells = [c.strip() for c in stripped.strip("|").split("|")]
            if all(set(c) <= set("- :") for c in cells):
                continue  # markdown header-divider row (|---|---|)
            # First row after <table> is the header row; subsequent rows are body.
            is_header = not any("<tr>" in ln for ln in out_lines[-3:])
            tag = "th" if is_header else "td"
            out_lines.append(
                "<tr>" + "".join(
                    f"<{tag}>{_format_inline_md(c)}</{tag}>" for c in cells
                ) + "</tr>"
            )
        else:
            if in_table:
                out_lines.append("</table>")
                in_table = False
            if stripped:
                out_lines.append(f"<p>{_format_inline_md(stripped)}</p>")
    if in_table:
        out_lines.append("</table>")
    return "\n".join(out_lines)


def _format_inline_md(s: str) -> str:
    """Tiny inline-markdown: **bold**, `code`, [text](url) become HTML."""
    import re as _re
    out = _html_escape(s)
    # Inline outcome labels (clean-positive / clean-negative / etc.) get a
    # tooltip-decorated <span> so the user sees the definition on hover.
    for outcome, tip in TIPS_OUTCOME.items():
        out = out.replace(
            f"**{outcome}**",
            f'<strong><span class="outcome" title="{_html_escape(tip)}">{outcome}</span></strong>',
        )
    out = _re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", out)
    out = _re.sub(r"`([^`]+)`", r"<code>\1</code>", out)
    out = _re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', out)
    return out


def generate_html_report(root: Path) -> str:
    if not root.exists():
        return _html_empty("No compose runs found at this root.")

    runs: list[tuple[Path, dict]] = []
    for run_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        state = _load_run(run_dir)
        if state is None:
            continue
        runs.append((run_dir, state))

    if not runs:
        return _html_empty("No compose runs found at this root.")

    agg = _aggregate(runs)
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    metric_html = []
    metric_html.append(_metric("runs", agg["n_runs"], TIPS_METRIC["runs"]))
    metric_html.append(_metric("compositions", sum(agg["l3_counts"].values()),
                               TIPS_METRIC["compositions"]))
    metric_html.append(_metric("relevance skipped",
                               f"{agg['skipped_relevance']} / {agg['n_runs']}",
                               TIPS_METRIC["relevance skipped"]))
    if agg["total_llm_seconds"] > 0:
        metric_html.append(_metric("total LLM wall time",
                                   f"{agg['total_llm_seconds']:.0f}s",
                                   TIPS_METRIC["total LLM wall time"]))
        metric_html.append(_metric("avg per run",
                                   f"{agg['total_llm_seconds'] / agg['n_runs']:.1f}s",
                                   TIPS_METRIC["avg per run"]))

    rows_html: list[str] = []
    for r in agg["rows"]:
        skip_cls = ' class="warn"' if r["relevance_skipped"] else ""
        outcome = r.get("outcome", "")
        if outcome:
            badge_cls = "positive" if "positive" in outcome else "negative"
            row_cls = f"row-{badge_cls}"
            tip = TIPS_OUTCOME.get(outcome, "")
            badge_html = (
                f'<span class="badge {badge_cls}" title="{_html_escape(tip)}">'
                f'{_html_escape(outcome)}</span>'
            )
            sort_key = outcome
        else:
            row_cls = "row-unverified"
            badge_html = '<span class="muted">—</span>'
            sort_key = "zzz_unverified"  # sort unverified last
        rows_html.append(
            f'<tr class="{row_cls}" data-outcome="{_html_escape(outcome) or "none"}">'
            f'<td>{_html_escape(r["run_id"])}</td>'
            f'<td class="problem" title="{_html_escape(r["problem"])}">'
            f'{_html_escape(r["problem"])}</td>'
            f'<td data-sort="{_html_escape(sort_key)}">{badge_html}</td>'
            f'<td class="num" data-sort="{r["corpus_size"]}">{r["corpus_size"]}</td>'
            f'<td class="num" data-sort="{r["dimensions"]}">{r["dimensions"]}</td>'
            f'<td class="num" data-sort="{r["compositions"]}">{r["compositions"]}</td>'
            f'<td>{_html_escape(r["status"])}</td>'
            f'<td class="num" data-sort="{r["llm_seconds"]}">{r["llm_seconds"]}</td>'
            f'<td{skip_cls}>{"yes" if r["relevance_skipped"] else "no"}</td>'
            "</tr>"
        )

    # Outcome summary chip above the table
    outcome_chips = []
    for outcome in ("clean-positive", "partial-positive", "partial-negative", "clean-negative"):
        n = agg["outcome_counts"].get(outcome, 0)
        if n == 0:
            continue
        cls = "positive" if "positive" in outcome else "negative"
        tip = TIPS_OUTCOME.get(outcome, "")
        outcome_chips.append(
            f'<span class="badge {cls}" title="{_html_escape(tip)}">'
            f"{n} × {outcome}</span>"
        )
    if not outcome_chips:
        outcome_chips.append(
            '<span class="muted">no verified runs in this set</span>'
        )
    outcome_summary = (
        '<div class="metric-row" style="margin-bottom: 12px;">'
        + " ".join(outcome_chips) + "</div>"
    )

    raw_payload = {
        "generated": generated,
        "root": str(root),
        "n_runs": agg["n_runs"],
        "l3_counts": dict(agg["l3_counts"]),
        "l4_counts": dict(agg["l4_counts"]),
        "skipped_relevance": agg["skipped_relevance"],
        "total_llm_seconds": agg["total_llm_seconds"],
        "rows": agg["rows"],
    }

    body = f"""
<header>
  <h1>Congruity calibration dashboard</h1>
  <p class="muted">Generated {generated} · aggregating {agg['n_runs']} compose run(s) under <code>{_html_escape(str(root))}</code></p>
  <div class="metric-row">{''.join(metric_html)}</div>
</header>

<section>
  <h2>Verdict mix</h2>
  <div class="charts">
    <div>
      <h3>L3 (adversarial)</h3>
      {_bar_svg(agg['l3_counts'], TIPS_L3)}
    </div>
    <div>
      <h3>L4 (testability)</h3>
      {_bar_svg(agg['l4_counts'], TIPS_L4)}
    </div>
  </div>
  <h3 style="margin-top: 18px;">Reconciled L3+L4 verdict</h3>
  <p class="muted" style="margin: 4px 0 8px;">Single combined status per composition. Use this for "what really happened"; raw L3 and L4 are above for audit.</p>
  <div class="charts">
    <div>
      {_bar_svg(agg['reconciled_counts'], TIPS_RECONCILED)}
    </div>
    <div>
      {_l3_l4_inconsistency_callout(agg)}
    </div>
  </div>
</section>

<section>
  <h2>Runs</h2>
  {outcome_summary}
  <div class="filter-bar">
    <label><input type="checkbox" id="only-verified"/> Show only verified runs</label>
    <span class="muted">· Click a column header to sort. Verified runs show a colored badge and a left edge bar (green = positive, red = negative).</span>
  </div>
  <table class="sortable" id="runs-table">
    <thead>
      <tr>
        <th title="{_html_escape(TIPS_COL['run_id'])}">run_id</th>
        <th title="{_html_escape(TIPS_COL['problem'])}">problem</th>
        <th title="Verified outcome from findings/INDEX.md. Blank means no verification doc references this run_id.">outcome</th>
        <th title="{_html_escape(TIPS_COL['corpus'])}">corpus</th>
        <th title="{_html_escape(TIPS_COL['dims'])}">dims</th>
        <th title="{_html_escape(TIPS_COL['comps'])}">comps</th>
        <th title="{_html_escape(TIPS_COL['status'])}">status</th>
        <th title="{_html_escape(TIPS_COL['llm_s'])}">llm_s</th>
        <th title="{_html_escape(TIPS_COL['rel.skip'])}">rel.skip</th>
      </tr>
    </thead>
    <tbody>
      {''.join(rows_html)}
    </tbody>
  </table>
</section>

<section class="track-record">
  <h2>Verified track record</h2>
  {_render_index_md_as_html()}
</section>

<footer class="footer">
  <p>Source: <code>pipeline/calibration.py</code>. Re-run <code>uv run python -m pipeline.calibration</code> to refresh.</p>
  <details>
    <summary>raw JSON (export)</summary>
    <pre>{_html_escape(json.dumps(raw_payload, indent=2, default=str))}</pre>
  </details>
</footer>
"""
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>Congruity calibration dashboard</title>
  <style>{_HTML_CSS}</style>
</head>
<body>
{body}
<script>{_HTML_JS}</script>
</body>
</html>
"""


def _l3_l4_inconsistency_callout(agg: dict) -> str:
    """Sidebar card next to the reconciled-verdict chart. Calls out the
    count of compositions where L3 and L4 told different stories."""
    n = agg.get("l3_l4_inconsistent", 0)
    total_comps = sum(agg.get("reconciled_counts", {}).values())
    if total_comps == 0:
        return '<p class="muted">(no compositions to reconcile)</p>'
    if n == 0:
        return (
            '<div class="metric" title="Number of compositions where L3 and L4 produced '
            'logically conflicting outputs (e.g. L3=invalid but L4=PURELY_TESTABLE).">'
            '<div class="v">0</div><div class="k">L3 vs L4 disagreements</div>'
            f'<p class="muted" style="margin: 6px 0 0; font-size: 12px;">'
            f'all {total_comps} compositions had consistent L3↔L4 outputs.</p></div>'
        )
    pct = 100.0 * n / total_comps
    return (
        '<div class="metric" style="border-color: #b45309;" '
        'title="Compositions where L3 (adversarial verdict) and L4 (testability) '
        'disagreed. Inspect these by hand — likely worth tightening prompts.">'
        f'<div class="v">{n}</div>'
        '<div class="k">L3 vs L4 disagreements</div>'
        f'<p class="muted" style="margin: 6px 0 0; font-size: 12px;">'
        f'{n} of {total_comps} compositions ({pct:.0f}%) had L3 and L4 telling '
        f'different stories — typically L3=invalid + L4=PURELY_TESTABLE.</p></div>'
    )


def _metric(key: str, value, tip: str = "") -> str:
    title_attr = f' title="{_html_escape(tip)}"' if tip else ""
    return (
        f'<div class="metric"{title_attr}>'
        f'<div class="v">{_html_escape(str(value))}</div>'
        f'<div class="k">{_html_escape(key)}</div></div>'
    )


def _html_empty(msg: str) -> str:
    return f"""<!doctype html>
<html><head><meta charset="utf-8"/><title>Calibration</title></head>
<body><h1>Calibration</h1><p>{_html_escape(msg)}</p></body></html>
"""


def main() -> int:
    p = argparse.ArgumentParser(description="Generate the Congruity calibration report")
    p.add_argument("--root", default=str(DEFAULT_ROOT),
                   help="root directory holding compose run subdirectories")
    p.add_argument("--out", default=None,
                   help="write report here; default: findings/calibration-<today>.<ext>")
    p.add_argument("--format", choices=["html", "md"], default="html",
                   help="output format (default: html)")
    args = p.parse_args()
    root = Path(args.root)

    fmt = args.format
    if args.out:
        out_path = Path(args.out)
        if out_path.suffix.lower() in (".html", ".htm"):
            fmt = "html"
        elif out_path.suffix.lower() == ".md":
            fmt = "md"
    else:
        today = datetime.now(timezone.utc).strftime("%Y%m%d")
        ext = "html" if fmt == "html" else "md"
        out_path = REPO_ROOT / "findings" / f"calibration-{today}.{ext}"

    if fmt == "html":
        report = generate_html_report(root)
    else:
        report = generate_report(root)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report)
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
