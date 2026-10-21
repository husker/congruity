"""Rescue sweep: re-judge previously gate-rejected compositions under the fixed
gate (prioritize-not-veto ranking + baseline-relative L4).

Two parts:
  1. FREE (no LLM): rank every saved composition by the new priority score and
     flag the per-run top-K. Compositions that were rejected ONLY on L3's verdict
     (L3=invalid but fund=0 / purely_testable=True) are rescued by the ranking
     alone — no re-classification needed.
  2. CODEX (--with-l4): re-run the recalibrated L4 on the rejected compositions
     that carried a fundamental objection (purely_testable=False). Report which
     flip to fund=0 under the baseline-relative criterion.

Only code-testable domains are swept (wet-lab/physics problems and empty
"no composition" placeholders are excluded). Writes a markdown shortlist.

Run:
    uv run python scripts/rescue_sweep.py --out findings/rescue-sweep-<date>.md
    uv run python scripts/rescue_sweep.py --no-l4   # part 1 only (instant)
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.compose import (  # noqa: E402
    DEFAULT_RECIPE_TOP_K,
    _assign_verify_priority,
    _priority_score,
    extract_testable_predictions,
)

REPO = Path(__file__).resolve().parent.parent
COMPOSITIONS = REPO / "data" / "pipeline" / "compositions"

# Runs whose problem is verifiable with code (exclude wet-lab/physics and the
# "no compatible composition" placeholder runs).
EXCLUDE_SUBSTRINGS = (
    "direct air capture", "sorbent",            # chemistry / wet lab
    "superconductivity",                         # physics
    "colchicine", "repurpos",                    # clinical / wet lab
)


def _is_code_testable(problem: str) -> bool:
    p = problem.lower()
    return not any(s in p for s in EXCLUDE_SUBSTRINGS)


def _is_rejected(c: dict) -> bool:
    ver = c.get("verification") or {}
    l3 = (ver.get("L3") or {}).get("verdict")
    vs = c.get("verification_summary") or {}
    return l3 == "invalid" or (vs.get("objections_fundamental") or 0) > 0


def _is_placeholder(c: dict) -> bool:
    return "no compatible" in (c.get("synthesis_claim", "") or "").lower() or \
           "no evidence-grounded" in (c.get("synthesis_claim", "") or "").lower()


def load_runs() -> list[dict]:
    runs = []
    for d in sorted(COMPOSITIONS.iterdir()):
        sp = d / "state.json"
        if not sp.exists():
            continue
        try:
            s = json.loads(sp.read_text())
        except json.JSONDecodeError:
            continue
        if s.get("compositions"):
            runs.append({"run_id": d.name, "problem": s.get("problem", ""),
                         "compositions": s["compositions"]})
    return runs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--date", default="undated")
    ap.add_argument("--no-l4", action="store_true", help="skip the codex L4 re-run")
    ap.add_argument("--top-k", type=int, default=DEFAULT_RECIPE_TOP_K)
    args = ap.parse_args()

    runs = load_runs()
    rescued_by_rank: list[dict] = []   # rejected, fund=0, now top-K by priority
    flipped_by_l4: list[dict] = []     # rejected, fund>0, now fund=0 after new L4
    stayed_rejected: list[dict] = []
    n_rejected = 0

    for r in runs:
        if not _is_code_testable(r["problem"]):
            continue
        comps = [c for c in r["compositions"] if not _is_placeholder(c)]
        _assign_verify_priority(comps, top_k=args.top_k)  # marks verify_recommended
        for c in comps:
            if not _is_rejected(c):
                continue
            n_rejected += 1
            vs = c.get("verification_summary") or {}
            rec = c["verification"]["reconciled"]
            fund = vs.get("objections_fundamental") or 0
            row = {
                "run": r["run_id"], "cid": c.get("id"),
                "problem": r["problem"][:60],
                "l3": (c["verification"].get("L3") or {}).get("verdict"),
                "old_fund": fund,
                "priority": rec.get("priority_score"),
                "recommended": rec.get("verify_recommended"),
                "claim": (c.get("synthesis_claim", "") or "")[:90],
            }
            if fund == 0:
                # Rejected only on L3 verdict → rescued by ranking if now top-K.
                (rescued_by_rank if row["recommended"] else stayed_rejected).append(row)
            elif args.no_l4:
                stayed_rejected.append(row)
            else:
                # Re-run the fixed (baseline-relative) L4.
                with tempfile.TemporaryDirectory() as tmp:
                    v4 = extract_testable_predictions(c, {}, r["problem"],
                                                      run_dir=Path(tmp))
                new_fund = (v4.get("summary") or {}).get("fundamental_count", fund)
                row["new_fund"] = new_fund
                # Recompute priority with the updated testability.
                c["verification"]["L4"] = v4
                c["verification_summary"]["objections_fundamental"] = new_fund
                c["verification_summary"]["purely_testable"] = (new_fund == 0)
                row["new_priority"] = round(_priority_score(c), 4)
                print(f"  L4 re-run {row['run']} {row['cid']}: fund {fund}->{new_fund}",
                      flush=True)
                (flipped_by_l4 if new_fund == 0 else stayed_rejected).append(row)

    # ---- report ----
    out = []
    w = out.append
    w(f"# Rescue sweep — re-judging gate-rejected compositions ({args.date})")
    w("")
    w("Re-evaluates previously gate-rejected compositions under the fixed gate "
      "(prioritize-not-veto ranking + baseline-relative L4). Code-testable "
      "domains only.")
    w("")
    total_rescued = len(rescued_by_rank) + len(flipped_by_l4)
    w(f"- Code-testable rejected compositions examined: **{n_rejected}**")
    w(f"- Rescued by priority ranking alone (L3=invalid, fund=0, now top-K): "
      f"**{len(rescued_by_rank)}**")
    if not args.no_l4:
        w(f"- Flipped fund>0 → fund=0 under baseline-relative L4: "
          f"**{len(flipped_by_l4)}**")
    w(f"- Total rescued (now verify-eligible / testable): **{total_rescued}** "
      f"({100 * total_rescued / max(1, n_rejected):.0f}% of rejected)")
    w(f"- Still rejected: **{len(stayed_rejected)}**")
    w("")

    def table(title, rows, cols):
        w(f"## {title} ({len(rows)})")
        w("")
        if not rows:
            w("_none_")
            w("")
            return
        w("| " + " | ".join(cols) + " |")
        w("|" + "|".join("---" for _ in cols) + "|")
        for r in sorted(rows, key=lambda x: -(x.get("new_priority") or x.get("priority") or 0)):
            w("| " + " | ".join(str(r.get(c.split()[0].lower().replace("-", "_"), "")) for c in cols) + " |")
        w("")

    w("## Rescued by ranking (rejected on L3 verdict only, now top-K)")
    w("")
    w("| run | cid | priority | L3 | claim |")
    w("|---|---|---|---|---|")
    for r in sorted(rescued_by_rank, key=lambda x: -(x["priority"] or 0)):
        w(f"| {r['run']} | {r['cid']} | {r['priority']} | {r['l3']} | {r['claim']} |")
    w("")

    if not args.no_l4:
        w("## Flipped by baseline-relative L4 (fund>0 → fund=0)")
        w("")
        w("| run | cid | old_fund | new_fund | new_priority | claim |")
        w("|---|---|---|---|---|---|")
        for r in sorted(flipped_by_l4, key=lambda x: -(x.get("new_priority") or 0)):
            w(f"| {r['run']} | {r['cid']} | {r['old_fund']} | {r['new_fund']} | "
              f"{r.get('new_priority')} | {r['claim']} |")
        w("")

    w("## Still rejected")
    w("")
    w("| run | cid | old_fund | new_fund | L3 | claim |")
    w("|---|---|---|---|---|---|")
    for r in sorted(stayed_rejected, key=lambda x: (x["run"], x["cid"])):
        w(f"| {r['run']} | {r['cid']} | {r['old_fund']} | {r.get('new_fund', '-')} | "
          f"{r['l3']} | {r['claim']} |")
    w("")

    report = "\n".join(out)
    if args.out:
        args.out.write_text(report)
        print(f"\nwrote {args.out}")
    else:
        print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
