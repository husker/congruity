"""Funnel + economics report: pass rates per stage and cost per valid label.

Reads data/pipeline/compositions/*/state.json, data/pipeline/outcomes.jsonl,
and every llm_calls/*/meta.json (compose runs and experiments/auto) to report:

  - corpus gate pass rates per run
  - compositions and recipes surfaced
  - outcome grades and confidence distribution
  - codex call counts + wall-clock as the cost proxy
  - codex minutes per valid (non-quarantined) label

Usage:
    uv run python scripts/funnel_report.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def collect(repo: Path = REPO) -> dict:
    comps_dir = repo / "data" / "pipeline" / "compositions"
    runs = []
    for state_path in sorted(comps_dir.glob("*/state.json")):
        try:
            s = json.loads(state_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        gate = s.get("corpus_gate") or {}
        comps = s.get("compositions") or []
        runs.append({
            "run_id": s.get("run_id", state_path.parent.name),
            "corpus": s.get("corpus_size", 0),
            "gate_passed": gate.get("passed"),
            "gate_total": gate.get("total"),
            "compositions": len(comps),
            "recipes": sum(1 for c in comps
                           if (c.get("verification") or {}).get("recipe")),
            "status": s.get("status", ""),
        })

    outcomes = []
    ledger = repo / "data" / "pipeline" / "outcomes.jsonl"
    if ledger.exists():
        for line in ledger.read_text().splitlines():
            try:
                outcomes.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    codex_calls, codex_seconds = 0, 0.0
    for meta in repo.glob("data/pipeline/compositions/*/llm_calls/*/*/meta.json"):
        codex_calls += 1
        try:
            codex_seconds += float(json.loads(meta.read_text()).get("duration_s", 0))
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            pass
    for meta in repo.glob("experiments/auto/**/meta.json"):
        codex_calls += 1
        try:
            codex_seconds += float(json.loads(meta.read_text()).get("duration_s", 0))
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            pass

    return {"runs": runs, "outcomes": outcomes,
            "codex_calls": codex_calls, "codex_seconds": codex_seconds}


def main() -> int:
    data = collect()

    print("== compose runs ==")
    print(f"{'run':18s} {'corpus':>6s} {'gate':>9s} {'comps':>5s} {'recipes':>7s}  status")
    for r in data["runs"]:
        gate = (f"{r['gate_passed']}/{r['gate_total']}"
                if r["gate_passed"] is not None else "—")
        print(f"{r['run_id']:18s} {r['corpus']:6d} {gate:>9s} "
              f"{r['compositions']:5d} {r['recipes']:7d}  {r['status']}")

    outs = data["outcomes"]
    grades = Counter(o.get("grade") for o in outs)
    confs = Counter(o.get("confidence") for o in outs)
    valid_labels = sum(1 for o in outs
                       if o.get("valid") and o.get("confidence") != "quarantined")
    print("\n== outcomes ledger ==")
    print(f"  total: {len(outs)}  grades: {dict(grades)}  confidence: {dict(confs)}")

    minutes = data["codex_seconds"] / 60
    print("\n== economics (codex as cost proxy) ==")
    print(f"  codex calls: {data['codex_calls']}  wall-clock: {minutes:.0f} min")
    if valid_labels:
        print(f"  codex minutes per valid label: {minutes / valid_labels:.0f}  "
              f"({valid_labels} valid non-quarantined labels)")
    else:
        print("  no valid non-quarantined labels yet")
    return 0


if __name__ == "__main__":
    sys.exit(main())
