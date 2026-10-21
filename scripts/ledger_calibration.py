"""Ledger calibration (roadmap #8): which composition features predict outcomes?

Joins data/pipeline/outcomes.jsonl with each composition's compose-time
features (codex confidence, transfer distance, mechanism_map presence) and
cross-tabulates them against the recorded grades. Replaces the n~24
manual-era heuristic ("confidence is the only weakly predictive signal") as
the ledger accrues valid labels — and says so plainly while n is small.

Usage:
    uv run python scripts/ledger_calibration.py
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
LEDGER = REPO / "data" / "pipeline" / "outcomes.jsonl"
COMPOSITIONS = REPO / "data" / "pipeline" / "compositions"

POSITIVE_GRADES = {"transfer-win", "new-record"}


def load_joined() -> list[dict]:
    rows: list[dict] = []
    if not LEDGER.exists():
        return rows
    states: dict[str, dict] = {}
    for line in LEDGER.read_text().splitlines():
        try:
            o = json.loads(line)
        except json.JSONDecodeError:
            continue
        run_id = o.get("run_id", "")
        if run_id not in states:
            sp = COMPOSITIONS / run_id / "state.json"
            try:
                states[run_id] = json.loads(sp.read_text())
            except (OSError, json.JSONDecodeError):
                states[run_id] = {}
        comp = next((c for c in states[run_id].get("compositions", [])
                     if c.get("id") == o.get("cid")), {})
        rows.append({
            "run_id": run_id, "cid": o.get("cid"),
            "grade": o.get("grade"), "valid": o.get("valid"),
            "confidence_label": o.get("confidence"),
            "delta_vs_home": o.get("delta_vs_home"),
            "transfer_distance": o.get("transfer_distance"),
            "compose_confidence": comp.get("confidence"),
            "has_mechanism_map": bool(comp.get("mechanism_map")),
            "searched": "autosearch" in (o.get("notes") or ""),
        })
    return rows


def main() -> int:
    rows = load_joined()
    usable = [r for r in rows if r["valid"]]
    print(f"ledger outcomes: {len(rows)} total, {len(usable)} valid")
    if len(usable) < 10:
        print("NOTE: n < 10 valid labels — nothing here is statistically "
              "meaningful yet; this report becomes informative as the "
              "flywheel accrues labels.")

    by_grade: dict[str, list[dict]] = defaultdict(list)
    for r in usable:
        by_grade[r["grade"]].append(r)

    print(f"\n{'grade':14s} {'n':>3s} {'mean compose conf':>18s} "
          f"{'mean transfer dist':>19s} {'searched':>9s}")
    for g, rs in sorted(by_grade.items()):
        confs = [r["compose_confidence"] for r in rs
                 if isinstance(r["compose_confidence"], (int, float))]
        dists = [r["transfer_distance"] for r in rs
                 if isinstance(r["transfer_distance"], (int, float))]
        mc = f"{sum(confs) / len(confs):.1f}" if confs else "—"
        md = f"{sum(dists) / len(dists):.1f}" if dists else "—"
        ns = sum(1 for r in rs if r["searched"])
        print(f"{g:14s} {len(rs):3d} {mc:>18s} {md:>19s} {ns:>9d}")

    pos = [r for r in usable if r["grade"] in POSITIVE_GRADES]
    if pos:
        print(f"\npositive labels ({len(pos)}):")
        for r in pos:
            print(f"  {r['run_id']}/{r['cid']}: {r['grade']} "
                  f"(compose confidence {r['compose_confidence']}, "
                  f"transfer distance {r['transfer_distance']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
