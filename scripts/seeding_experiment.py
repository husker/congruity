"""Seeding experiment (roadmap #5): does literature composition out-discover
its alternatives at equal search budget?

Runs three autosearch arms on the same composition, pack, and budget:

  composition — candidates seeded from the literature composition (normal)
  unseeded    — same loop, no literature hint; design from first principles
  evolve-sota — same loop, candidates evolve the home-field SOTA directly

Only the composition arm writes the outcomes ledger; control summaries land
in each arm's search dir. The findings table is written to
findings/seeding-experiment-<run>-<cid>-<date>.md.

Usage:
    uv run python scripts/seeding_experiment.py \
        --run 20260611-022616 --cid C1 --pack graph_coloring
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from pipeline.autosearch import orchestrate_search  # noqa: E402

ARMS = ["composition", "unseeded", "evolve-sota"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True)
    ap.add_argument("--cid", required=True)
    ap.add_argument("--pack", required=True)
    ap.add_argument("--population", type=int, default=4)
    ap.add_argument("--generations", type=int, default=3)
    ap.add_argument("--elites", type=int, default=2)
    args = ap.parse_args()

    summaries: dict[str, dict] = {}
    for arm in ARMS:
        print(f"\n========== arm: {arm} ==========", flush=True)
        summaries[arm] = orchestrate_search(
            args.run, args.cid, datetime.now(timezone.utc),
            pack_id=args.pack, population=args.population,
            generations=args.generations, elites=args.elites, arm=arm)
        print(json.dumps(summaries[arm], indent=2), flush=True)

    date = datetime.now(timezone.utc).strftime("%Y%m%d")
    out = REPO / "findings" / f"seeding-experiment-{args.run}-{args.cid}-{date}.md"
    rows = []
    for arm in ARMS:
        s = summaries[arm]
        rows.append(
            f"| {arm} | {s.get('grade', s.get('error', '?'))} "
            f"| {s.get('train_delta_best', '—')} | {s.get('holdout_delta', '—')} "
            f"| {s.get('candidates_evaluated', '—')} |")
    out.write_text(f"""# Seeding experiment — {args.run}/{args.cid} on {args.pack}

_{datetime.now(timezone.utc).isoformat()}. Equal budget per arm:
population={args.population}, generations={args.generations},
elites={args.elites}. Labels come from one holdout confirmation per arm;
only the composition arm writes the outcomes ledger._

| arm | grade | best train delta | holdout delta | candidates |
|---|---|---|---|---|
{chr(10).join(rows)}

Interpretation guide: if the composition arm does not beat both controls on
holdout delta, literature seeding is not paying for itself on this problem
(roadmap #5) — the literature layer should become one seed source among
several rather than the front of the pipeline.
""")
    print(f"\nfindings table: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
