"""One-command flywheel: compose -> autosearch every recipe-bearing composition.

Chains the whole discovery loop for one domain so it can run unattended
(nightly cron, or after adding a new pack):

  1. compose with --computable-pack context and recipes
  2. for each composition that got a recipe, run the autosearch loop
     (train-split evolution, one holdout confirmation, ledger label)

Example:
    uv run python scripts/flywheel.py --pack graph_coloring \
        --paper-search "graph coloring heuristic tabu search;chromatic number" \
        --problem "Color benchmark graphs with fewer colors than TabuCol..."

Cron (nightly at 2:17am):
    17 2 * * * cd ~/Projects/congruity && uv run python scripts/flywheel.py ... >> data/pipeline/logs/flywheel.log 2>&1
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
from pipeline.compose import compose_one  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pack", required=True, help="search-ready computable pack id")
    ap.add_argument("--problem", required=True)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--paper-search", help="semicolon-separated OpenAlex queries")
    src.add_argument("--topic-list", type=Path)
    ap.add_argument("--max-papers", type=int, default=120)
    ap.add_argument("--max-compositions", type=int, default=None,
                    help="verify at most N compositions (by priority order)")
    ap.add_argument("--population", type=int, default=4)
    ap.add_argument("--generations", type=int, default=3)
    ap.add_argument("--elites", type=int, default=2)
    args = ap.parse_args()

    queries = None
    if args.paper_search:
        queries = [q.strip() for q in args.paper_search.split(";") if q.strip()]

    print("=== flywheel: compose ===", flush=True)
    state = compose_one(
        topic_list_path=None if queries else args.topic_list,
        search_queries=queries,
        problem=args.problem,
        max_papers=args.max_papers,
        computable_pack=args.pack,
        with_recipes=True,
    )
    run_id = str(state.get("run_id") or "")
    comps = state.get("compositions") or []
    if not run_id or not comps:
        # compose_one prints its own diagnostics (gate aborts etc.)
        print(f"flywheel: compose produced no compositions "
              f"(status={state.get('status')}); stopping", flush=True)
        return 1
    with_recipe = [c for c in comps
                   if (c.get("verification") or {}).get("recipe")]
    if args.max_compositions:
        with_recipe = with_recipe[:args.max_compositions]
    print(f"\n=== flywheel: autosearch {len(with_recipe)} compositions ===",
          flush=True)

    summaries = {}
    for c in with_recipe:
        cid = c.get("id")
        if not cid:
            continue
        print(f"\n--- {run_id}/{cid} ---", flush=True)
        summaries[cid] = orchestrate_search(
            run_id, cid, datetime.now(timezone.utc), pack_id=args.pack,
            population=args.population, generations=args.generations,
            elites=args.elites)
        print(json.dumps(summaries[cid], indent=2), flush=True)

    print("\n=== flywheel summary ===")
    for cid, s in summaries.items():
        print(f"  {cid}: {s.get('grade', s.get('error'))} "
              f"(holdout delta {s.get('holdout_delta', '—')})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
