"""Neighborhood search in bottom-space for the four-subcube construction.

Random restarts plateaued (71 diverse bottoms, none beat 185). This walks
the neighborhood of the incumbent best bottom instead: unwind k vertices
from either end, force one divergent move (forbid the original
continuation), exactly recomplete — a valid neighboring Q8 snake — and
rescore by the true objective (best validated four-subcube Q9 total).
Hill-climb with sideways acceptance; checkpointed per step; run in
--minutes chunks like bottom_bilevel.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

from experiments.snake_in_box import fastcore
from experiments.snake_in_box.baselines import validate_snake
from experiments.snake_in_box.bottom_bilevel import score_bottom

STATE = Path("data/snake_in_box/bottom_neighborhood_state.json")
CERTS = Path("data/snake_in_box/certificates")


def neighbor(bottom: list[int], rng: random.Random) -> list[int] | None:
    """A valid Q8 snake near `bottom`: reverse maybe, unwind k, forbid the
    original next vertex, exactly recomplete (capped)."""
    seq = bottom[::-1] if rng.random() < 0.5 else list(bottom)
    k = rng.randint(3, min(40, len(seq) - 10))
    prefix = seq[: len(seq) - k]
    banned = seq[len(seq) - k]
    out, _, _ = fastcore.complete(8, prefix, 400_000_000,
                                  rng_seed=rng.randrange(1 << 60),
                                  forbid_first=banned)
    if len(out) - 1 < 90:
        return None
    ok, _ = validate_snake(8, out)
    return out if ok else None


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=8.0)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args(argv)

    if STATE.exists():
        st = json.loads(STATE.read_text())
    else:
        incumbent = json.loads(
            (CERTS / "q8_len95_bilevel.json").read_text())["seq"]
        st = {"bottom": incumbent, "score": 185, "best": 185,
              "steps": 0, "rng_state": args.seed}
    rng = random.Random(st["rng_state"])
    t0 = time.monotonic()
    print(f"resuming: step {st['steps']}, incumbent score {st['score']}, "
          f"best {st['best']}", flush=True)

    while (time.monotonic() - t0) < args.minutes * 60:
        cand = neighbor(st["bottom"], rng)
        st["steps"] += 1
        st["rng_state"] = rng.randrange(1 << 60)
        if cand is None:
            STATE.write_text(json.dumps(st))
            continue
        quick, _ = score_bottom(cand, budget_a=200_000_000,
                                budget_b=100_000_000)
        if quick < st["score"] - 3:
            STATE.write_text(json.dumps(st))
            continue
        full, snake = score_bottom(cand)
        accept = full >= st["score"]  # greedy with sideways moves
        if accept:
            st["bottom"], st["score"] = cand, full
        if full > st["best"] and snake is not None:
            st["best"] = full
            (CERTS / f"q9_len{full}_joint.json").write_text(json.dumps(
                {"n": 9, "length": full, "seq": snake,
                 "method": f"bottom neighborhood search (step {st['steps']})"}))
            print(f"step {st['steps']}: bottom {len(cand)-1}e -> Q9 {full}e "
                  f"<<< WRITTEN" + ("  *** WORLD RECORD ***" if full > 190 else ""),
                  flush=True)
        elif full >= st["score"]:
            print(f"step {st['steps']}: move to {full}e "
                  f"(bottom {len(cand)-1}e)", flush=True)
        STATE.write_text(json.dumps(st))
    print(f"chunk done: steps {st['steps']}, incumbent {st['score']}, "
          f"best {st['best']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
