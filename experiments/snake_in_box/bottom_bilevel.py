"""Bottom-level bilevel sweep for the four-subcube Q9 construction.

Generates diverse Q8 bottom fragments (Q7 pair glues from randomized exact
searches) and scores each by its TRUE objective: the best validated Q9 total
from the four-subcube construction (bottom + complement-constrained topA +
topB). Checkpoints to disk after every bottom so interrupted runs resume;
run with --minutes N to time-box a chunk.

Any validated Q9 > 190 is a new world lower bound; the certificate is
written immediately.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from experiments.snake_in_box import fastcore
from experiments.snake_in_box.baselines import validate_snake
from experiments.snake_in_box.complete_certificate import polish_sequence

STATE = Path("data/snake_in_box/bottom_bilevel_state.json")
CERTS = Path("data/snake_in_box/certificates")
LEDGER = Path("data/snake_in_box/bottom_ledger.jsonl")


def ledger_append(entry: dict) -> None:
    """Provenance ledger: one line per scored bottom — canonical form,
    exact top value, and search budgets. This is the bookkeeping that turns
    a completed enumeration into an impossibility theorem instead of an
    anecdote; kept from the start, per review."""
    with LEDGER.open("a") as f:
        f.write(json.dumps(entry) + "\n")


def ledger_canons() -> set[str]:
    if not LEDGER.exists():
        return set()
    return {json.loads(line)["canon"] for line in LEDGER.open()}


def make_bottom(seed: int) -> list[int] | None:
    """Diverse Q8 bottom: randomized-DFS Q7 fragment, polished, + its exact
    complement, glued across bit 7."""
    A, _, _ = fastcore.complete(7, [0, 1], 30_000_000, rng_seed=seed)
    A = polish_sequence(7, A, 24, node_cap=20_000_000)
    if len(A) - 1 < 45:
        return None
    best = None
    for u in (A[0], A[-1]):
        blocked = [v for v in A if v != u]
        B, _, _ = fastcore.complete_blocked(7, [u], blocked,
                                            500_000_000, rng_seed=-1)
        if len(B) < 44:  # need a decent second half
            continue
        Ao = A if A[-1] == u else A[::-1]
        q8 = Ao + [v | 128 for v in B]
        ok, _l = validate_snake(8, q8)
        if ok and (best is None or len(q8) > len(best)):
            best = q8
    return best


def score_bottom(bottom: list[int], budget_a: int = 4_000_000_000,
                 budget_b: int = 1_000_000_000) -> tuple[int, list[int] | None]:
    """Best validated Q9 total for this bottom over orientation x junction x
    topA-trim. Returns (length, snake)."""
    best_len, best_snake = 0, None
    for variant in (bottom, [v ^ 128 for v in bottom]):
        Lb = set(variant)
        lb_low = {v for v in Lb if v < 128}
        lb_high = {v - 128 for v in Lb if v >= 128}
        for j8 in {variant[0], variant[-1]}:
            if j8 >= 128:
                continue
            bot = variant if variant[-1] == j8 else variant[::-1]
            blockedA = [v for v in lb_low if v != j8]
            topA_full, _, _ = fastcore.complete_blocked(
                7, [j8], blockedA, budget_a, rng_seed=-1)
            for trimA in range(0, 8):
                if trimA >= len(topA_full) - 1:
                    break
                tA = topA_full[: len(topA_full) - trimA] if trimA else topA_full
                j7 = tA[-1]
                blockedB = list((lb_high | set(tA)) - {j7})
                topB, _, _ = fastcore.complete_blocked(
                    7, [j7], blockedB, budget_b, rng_seed=-1)
                snake = bot + [v | 256 for v in tA] + [v | 384 for v in topB]
                ok, l9 = validate_snake(9, snake)
                if ok and int(l9) > best_len:
                    best_len, best_snake = int(l9), snake
    return best_len, best_snake


def eval_seed(seed: int) -> tuple[int, int, int, list[int] | None, dict | None]:
    """(seed, bottom_len, q9_len, snake, ledger_entry)."""
    from experiments.snake_in_box.canon import canonical_form
    bottom = make_bottom(seed)
    if bottom is None or len(bottom) - 1 < 92:
        return seed, 0, 0, None, None
    canon = canonical_form(bottom)
    quick, _ = score_bottom(bottom, budget_a=200_000_000, budget_b=100_000_000)
    if quick < 181:  # prefilter: only promising bottoms get the full budget
        entry = {"canon": canon, "seed": seed, "len8": len(bottom) - 1,
                 "top_quick": quick - len(bottom), "q9": quick,
                 "budgets": [200_000_000, 100_000_000], "exact": False}
        return seed, len(bottom) - 1, quick, None, entry
    full, snake = score_bottom(bottom)
    entry = {"canon": canon, "seed": seed, "len8": len(bottom) - 1,
             "q9": full, "budgets": [4_000_000_000, 1_000_000_000],
             "exact": True}
    return seed, len(bottom) - 1, full, snake, entry


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=8.0)
    ap.add_argument("--seed-start", type=int, default=5000)
    ap.add_argument("--seed-end", type=int, default=9000)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args(argv)

    from concurrent.futures import ProcessPoolExecutor

    state = (json.loads(STATE.read_text())
             if STATE.exists() else {"next_seed": args.seed_start,
                                     "best": 185, "scored": 0})
    t0 = time.monotonic()
    print(f"resuming at seed {state['next_seed']}, best {state['best']}e",
          flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        while (state["next_seed"] < args.seed_end
               and (time.monotonic() - t0) < args.minutes * 60):
            batch = list(range(state["next_seed"],
                               min(state["next_seed"] + args.workers,
                                   args.seed_end)))
            for seed, l8, q9len, snake, entry in pool.map(eval_seed, batch):
                if entry is not None:
                    ledger_append(entry)
                if l8:
                    state["scored"] += 1
                if q9len > state["best"] and snake is not None:
                    state["best"] = q9len
                    CERTS.mkdir(parents=True, exist_ok=True)
                    (CERTS / f"q9_len{q9len}_joint.json").write_text(json.dumps(
                        {"n": 9, "length": q9len, "seq": snake,
                         "method": f"four-subcube bottom-bilevel (seed {seed})"}))
                    print(f"seed {seed}: bottom {l8}e -> Q9 {q9len}e  <<< WRITTEN"
                          + ("  *** WORLD RECORD ***" if q9len > 190 else ""),
                          flush=True)
                elif q9len >= state["best"] - 2 and l8:
                    print(f"seed {seed}: bottom {l8}e -> Q9 ~{q9len}e (near)",
                          flush=True)
            state["next_seed"] = batch[-1] + 1
            STATE.write_text(json.dumps(state))
    print(f"chunk done: next_seed {state['next_seed']}, scored {state['scored']}, "
          f"best {state['best']}e", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
