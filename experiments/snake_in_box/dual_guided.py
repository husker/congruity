"""Dual-guided bottom generation — the stage-0.5 column-generation loop.

The master LP (master_lp.py) prices every label's externality on the
rest of the assembly. This harness spends the expensive exact scorer
(bottom_bilevel.score_bottom) only where the duals say the value is:

  1. solve the restricted master over the on-disk fragment pool,
  2. generate a shared candidate pool of diverse Q8 bottoms
     (bottom_bilevel.make_bottom, fresh seeds),
  3. rank by reduced cost = bottom edges - shadow price paid,
  4. A/B: exact-score the top-K (guided arm) vs a uniform-random K
     (control arm) at identical budgets,
  5. report per-arm max/mean Q9; ledger everything (kill-safe: reruns
     skip already-scored candidates by canonical form).

Conflict-core instrumentation rides along: candidates covering the
known 185-topA core (junction 13) are flagged — measured transfer was
zero on router bottoms; this tracks it on make_bottom diversity.

If the guided arm doesn't beat control on exact Q9 yield, the
weighted-DFS C pricing does NOT get funded (HANDOFF, funded program).
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

from experiments.snake_in_box.bottom_bilevel import make_bottom, score_bottom
from experiments.snake_in_box.canon import canonical_form
from experiments.snake_in_box.master_lp import load_pool, solve

LEDGER = Path("data/snake_in_box/dual_guided_ledger.jsonl")

TOPA_CORE = {12, 20, 23, 24, 25, 32, 39, 41, 51, 61, 66, 70, 76, 84, 89,
             98, 101, 105, 106, 108, 112, 115, 120}
TOPA_J8 = 13

# bottom cells pay these seams' prices (indices into SEAMS)
CELL_SEAMS = {(0, 0): {0, 2}, (0, 1): {0, 3}}

# word-view reference: Wynn-190's bottom letter multiplicities, sorted.
# Long assemblies are POLARIZED (tail letters barely spent: ...6,4,1);
# make_bottom candidates are flat (...10,9,1). Arm D selects by shape.
WYNN_BOTTOM_PROFILE = [20, 18, 17, 16, 12, 6, 4, 1]


def polar_dist(bottom: list[int]) -> int:
    """L1 distance between the bottom word's sorted letter-multiplicity
    vector and Wynn's bottom profile (sorted = axis-relabel invariant)."""
    from collections import Counter
    trans = [(a ^ b).bit_length() - 1 for a, b in zip(bottom, bottom[1:])]
    prof = sorted(Counter(trans).values(), reverse=True)
    prof += [0] * (len(WYNN_BOTTOM_PROFILE) - len(prof))
    return sum(abs(a - b) for a, b in zip(prof, WYNN_BOTTOM_PROFILE))


def dual_score(bottom: list[int], duals: dict) -> float:
    """Reduced cost: edges minus shadow price paid by the bottom's
    labels on the seams its two cells border."""
    paid = 0.0
    for v in set(bottom):
        cell = (0, (v >> 7) & 1)
        lab = v & 127
        for si in CELL_SEAMS[cell]:
            paid += duals.get((si, lab), 0.0)
    return (len(bottom) - 1) - paid


def core_hit(bottom: list[int]) -> bool:
    lb_low = {v for v in bottom if v < 128}
    return (TOPA_CORE <= lb_low
            and TOPA_J8 in (bottom[0] & 127, bottom[-1] & 127))


def ledger_canons() -> set[str]:
    if not LEDGER.exists():
        return set()
    return {json.loads(line)["canon"]
            for line in LEDGER.open() if line.strip()}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", type=int, default=40)
    ap.add_argument("--k", type=int, default=6, help="per-arm exact scores")
    ap.add_argument("--seed-start", type=int, default=6000)
    ap.add_argument("--budget-a", type=int, default=1_000_000_000)
    ap.add_argument("--budget-b", type=int, default=300_000_000)
    ap.add_argument("--rng", type=int, default=99, help="control-arm RNG")
    args = ap.parse_args(argv)

    cols = load_pool(["data/snake_in_box/hrouter_*.json",
                      "data/snake_in_box/certificates/q9_len1*_joint.json",
                      "data/snake_in_box/certificates/q9_len190_wynn.json"])
    lp_val, _x, duals = solve(cols)
    print(f"master LP: {len(cols)} columns, bound {lp_val:.1f}, "
          f"{len(duals)} nonzero label prices", flush=True)

    t0 = time.time()
    cache = Path(f"data/snake_in_box/dual_cands_"
                 f"{args.seed_start}_{args.candidates}.json")
    cands = json.loads(cache.read_text()) if cache.exists() else []
    if cands:
        # dual scores depend on the current pool — recompute
        for c in cands:
            c["dual"] = dual_score(c["bottom"], duals)
        print(f"loaded {len(cands)} cached candidates", flush=True)
    if len(cands) < args.candidates:
        seed = (cands[-1]["seed"] + 1) if cands else args.seed_start
        while len(cands) < args.candidates and seed < args.seed_start + 3000:
            b = make_bottom(seed)
            seed += 1
            if b is None:
                continue
            cands.append({"seed": seed - 1, "bottom": b,
                          "edges": len(b) - 1,
                          "dual": dual_score(b, duals),
                          "core_hit": core_hit(b),
                          "canon": canonical_form(b)[:64]})
            cache.write_text(json.dumps(cands))   # checkpoint per candidate
        hits = sum(c["core_hit"] for c in cands)
        print(f"pool at {len(cands)} candidates ({time.time()-t0:.0f}s), "
              f"core hits {hits}/{len(cands)}", flush=True)

    ranked = sorted(cands, key=lambda c: -c["dual"])
    guided = ranked[: args.k]
    rng = random.Random(args.rng)
    control = rng.sample(cands, min(args.k, len(cands)))
    # length arm: kills the confounder — reduced cost includes edges,
    # so guided may just be picking long bottoms. If length-only
    # matches guided, the duals add nothing beyond length.
    length_arm = sorted(cands, key=lambda c: (-c["edges"],
                                              -c["dual"]))[: args.k]
    # polarization arm: closest word-shape to Wynn's bottom, edges
    # tie-break — tests whether the word-domain signature predicts
    # downstream value beyond length
    for c in cands:
        c["polar"] = polar_dist(c["bottom"])
    polar_arm = sorted(cands, key=lambda c: (c["polar"],
                                             -c["edges"]))[: args.k]
    done = ledger_canons()

    def run_arm(name: str, picks: list[dict]) -> list[int]:
        vals = []
        for c in picks:
            if c["canon"] in done:
                prev = [json.loads(line) for line in LEDGER.open()
                        if line.strip()]
                q9 = next(e["q9"] for e in prev if e["canon"] == c["canon"])
            else:
                t = time.time()
                q9, _snake = score_bottom(c["bottom"], args.budget_a,
                                          args.budget_b)
                entry = {**{k: c[k] for k in
                            ("seed", "edges", "dual", "core_hit", "canon")},
                         "arm": name, "q9": q9,
                         "secs": round(time.time() - t, 1)}
                with LEDGER.open("a") as f:
                    f.write(json.dumps(entry) + "\n")
                done.add(c["canon"])
            vals.append(q9)
            print(f"  [{name}] seed {c['seed']} {c['edges']}e "
                  f"dual {c['dual']:.1f} -> Q9 {q9}", flush=True)
        return vals

    gv = run_arm("guided", guided)
    cv = run_arm("control", control)
    lv = run_arm("length", length_arm)
    pv = run_arm("polar", polar_arm)
    print(f"\nA/B/C/D (k={args.k}, budgets {args.budget_a}/{args.budget_b}):")
    print(f"  guided : max {max(gv)}  mean {sum(gv)/len(gv):.1f}")
    print(f"  control: max {max(cv)}  mean {sum(cv)/len(cv):.1f}")
    print(f"  length : max {max(lv)}  mean {sum(lv)/len(lv):.1f}")
    print(f"  polar  : max {max(pv)}  mean {sum(pv)/len(pv):.1f}")
    print(f"total {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
