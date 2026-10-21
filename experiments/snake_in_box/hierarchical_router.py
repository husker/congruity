"""Hierarchical fragment router (v2): recursive multi-crossing assembly.

v1 measured that multi-crossing at the TOP split is geometrically
self-defeating (an occupied Q8 layer forecloses re-entry; Q9 plateau 155).
Wynn's record profile (1, 2, 8 crossings at axes 8, 7, 6) says the weave
belongs at the DEEP axes. This router recurses: a fragment in Q_m is built
by 2-layer routing at bit m-1, whose sub-fragments are built by routing in
Q_(m-1), down to a base dimension where raw exact DFS takes over.

Context threads down cleanly because twin-adjacency IS the matching:
  - a FROZEN vertex (same-space, adjacency = chord) projects as FROZEN in
    its own child layer, and as a LABEL-BLOCK in the twin layer;
  - a BLOCKED label (label-only exclusion) projects as BLOCKED in its
    own child layer.
The per-level crossing budgets are configuration: {9: 1, 8: 2, 7: 4}
mirrors the record's architecture; sweeping them explores the family.

Every assembled snake is pack-validated; nothing internal is trusted.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from experiments.snake_in_box import fastcore
from experiments.snake_in_box.baselines import validate_snake
from experiments.snake_in_box.canon import canonical_form

CERTS = Path("data/snake_in_box/certificates")
LEDGER = Path("data/snake_in_box/hrouter_ledger.jsonl")

# rough per-run instrumentation (reset in main): stage pruning pressure and
# how often probe rescoring changed the surviving top-beam set
STATS = {"stages": 0, "cands": 0, "probe_moved": 0,
         "leaf_calls": 0, "leaf_exhausted": 0}

DEFAULT_KS = {9: 1, 8: 2, 7: 4}
BASE_M = 6


def _stop_points(frag_len: int, step: int) -> list[int]:
    pts = list(range(max(4, step), frag_len, step)) + [frag_len]
    return sorted({p for p in pts if 1 <= p <= frag_len}, reverse=True)


def _adj_any(m: int, v: int, group: set[int]) -> bool:
    return any((v ^ (1 << b)) in group for b in range(m))


def hroute(m: int, start: int, frozen: set[int], blocked: set[int],
           ks: dict[int, int], node_cap: int, beam_width: int = 4,
           stop_step: int = 6, probe_cap: int = 0,
           probe_top: int = 16, rng_seed: int = -1,
           leaf_pool: int = 1, final_cap: int = 0,
           probe_deep: bool = False,
           min_stops: dict[int, int] | None = None) -> list[int]:
    """Longest-found induced path in Q_m from `start` under the context
    (frozen: adjacency-excluded; blocked: label-excluded). Recursive
    2-layer routing above BASE_M, exact DFS at/below it.

    probe_cap > 0 enables bilevel stop-point scoring: each legal crossing
    is ranked by committed length + a node-capped exact complement probe
    of the target layer it crosses into (what the stop point LEAVES, not
    just what it committed — the criterion behind Q8=95 and the 185).
    Only the probe_top candidates by committed length are probed; the
    rest keep their committed length, a valid lower bound on the score."""
    if m <= BASE_M or m not in ks:
        frag, _n, _e = fastcore.complete_frozen(
            m, [start], sorted(frozen - {start}), sorted(blocked - {start}),
            node_cap, rng_seed=rng_seed)
        STATS["leaf_calls"] += 1
        STATS["leaf_exhausted"] += int(_e)
        return frag

    half = 1 << (m - 1)
    mask = half - 1

    def project(vs: set[int], layer: int) -> tuple[set[int], set[int]]:
        """(frozen_in_layer, twin_blocks_from_other_layer) for child level."""
        same = {v & mask for v in vs if (v >> (m - 1)) & 1 == layer}
        twin = {v & mask for v in vs if (v >> (m - 1)) & 1 != layer}
        return same, twin

    # states: (path_in_Qm, layer_of_end, used0, used1, crossings)
    start_layer = (start >> (m - 1)) & 1
    init = ([start], start_layer,
            {start & mask} if start_layer == 0 else set(),
            {start & mask} if start_layer == 1 else set(), 0)
    beam = [init]
    best: list[int] = [start]

    for _stage in range(ks[m] + 1):
        cands = []
        for path, layer, u0, u1, crossings in beam:
            own, other = (u0, u1) if layer == 0 else (u1, u0)
            f_same, f_twin = project(frozen, layer)
            b_same, _ = project(blocked, layer)
            child_frozen = f_same | own
            child_blocked = b_same | f_twin | other
            cur = path[-1] & mask
            # leaf pool: when the child is a leaf (exact search, not a
            # routed object), pull K diverse fragments — deterministic
            # first, seeded alternates after — and pool stop points
            # across all of them. Shape diversity at the leaves is what
            # the diverse-bottom pool gave the four-subcube 185.
            child_is_leaf = (m - 1 <= BASE_M) or ((m - 1) not in ks)
            # cap schedule: fragments with crossings still to spend stay
            # shallow (deep-greedy early fragments hog the cube — measured
            # 168 vs 175); the FINAL complement of each level gets the
            # deep cap and no pool (pure depth, nothing after it).
            is_final = crossings >= ks[m]
            child_cap = final_cap if (is_final and final_cap) else node_cap
            n_frags = 1 if is_final else \
                (leaf_pool if child_is_leaf else 1)
            frags, seen_frags = [], set()
            for i in range(n_frags):
                seed = rng_seed if i == 0 else \
                    1_000_003 * i + (abs(rng_seed) if rng_seed > 0 else 7)
                fr = hroute(m - 1, cur, child_frozen, child_blocked,
                            ks, child_cap, beam_width, stop_step,
                            probe_cap, probe_top, seed, leaf_pool,
                            final_cap, probe_deep, min_stops)
                key = tuple(fr)
                if len(fr) >= 2 and key not in seen_frags:
                    seen_frags.add(key)
                    frags.append(fr)
            for frag in frags:
              for stop in _stop_points(len(frag) - 1, stop_step):
                f = frag[: stop + 1]
                lifted = [v | (layer << (m - 1)) for v in f[1:]]
                new_path = path + lifted
                new_own = own | set(f)
                if len(new_path) > len(best):
                    best = new_path
                # balance gate: the record's four Q7 cells are near-
                # uniform (48/47/48/48 vertices); ours droop to 42-43.
                # Forbid crossings that leave the current cell under-
                # filled (len(f) = vertices committed this fragment).
                if (min_stops and len(f) < min_stops.get(m, 0)):
                    continue
                if crossings < ks[m]:
                    end = f[-1]
                    # crossing legality in the TARGET layer: the crossing
                    # vertex must be unused there, not blocked there (parent
                    # label-blocks project by layer!), and not adjacent to
                    # anything committed or frozen there
                    tgt_frozen, _ = project(frozen, 1 - layer)
                    tgt_blocked, _ = project(blocked, 1 - layer)
                    tgt_committed = other | tgt_frozen
                    if (end not in tgt_committed
                            and end not in tgt_blocked
                            and not _adj_any(m - 1, end, tgt_committed)):
                        crossed = new_path + [ (end | ((1 - layer) << (m - 1))) ]
                        nu0, nu1 = (new_own, other | {end}) if layer == 0 \
                            else (other | {end}, new_own)
                        # probe context = the child context the NEXT stage
                        # would compute for this state (frozen/blocked in
                        # the target layer)
                        cands.append((len(crossed),
                                      (end, tgt_frozen | other,
                                       tgt_blocked | f_same | new_own),
                                      (crossed, 1 - layer, nu0, nu1,
                                       crossings + 1)))
        if not cands:
            break
        STATS["stages"] += 1
        STATS["cands"] += len(cands)
        if probe_cap > 0:
            # bilevel stop-point scoring: rank a crossing by committed
            # length + what it LEAVES in the layer it enters (node-capped
            # exact complement probe). Unprobed tail keeps committed
            # length — a lower bound on the same score.
            cands.sort(key=lambda t: -t[0])
            base_top = {id(t[2]) for t in cands[:beam_width]}
            rescored = []
            for i, (clen, (p_end, p_frozen, p_blocked), st) in \
                    enumerate(cands):
                score = clen
                if i < probe_top:
                    if probe_deep and (m - 1) in ks and m - 1 > BASE_M:
                        # downstream value = a small ROUTED assembly of
                        # the target layer, not a flat complement — ranks
                        # level-9 crossings by what the whole top is worth
                        pf = hroute(m - 1, p_end, p_frozen, p_blocked,
                                    ks, probe_cap, 2, stop_step * 2,
                                    0, 0, -1, 1, 0)
                    else:
                        pf, _n, _e = fastcore.complete_frozen(
                            m - 1, [p_end], sorted(p_frozen - {p_end}),
                            sorted(p_blocked - {p_end}), probe_cap,
                            rng_seed=-1)
                    score = clen + len(pf) - 1
                rescored.append((score, st))
            cands = rescored
            cands.sort(key=lambda t: -t[0])
            if {id(t[1]) for t in cands[:beam_width]} != base_top:
                STATS["probe_moved"] += 1
        else:
            cands = [(s, st) for s, _ctx, st in cands]
        cands.sort(key=lambda t: -t[0])
        # occupancy-stratified pruning (v1 lesson)
        bands: dict[int, list] = {}
        for _s, st in cands:
            band = min(len(st[2]), len(st[3])) // 8
            bands.setdefault(band, []).append(st)
        per = max(1, beam_width // max(1, len(bands)))
        beam = []
        for band in sorted(bands, reverse=True):
            beam.extend(bands[band][:per])
        beam = beam[:beam_width]
    return best


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ks", default="9:1,8:2,7:4",
                    help="per-level crossing budgets, e.g. 9:1,8:2,7:6")
    ap.add_argument("--node-cap", type=int, default=100_000_000)
    ap.add_argument("--beam", type=int, default=4)
    ap.add_argument("--stop-step", type=int, default=6)
    ap.add_argument("--probe-cap", type=int, default=0,
                    help="node cap for bilevel stop-point probes "
                         "(0 = greedy v2 behavior)")
    ap.add_argument("--probe-top", type=int, default=16,
                    help="probe only this many candidates per stage")
    ap.add_argument("--rng-seed", type=int, default=-1,
                    help="randomize base-fragment DFS tie-breaking "
                         "(-1 = deterministic); portfolio diversifier")
    ap.add_argument("--leaf-pool", type=int, default=1,
                    help="diverse leaf fragments per state (deterministic "
                         "+ K-1 seeded); stop points pooled across all")
    ap.add_argument("--final-cap", type=int, default=0,
                    help="deep node cap for each level's final complement "
                         "(0 = use --node-cap everywhere)")
    ap.add_argument("--min-stops", default="",
                    help="per-level minimum fragment vertices at a "
                         "crossing stop, ks syntax (e.g. 9:92,8:46) — "
                         "balance gate from the Wynn cell-size profile")
    ap.add_argument("--out", type=Path, default=None,
                    help="write best path JSON here regardless of length "
                         "(input to route_then_deepen)")
    ap.add_argument("--probe-deep", action="store_true",
                    help="probe routed target layers with a small "
                         "recursive route (downstream assembly value) "
                         "instead of a flat complement search")
    args = ap.parse_args(argv)

    min_stops = ({int(k): int(v) for k, v in
                  (kv.split(":") for kv in args.min_stops.split(","))}
                 if args.min_stops else None)
    ks = {int(k): int(v) for k, v in
          (kv.split(":") for kv in args.ks.split(","))}
    t0 = time.time()
    STATS.update(stages=0, cands=0, probe_moved=0,
                 leaf_calls=0, leaf_exhausted=0)
    snake = hroute(9, 0, set(), set(), ks, args.node_cap,
                   beam_width=args.beam, stop_step=args.stop_step,
                   probe_cap=args.probe_cap, probe_top=args.probe_top,
                   rng_seed=args.rng_seed, leaf_pool=args.leaf_pool,
                   final_cap=args.final_cap, probe_deep=args.probe_deep,
                   min_stops=min_stops)
    ok, length = validate_snake(9, snake)
    print(f"hroute ks={ks} beam={args.beam} step={args.stop_step} "
          f"probe={args.probe_cap}: "
          f"{int(length) if ok else 'INVALID'}e ({time.time()-t0:.0f}s)",
          flush=True)
    print(f"  stats: {STATS['stages']} stages, {STATS['cands']} cands, "
          f"probe moved top-beam in {STATS['probe_moved']} stages, "
          f"leaves {STATS['leaf_exhausted']}/{STATS['leaf_calls']} "
          f"exhausted", flush=True)
    if args.out and ok:
        args.out.write_text(json.dumps(
            {"n": 9, "length": int(length), "path": snake,
             "ks": args.ks, "beam": args.beam, "stop_step": args.stop_step,
             "node_cap": args.node_cap, "final_cap": args.final_cap,
             "probe_cap": args.probe_cap, "leaf_pool": args.leaf_pool}))
        print(f"  path -> {args.out}", flush=True)
    if ok:
        entry = {"ks": args.ks, "beam": args.beam,
                 "stop_step": args.stop_step, "node_cap": args.node_cap,
                 "probe_cap": args.probe_cap, "probe_top": args.probe_top,
                 "rng_seed": args.rng_seed, "leaf_pool": args.leaf_pool,
                 "final_cap": args.final_cap, "min_stops": args.min_stops,
                 "probe_deep": args.probe_deep,
                 "length": int(length), "canon": canonical_form(snake)[:64]}
        with LEDGER.open("a") as f:
            f.write(json.dumps(entry) + "\n")
        if int(length) > 185:
            (CERTS / f"q9_len{int(length)}_hrouted.json").write_text(
                json.dumps({"n": 9, "length": int(length), "seq": snake,
                            "method": f"hierarchical router {args.ks}"}))
            print("  <<< WRITTEN"
                  + ("  *** WORLD RECORD ***" if int(length) > 190 else ""),
                  flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
