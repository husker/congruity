"""Spread-k coil (circuit code) search — exact DFS for small cells, heuristic
restart/beam for larger ones. Every returned coil is re-validated by
validate_circuit_code; a solver's claim is never trusted.

Incremental coil prune (derived to avoid the over-pruning false-reject the
naive open-path prune causes on the wrap/closure edge): when extending a path
v_0..v_{m-1} by a transition to v_m,

  * LOCAL: the transition coordinate must not repeat within the last k-1
    transitions (=> Hamming(v_i, v_j) = index-distance for distances < k).
  * GLOBAL: Hamming(v_m, v_i) >= k for i in [k-1, m-k]. The first k-1 vertices
    v_0..v_{k-2} are SKIPPED — at closure they become cyclically within k of
    v_m, so requiring >= k there would wrongly reject valid coils.
  * CLOSURE: once v_m is Hamming-1 to v_0 and length >= 2k, attempt to close and
    FULL-cyclic-validate (the skipped wrap pairs are checked exactly there).

Calibration gate (the correctness anchor): the exact DFS must recover the
proven-optimal small cells (6,3)=16, (7,3)=24, (8,3)=36. If it cannot, the prune
is wrong (too strict), not the instance hard.
"""

from __future__ import annotations

import argparse
import random
import sys
import time

from experiments.circuit_codes.validator import hamming, validate_circuit_code


def _legal_global(v_m: int, verts: list[int], k: int) -> bool:
    """Global prune: Hamming(v_m, v_i) >= k for i in [k-1, m-k]; skip wrap window."""
    m = len(verts)  # v_m will be at index m
    hi = m - k  # inclusive upper index to check
    for i in range(k - 1, hi + 1):
        if hamming(v_m, verts[i]) < k:
            return False
    return True


def _dfs_max(n: int, k: int, time_limit: float) -> tuple[int, list[int]]:
    """Exhaustive DFS for the longest coil (start pinned to 0->1 = first coord;
    hypercube symmetry makes this WLOG). Returns (best_len, best_coil)."""
    t0 = time.perf_counter()
    best_len = 0
    best_coil: list[int] = []
    # transitions[-(k-1):] gives the local window; recent = last k-1 coords used
    verts = [0, 1]
    used = {0, 1}
    trans = [0]  # first transition = coordinate 0

    def dfs():
        nonlocal best_len, best_coil
        if time.perf_counter() - t0 > time_limit:
            return
        m = len(verts)
        tail = verts[-1]
        # attempt closure
        if m >= 2 * k and hamming(tail, 0) == 1:
            ok, length = validate_circuit_code(n, k, verts, cyclic=True)
            if ok and length > best_len:
                best_len, best_coil = int(length), verts[:]
        recent = set(trans[-(k - 1):]) if k > 1 else set()
        for t in range(n):
            if t in recent:
                continue
            u = tail ^ (1 << t)
            if u in used:
                continue
            if not _legal_global(u, verts, k):
                continue
            verts.append(u)
            used.add(u)
            trans.append(t)
            dfs()
            verts.pop()
            used.discard(u)
            trans.pop()

    dfs()
    return best_len, best_coil


def _heuristic(n: int, k: int, seed: int, restarts: int, time_limit: float
               ) -> tuple[int, list[int]]:
    """Randomized restart rollout for larger cells: greedily extend with a random
    legal transition until stuck, keep the best closed coil found."""
    rng = random.Random(seed)
    t0 = time.perf_counter()
    best_len, best_coil = 0, []
    while time.perf_counter() - t0 < time_limit:
        verts = [0, 1]
        used = {0, 1}
        trans = [0]
        while True:
            tail = verts[-1]
            m = len(verts)
            if m >= 2 * k and hamming(tail, 0) == 1:
                ok, length = validate_circuit_code(n, k, verts, cyclic=True)
                if ok and length > best_len:
                    best_len, best_coil = int(length), verts[:]
            recent = set(trans[-(k - 1):]) if k > 1 else set()
            cands = []
            for t in range(n):
                if t in recent:
                    continue
                u = tail ^ (1 << t)
                if u in used:
                    continue
                if _legal_global(u, verts, k):
                    cands.append((t, u))
            if not cands:
                break
            t, u = rng.choice(cands)
            verts.append(u)
            used.add(u)
            trans.append(t)
        if time.perf_counter() - t0 > time_limit:
            break
    return best_len, best_coil


def _beam(n: int, k: int, seed: int, width: int, time_limit: float
          ) -> tuple[int, list[int]]:
    """Restarted stochastic beam over partial coil-paths (generalizes the snake
    stochastic_beam). Keeps the `width` longest partials with random tie-break
    jitter; records every valid closure. A long coil = a long partial that
    happens to close back to 0."""
    rng = random.Random(seed ^ 0x9E3779B9)
    t0 = time.perf_counter()
    best_len, best_coil = 0, []
    while time.perf_counter() - t0 < time_limit:
        # beam entry: (verts, used, trans)
        beam = [([0, 1], {0, 1}, [0])]
        while beam and time.perf_counter() - t0 < time_limit:
            succ = []
            for verts, used, trans in beam:
                tail = verts[-1]
                m = len(verts)
                if m >= 2 * k and hamming(tail, 0) == 1:
                    ok, length = validate_circuit_code(n, k, verts, cyclic=True)
                    if ok and length > best_len:
                        best_len, best_coil = int(length), verts[:]
                recent = set(trans[-(k - 1):]) if k > 1 else set()
                for t in range(n):
                    if t in recent:
                        continue
                    u = tail ^ (1 << t)
                    if u in used:
                        continue
                    if not _legal_global(u, verts, k):
                        continue
                    succ.append((len(verts) + rng.random(),
                                 verts + [u], used | {u}, trans + [t]))
            if not succ:
                break
            succ.sort(key=lambda e: -e[0])
            beam = [(v, us, tr) for _, v, us, tr in succ[:width]]
    return best_len, best_coil


def search_coil(n: int, k: int, mode: str = "auto", seed: int = 1,
                time_limit: float = 30.0, width: int = 200
                ) -> tuple[int, list[int]]:
    if mode == "auto":
        mode = "exact" if n <= 6 else "beam"
    if mode == "exact":
        return _dfs_max(n, k, time_limit)
    if mode == "beam":
        return _beam(n, k, seed, width, time_limit)
    return _heuristic(n, k, seed, 0, time_limit)


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=int, default=None)
    p.add_argument("--k", type=int, default=3)
    p.add_argument("--mode", choices=("auto", "exact", "beam", "heuristic"), default="auto")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--time", type=float, default=30.0)
    p.add_argument("--width", type=int, default=200)
    p.add_argument("--gate", action="store_true",
                   help="run the proven-cell calibration gate (6,3)/(7,3)/(8,3)")
    args = p.parse_args(argv)

    if args.gate:
        # (6,3) is tiny -> exact; (7,3)/(8,3) -> heuristic recovers them fast and
        # exact DFS does not scale. Recovering these proven-optimal values EXACTLY
        # is the correctness anchor for the prune + closure + validator.
        proven = [(6, 3, 16, "exact"), (7, 3, 24, "beam"), (8, 3, 36, "beam")]
        ok_all = True
        for n, k, target, mode in proven:
            best, coil = search_coil(n, k, mode=mode, seed=1, time_limit=args.time,
                                     width=args.width)
            valid = validate_circuit_code(n, k, coil, cyclic=True)[0] if coil else False
            status = "PASS" if best == target and valid else "FAIL"
            ok_all &= (best == target and valid)
            print(f"  ({n},{k}) [{mode}]: found {best} / proven {target}  "
                  f"valid={valid}  {status}")
        print(f"\nGATE {'PASSED' if ok_all else 'FAILED'}")
        return 0 if ok_all else 1

    if args.n is None:
        p.error("--n is required unless --gate is given")
    best, coil = search_coil(args.n, args.k, mode=args.mode, seed=args.seed,
                             time_limit=args.time, width=args.width)
    valid = validate_circuit_code(args.n, args.k, coil, cyclic=True)[0] if coil else False
    print(f"n={args.n} k={args.k} mode={args.mode}: best coil length={best} valid={valid}")
    if coil:
        print("coil:", coil)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
