"""Exhaustive completion of a stored snake certificate (hybrid exact step).

Every miner attempt starts from vertex 0 and searches stochastically; the
certificates it leaves behind are never exploited. This tool takes a stored
snake, unwinds its last `k` vertices, and exhaustively enumerates every
chord-free completion of the surviving prefix — a deterministic, complete
search of the basin around a known-good solution. If any longer snake shares
that prefix, this finds it; if none does at depth k, that subtree is *proved*
dead and diversity beats depth.

Induced-path DFS with the standard neighbor-count trick: a vertex may extend
the snake iff it is unvisited and has exactly one neighbor already on the
snake (the current endpoint). Results are re-validated with the pack's own
validate_snake before being reported or written.

Usage:
  uv run python -m experiments.snake_in_box.complete_certificate \
      --cert data/snake_in_box/certificates/q7_len49.json \
      --unwind 8,12,16,20 --node-cap 200000000
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from experiments.snake_in_box.baselines import validate_snake


def load_sequence(path: Path) -> tuple[int, list[int]]:
    data = json.loads(path.read_text())
    if "seq" in data:                      # certificate file
        return int(data["n"]), list(data["seq"])
    best = data["best"]                    # record_run state file
    dim = sorted(best, key=lambda d: best[d]["length"])[-1]
    return int(dim), list(best[dim]["seq"])


def complete(n: int, prefix: list[int], node_cap: int, rng=None,
             forbid_first: set[int] | None = None) -> tuple[list[int], int, bool]:
    """Exhaustive DFS over chord-free extensions of `prefix`.

    Returns (best_sequence, nodes_visited, exhausted) where `exhausted` is
    True when the whole subtree was enumerated within the node cap. With
    `rng`, child order is randomized per node (budgeted stochastic-DFS mode);
    `forbid_first` bans specific vertices as the first extension — used by
    primer attempts to force divergence from the primer's own move.

    Delegates to the C core when available and the case is a shape it
    supports (deterministic order, or a randomized run with at most one
    forbidden first move — the exhaustive-completion and primer hot paths).
    """
    from experiments.snake_in_box import fastcore
    if fastcore.CORE_AVAILABLE and (forbid_first is None or len(forbid_first) <= 1):
        ff = next(iter(forbid_first)) if forbid_first else -1
        rng_seed = rng.randrange(1 << 62) if rng is not None else -1
        return fastcore.complete(n, prefix, node_cap, rng_seed=rng_seed,
                                 forbid_first=ff)
    return _complete_py(n, prefix, node_cap, rng, forbid_first)


def _complete_py(n: int, prefix: list[int], node_cap: int, rng=None,
                 forbid_first: set[int] | None = None) -> tuple[list[int], int, bool]:
    """Pure-Python reference for complete() — the correctness anchor and the
    fallback when no C compiler is available."""
    size = 1 << n
    neighbor_count = [0] * size
    on_snake = [False] * size
    for v in prefix:
        on_snake[v] = True
        for b in range(n):
            neighbor_count[v ^ (1 << b)] += 1

    best = list(prefix)
    path = list(prefix)
    nodes = 0
    exhausted = True

    def dfs(cur: int) -> None:
        nonlocal nodes, best, exhausted
        if nodes >= node_cap:
            exhausted = False
            return
        nodes += 1
        bits = list(range(n))
        if rng is not None:
            rng.shuffle(bits)
        for b in bits:
            v = cur ^ (1 << b)
            if on_snake[v] or neighbor_count[v] != 1:
                continue
            if forbid_first and len(path) == len(prefix) and v in forbid_first:
                continue
            on_snake[v] = True
            path.append(v)
            for bb in range(n):
                neighbor_count[v ^ (1 << bb)] += 1
            if len(path) > len(best):
                best = list(path)
            dfs(v)
            for bb in range(n):
                neighbor_count[v ^ (1 << bb)] -= 1
            path.pop()
            on_snake[v] = False

    sys.setrecursionlimit(size + 100)
    dfs(prefix[-1])
    return best, nodes, exhausted


def primer_attempt(n: int, primer: list[int], seed: int, budget: int) -> list[int]:
    """Meta-NRPA-style primer playout (Dang et al., AAAI-23): replay a
    random-depth prefix of the primer certificate, force one divergent
    choice, then spend the attempt budget on randomized capped DFS from
    there. One DFS node = one candidate-extension evaluation, so budgets
    are comparable with solver attempts."""
    import random
    rng = random.Random(seed)
    length = len(primer)
    for _ in range(8):
        k = rng.randint(2, max(3, length * 3 // 4))
        prefix = primer[: length - k]
        banned = {primer[length - k]}  # the primer's own next move
        best, _nodes, _exh = complete(n, prefix, budget, rng=rng,
                                      forbid_first=banned)
        if len(best) > len(prefix):   # divergence existed at this depth
            return best
    return list(primer)


def polish_sequence(n: int, seq: list[int], max_unwind: int,
                    node_cap: int = 3_000_000) -> list[int]:
    """Exact-completion polish for a miner attempt: exhaustively complete the
    snake's prefixes at unwind depths 4, 8, ..., max_unwind (node-capped per
    depth) and return the best sequence found (>= the input). Used by
    record_run --polish so every stochastic attempt gets its local basin
    searched exactly instead of being discarded."""
    ok, _ = validate_snake(n, seq)
    if not ok or len(seq) < 6:
        return seq
    best = list(seq)
    # A path has two ends and reversal is a hypercube symmetry: polish both.
    for oriented in (list(seq), list(reversed(seq))):
        for k in range(4, max_unwind + 1, 4):
            if k >= len(oriented) - 1:
                break
            found, _nodes, _exhausted = complete(
                n, oriented[: len(oriented) - k], node_cap)
            if len(found) > len(best):
                best = found
    return best


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cert", required=True, type=Path,
                    help="certificate JSON or record_run state file")
    ap.add_argument("--unwind", default="8,12,16,20",
                    help="comma-separated unwind depths (vertices removed)")
    ap.add_argument("--node-cap", type=int, default=200_000_000)
    args = ap.parse_args(argv)

    n, seq = load_sequence(args.cert)
    ok, length = validate_snake(n, seq)
    if not ok:
        print(f"stored sequence failed validation ({args.cert})", file=sys.stderr)
        return 1
    print(f"certificate: Q{n} length {length} ({len(seq)} vertices) from {args.cert}")

    overall_best = list(seq)
    for k in (int(x) for x in args.unwind.split(",")):
        if k >= len(seq) - 1:
            print(f"  unwind {k}: skipped (would remove the whole snake)")
            continue
        prefix = seq[: len(seq) - k]
        t0 = time.monotonic()
        found, nodes, exhausted = complete(n, prefix, args.node_cap)
        dt = time.monotonic() - t0
        verdict = "EXHAUSTED" if exhausted else "node-cap hit"
        marker = ""
        if len(found) > len(overall_best):
            overall_best = found
            marker = "  <-- IMPROVEMENT"
        print(f"  unwind {k:3d}: best {len(found) - 1} edges | "
              f"{nodes:,} nodes | {dt:.1f}s | {verdict}{marker}")

    new_len = len(overall_best) - 1
    ok, _ = validate_snake(n, overall_best)
    if not ok:
        print("INTERNAL ERROR: best completion failed validation", file=sys.stderr)
        return 1
    if new_len > length:
        out = args.cert.parent / f"q{n}_len{new_len}_completed.json"
        out.write_text(json.dumps({
            "n": n, "length": new_len, "seq": overall_best,
            "method": "complete_certificate", "source_cert": str(args.cert)}, indent=1))
        print(f"\nNEW BEST: length {new_len} (was {length}) — written to {out}")
    else:
        print(f"\nno improvement: {length} stands (deepest searched subtrees proved dead)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
