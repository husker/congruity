"""Wang-Landau density-of-states estimation for snake-in-the-box.

Estimates S(L) = log N(L), the log-count of directed induced paths
(snakes) of length L in Q_n. The strategic question it answers: does
the entropy collapse near the record (=> the census game is the right
investment) or is there a fat tail above 185 (=> keep hunting)?

Method: flat-histogram random walk over ALL snakes with grow/shrink
moves at either end — ergodic without constraint softening (any snake
shrinks to a point and regrows). Wang-Landau acceptance
min(1, exp(g(L) - g(L')) * hastings) with hastings = 9 for grow and
1/9 for shrink (proposal asymmetry: grow picks one of 9 directions,
shrink picks an end). Known small bias at the L=0/1 boundary (the two
ends of a single vertex coincide); irrelevant to the high-L tail and
visible in calibration if it ever matters.

Normalization: g is defined up to a constant; S(L) is anchored at the
exactly-known bin N(0) = 2^n.

Calibration: exact_counts(n) enumerates directed snakes from vertex 0
(vertex-transitivity: total = 2^n * from-0 count). Feasible through
n=5 in seconds; n=6 minutes.

Checkpointing: full state (g, H, f, path, rng) dumped every chunk;
rerunning with the same --checkpoint resumes. Kill-safe by design.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import Counter
from pathlib import Path

SNAKE_MAX = {3: 4, 4: 7, 5: 13, 6: 26, 7: 50, 8: 98, 9: 200}


def exact_counts(n: int) -> dict[int, int]:
    """N(L): directed induced paths of length L, exact, by DFS from
    vertex 0 times 2^n (vertex-transitivity)."""
    size = 1 << n
    on = bytearray(size)
    nc = bytearray(size)
    counts: Counter[int] = Counter()

    def grow_ok(w: int) -> bool:
        return not on[w] and nc[w] == 1

    def dfs(v: int, length: int) -> None:
        counts[length] += 1
        for b in range(n):
            w = v ^ (1 << b)
            if grow_ok(w):
                on[w] = 1
                for c in range(n):
                    nc[w ^ (1 << c)] += 1
                dfs(w, length + 1)
                on[w] = 0
                for c in range(n):
                    nc[w ^ (1 << c)] -= 1

    on[0] = 1
    for c in range(n):
        nc[1 << c] += 1
    dfs(0, 0)
    return {length: cnt * size for length, cnt in sorted(counts.items())}


def apply_sigma(n: int, sigma: list[int], x: int) -> int:
    """Apply the coordinate permutation sigma to the bits of x."""
    y = 0
    for b in range(n):
        if x >> b & 1:
            y |= 1 << sigma[b]
    return y


def pivot(n: int, path: list[int], i: int, sigma: list[int]) -> list[int] | None:
    """Fixed-length structural move: reflect/permute the suffix path[i+1:] about
    joint v_i by the coordinate permutation sigma (w -> v_i ^ sigma(w ^ v_i)).
    A coordinate permutation is a hypercube isometry fixing v_i, so the suffix's
    internal induced-path structure and the joint edge are preserved
    automatically; only prefix<->suffix collisions and chords need checking.
    Returns the new path (same length) or None if the pivot is illegal. This is
    the fixed-length basin-connecting move that restores within-window
    ergodicity (see findings/windowed-wl-calibration-tension-20260706.md)."""
    vi = path[i]
    prefix = path[: i + 1]
    pset = set(prefix)
    new_suffix = [vi ^ apply_sigma(n, sigma, w ^ vi) for w in path[i + 1:]]
    for j, w in enumerate(new_suffix):
        if w in pset:
            return None
        for b in range(n):
            u = w ^ (1 << b)
            if u in pset and not (j == 0 and u == vi):
                return None
    return prefix + new_suffix


class WLState:
    def __init__(self, n: int, lmax: int, seed: int = 0, lmin: int = 0,
                 p_pivot: float = 0.0):
        self.n, self.lmax, self.lmin = n, lmax, lmin
        self.p_pivot = p_pivot
        size = 1 << n
        self.g = [0.0] * (lmax + 1)
        self.h = [0] * (lmax + 1)
        self.f = 1.0
        self.moves = 0
        self.rng = random.Random(seed)
        self.on = bytearray(size)
        self.nc = bytearray(size)
        v0 = 0
        self.path = [v0]
        self.on[v0] = 1
        for b in range(n):
            self.nc[v0 ^ (1 << b)] += 1
        self.best_seen = 0

    # -- persistence ----------------------------------------------------
    def dump(self, fp: Path) -> None:
        # stage + atomic rename: a kill mid-write must not destroy the only
        # checkpoint of a multi-day run
        tmp = fp.with_name(fp.name + ".tmp")
        tmp.write_text(json.dumps({
            "n": self.n, "lmax": self.lmax, "lmin": self.lmin,
            "p_pivot": self.p_pivot,
            "g": self.g, "h": self.h,
            "f": self.f, "moves": self.moves, "path": self.path,
            "best_seen": self.best_seen, "rng": self.rng.getstate()}))
        tmp.replace(fp)

    @classmethod
    def load(cls, fp: Path) -> "WLState":
        d = json.loads(fp.read_text())
        st = cls(d["n"], d["lmax"], lmin=d.get("lmin", 0),
                 p_pivot=d.get("p_pivot", 0.0))
        st.g, st.h, st.f = d["g"], d["h"], d["f"]
        st.moves, st.best_seen = d["moves"], d["best_seen"]
        rng_state = d["rng"]
        st.rng.setstate((rng_state[0], tuple(rng_state[1]), rng_state[2]))
        st.path = d["path"]
        st._rebuild_occupancy()
        return st

    def _rebuild_occupancy(self) -> None:
        """Rebuild the on/nc occupancy arrays from self.path."""
        size = 1 << self.n
        self.on = bytearray(size)
        self.nc = bytearray(size)
        for v in self.path:
            self.on[v] = 1
            for b in range(self.n):
                self.nc[v ^ (1 << b)] += 1

    def seed_from(self, snake) -> None:
        """Initialise the walk from a known snake: occupancy rebuilt from
        its vertex path, truncated to fit within lmax. A prefix of an
        induced path is an induced path, so truncation stays legal."""
        self.path = list(snake)[: self.lmax + 1]
        self._rebuild_occupancy()
        self.best_seen = max(self.best_seen, len(self.path) - 1)

    # -- the chain -------------------------------------------------------
    def run(self, chunk_moves: int) -> None:
        n, lmax, lmin = self.n, self.lmax, self.lmin
        g, h, rng = self.g, self.h, self.rng
        on, nc, path = self.on, self.nc, self.path
        log9 = math.log(float(n))    # grow proposes one of n directions
        f = self.f
        p_pivot = self.p_pivot
        L = len(path) - 1
        for _ in range(chunk_moves):
            r = rng.random()
            if r < p_pivot and L >= 2:                   # fixed-length pivot
                i = rng.randrange(1, L)
                sigma = list(range(n))
                rng.shuffle(sigma)
                new = pivot(n, path, i, sigma)
                if new is not None and new != path:
                    for v in path[i + 1:]:
                        on[v] = 0
                        for b in range(n):
                            nc[v ^ (1 << b)] -= 1
                    for v in new[i + 1:]:
                        on[v] = 1
                        for b in range(n):
                            nc[v ^ (1 << b)] += 1
                    path[:] = new
            elif r < p_pivot + (1.0 - p_pivot) * 0.5:    # grow
                if L < lmax:
                    head = rng.random() < 0.5
                    e = path[-1] if head else path[0]
                    w = e ^ (1 << rng.randrange(n))
                    if (not on[w] and nc[w] == 1
                            and math.log(rng.random() + 1e-300)
                            < g[L] - g[L + 1] + log9):
                        (path.append(w) if head else path.insert(0, w))
                        on[w] = 1
                        for b in range(n):
                            nc[w ^ (1 << b)] += 1
                        L += 1
                        if L > self.best_seen:
                            self.best_seen = L
            else:                                        # shrink
                if L > lmin and (math.log(rng.random() + 1e-300)
                                 < g[L] - g[L - 1] - log9):
                    head = rng.random() < 0.5
                    w = path.pop() if head else path.pop(0)
                    on[w] = 0
                    for b in range(n):
                        nc[w ^ (1 << b)] -= 1
                    L -= 1
            g[L] += f
            h[L] += 1
        self.f = f
        self.moves += chunk_moves

    def flat_enough(self, threshold: float) -> bool:
        visited = [c for c in self.h if c > 0]
        if len(visited) < 2:
            return False
        return min(visited) >= threshold * (sum(visited) / len(visited))

    def entropy(self) -> dict[int, float]:
        """S(L). For a full run (lmin=0), anchored at N(0) = 2^n. For a
        windowed run (lmin>0) there is no exact in-band anchor, so the raw
        g is returned over visited band bins; stitching fixes the offset."""
        if self.lmin > 0:
            return {L: self.g[L]
                    for L in range(self.lmin, self.lmax + 1)
                    if self.g[L] > 0.0}
        base = self.g[0]
        return {L: self.g[L] - base + self.n * math.log(2.0)
                for L in range(self.lmax + 1) if self.g[L] > 0.0 or L == 0}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--lmax", type=int, default=0)
    ap.add_argument("--f-min", type=float, default=1e-4)
    ap.add_argument("--flat", type=float, default=0.8)
    ap.add_argument("--chunk", type=int, default=200_000)
    ap.add_argument("--max-minutes", type=float, default=8.0,
                    help="wall-clock budget for this invocation (chunked "
                         "foreground pattern; rerun to continue)")
    ap.add_argument("--checkpoint", type=Path, default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    lmax = args.lmax or SNAKE_MAX.get(args.n, 4 * args.n)
    ck = args.checkpoint or Path(
        f"data/snake_in_box/wl_q{args.n}_checkpoint.json")
    if ck.exists():
        st = WLState.load(ck)
        print(f"resumed: f={st.f:.2e} moves={st.moves:,} "
              f"best_seen={st.best_seen}", flush=True)
    else:
        st = WLState(args.n, lmax, args.seed)

    t0 = time.time()
    while st.f > args.f_min and (time.time() - t0) < args.max_minutes * 60:
        st.run(args.chunk)
        if st.flat_enough(args.flat):
            st.f /= 2.0
            st.h = [0] * (st.lmax + 1)
            print(f"f -> {st.f:.2e} at {st.moves:,} moves, "
                  f"best_seen={st.best_seen}", flush=True)
        st.dump(ck)
    s = st.entropy()
    tail = {L: round(v, 2) for L, v in s.items() if L >= max(s) - 12}
    print(f"moves={st.moves:,} f={st.f:.2e} best_seen={st.best_seen}")
    print(f"S(L) tail: {tail}")
    print("done" if st.f <= args.f_min else "budget reached — rerun to "
          "continue from checkpoint", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
