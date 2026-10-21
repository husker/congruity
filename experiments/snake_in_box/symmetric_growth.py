"""Symmetric growth: Wynn's permuted-copy construction as an exact search.

Grow a Q8 snake S vertex by vertex while simultaneously committing its
automorphic image g(S) in the top layer of Q9. The combined object
(bottom = S, top = g(S) lifted by bit 8, joined at one crossing) must stay
chord-free at every step, so the copy constraint prunes during growth —
this is the structure that produced the d9=190 record, run here with
exhaustive/capped DFS instead of 2012-era heuristics.

Construction shape (matching Wynn's 2-fragment, 1-crossing decomposition):
  bottom = S reversed (ends at S[0] = the junction label j)
  top    = g(S) lifted, starting at g(S)[0]; we REQUIRE g(j) = j so the
           junction edge (j, j+256) exists.
Feasibility invariant during growth: labels(S) and labels(g(S)) may share
only j, and both S and g(S) are chord-free Q8 snakes (g(S) automatically is,
being an automorphic image). Final Q9 length = 2*len(S) + 1 edges. A snake S
of 95 edges with a compatible g gives 191 — a new world lower bound.

g ranges over XOR-translations t (g(v) = v ^ t, t != 0) composed with
sampled coordinate permutations. For pure translations, g(j) = j is
impossible (t != 0), so the junction uses labels j and g(j) adjacent via
the top layer instead — handled by the assembler: the crossing edge is
(u, u+256) for u = last-grown bottom vertex whose image alignment permits
it. To keep the search simple and every result machine-verified, we grow
with the disjointness invariant only and run the exact assembler + pack
validator over all endpoint/orientation combinations at harvest time.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

from experiments.snake_in_box.baselines import validate_snake

CERTS = Path("data/snake_in_box/certificates")
STATE = Path("data/snake_in_box/symmetric_growth_state.json")


def _apply(v: int, t: int, perm: list[int] | None, m: int = 8) -> int:
    if perm is not None:
        w = 0
        for b in range(m):
            if (v >> b) & 1:
                w |= 1 << perm[b]
        v = w
    return v ^ t


def grow(t: int, perm: list[int] | None, node_cap: int,
         rng: random.Random | None) -> list[int]:
    """DFS-grow S maximizing length under the paired-disjointness invariant:
    S chord-free in Q8 AND labels(S) ∩ labels(g(S)) = ∅ (junction handled at
    assembly). Returns the best S found within the node cap."""
    size = 256
    on_s = [False] * size      # labels used by S
    on_g = [False] * size      # labels used by g(S)
    nc = [0] * size            # snake-neighbor counts for S's chord check
    path = [0]
    gimg = [_apply(0, t, perm)]
    if gimg[0] == 0:
        return [0]             # g fixes 0 → images collide immediately
    on_s[0] = True
    on_g[gimg[0]] = True
    for b in range(8):
        nc[1 << b] += 1
    best = [0]
    nodes = 0

    def dfs(cur: int) -> None:
        nonlocal nodes, best
        if nodes >= node_cap:
            return
        nodes += 1
        order = list(range(8))
        if rng is not None:
            rng.shuffle(order)
        for b in order:
            v = cur ^ (1 << b)
            gv = _apply(v, t, perm)
            # v must extend S legally; gv must not collide with S or g(S);
            # v must not collide with g(S)
            if on_s[v] or nc[v] != 1 or on_g[v] or on_s[gv] or on_g[gv] or v == gv:
                continue
            on_s[v] = True
            on_g[gv] = True
            path.append(v)
            gimg.append(gv)
            for bb in range(8):
                nc[v ^ (1 << bb)] += 1
            if len(path) > len(best):
                best = list(path)
            dfs(v)
            for bb in range(8):
                nc[v ^ (1 << bb)] -= 1
            path.pop()
            gimg.pop()
            on_s[v] = False
            on_g[gv] = False
            if nodes >= node_cap:
                return
    dfs(0)
    return best


def assemble(S: list[int], t: int, perm: list[int] | None) -> list[int] | None:
    """Best validated Q9 snake from S + g(S) over orientations and junction
    ends; the pack validator is the only referee."""
    G = [_apply(v, t, perm) for v in S]
    best = None
    for s_seq in (S, S[::-1]):
        for g_seq in (G, G[::-1]):
            # junction: bottom ends at u, top starts at w, edge iff u == w
            # cross-layer; with disjoint label sets u==w is impossible, so
            # the junction must be a NEW crossing vertex: bottom end u, top
            # start u requires relabel — instead join via u_bottom -> u_top
            # where u = s_seq[-1] and we prepend u to the top if g-side
            # adjacency allows: top actually starts at g_seq[0]; the edge
            # (s_seq[-1], g_seq[0]+256) exists iff labels are equal — else
            # try inserting the lifted bottom endpoint.
            cand = s_seq + [v | 256 for v in ([s_seq[-1]] + g_seq
                                              if bin(s_seq[-1] ^ g_seq[0]).count("1") == 1
                                              else g_seq)]
            ok, length = validate_snake(9, cand)
            if ok and (best is None or int(length) > len(best) - 1):
                best = cand
    return best


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=8.0)
    ap.add_argument("--node-cap", type=int, default=30_000_000)
    ap.add_argument("--perm-samples", type=int, default=6)
    args = ap.parse_args(argv)

    st = (json.loads(STATE.read_text()) if STATE.exists()
          else {"next_t": 1, "best": 185, "tried": 0})
    rng = random.Random(st["next_t"] * 7919)
    t0 = time.monotonic()
    print(f"resuming at t={st['next_t']}, best {st['best']}e", flush=True)

    while st["next_t"] < 256 and (time.monotonic() - t0) < args.minutes * 60:
        t = st["next_t"]
        st["next_t"] += 1
        perms: list[list[int] | None] = [None]
        for _ in range(args.perm_samples):
            p = list(range(8))
            rng.shuffle(p)
            perms.append(p)
        for perm in perms:
            if (time.monotonic() - t0) > args.minutes * 60:
                break
            S = grow(t, perm, args.node_cap, None)
            st["tried"] += 1
            if len(S) - 1 < 60:   # symmetric halves below 60e can't beat 185
                continue
            cand = assemble(S, t, perm)
            if cand is None:
                continue
            l9 = len(cand) - 1
            if l9 > st["best"]:
                st["best"] = l9
                (CERTS / f"q9_len{l9}_symmetric.json").write_text(json.dumps(
                    {"n": 9, "length": l9, "seq": cand,
                     "method": f"symmetric growth t={t} perm={perm}"}))
                print(f"t={t} perm={perm}: S {len(S)-1}e -> Q9 {l9}e <<< WRITTEN"
                      + ("  *** WORLD RECORD ***" if l9 > 190 else ""), flush=True)
        STATE.write_text(json.dumps(st))
    print(f"chunk done: next_t {st['next_t']}, tried {st['tried']}, "
          f"best {st['best']}e", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
