"""Meet-in-the-middle record attack on Q9 via the solved Q8 layers.

Q9 = two copies of Q8 joined by a perfect matching (split axis = bit 8).
Wynn's 190 record decomposes along that axis into exactly two fragments
(96+95 vertices) — it is a doubling. This tool is a HEURISTIC SAMPLED search
of the same low-crossing stratum (per-pair it samples permutations and takes
the first valid glue; absence of a glue here is NOT a non-existence proof —
see try_glue's docstring):

  1. CATALOG — mine a diverse set of long Q8 induced paths (the C Kinny
     rollout; this doubles as the Milestone-B calibration, target >= 95 edges).
  2. PAIR — for pairs of catalogued Q8 snakes A, B, search the Q8 automorphism
     group for a placement of B in the top layer whose vertex-label set is
     disjoint from A's in the bottom layer, sharing exactly one endpoint label
     (the single crossing). Such a pair glues into a Q9 snake of
     len(A) + len(B) + 1 edges. Two disjoint 98-snakes → ~197 = a new record.

The gluing is verified end-to-end with the pack's validate_snake before any
certificate is written — a candidate is never trusted.

Symmetry: the automorphisms of Q8 are the 2^8 XOR-translations composed with
the 8! coordinate permutations. We search translations exhaustively (256, and
they alone often suffice to separate two snakes' label sets) and a bounded
random sample of permutations. This is a heuristic over the pairing, not an
exhaustive proof — a hit is a real record; the absence of a hit is not a
theorem (that needs the full stratum enumeration).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

from experiments.snake_in_box import fastcore
from experiments.snake_in_box.baselines import validate_snake
from experiments.snake_in_box.complete_certificate import polish_sequence

TOP = 1 << 8  # default split bit (Q8 -> Q9); overridden per dimension below


def split_bit(n: int) -> int:
    """The matching-edge bit that splits Qn into two Q(n-1) layers."""
    return 1 << (n - 1)


def mine_catalog(m: int, count: int, budget: int, min_len: int, seed0: int,
                 polish: int = 0, log=lambda s: None) -> list[list[int]]:
    """Distinct Q_m snakes of >= min_len edges, mined by C Kinny rollouts
    (optionally exact-completion polished). Distinctness is by canonical
    vertex-set signature under XOR-translation, so automorphic copies of one
    snake are not stockpiled."""
    catalog: list[list[int]] = []
    seen: set[frozenset[int]] = set()
    seed = seed0
    best = 0
    while len(catalog) < count and seed < seed0 + 300 * count:
        seq = fastcore.rollout_kinny(m, seed, budget)
        seed += 1
        if polish:
            # C-speed completion affords a much deeper polish cap than the
            # in-miner default; deep unwinds are where the lifts live
            seq = polish_sequence(m, seq, polish, node_cap=50_000_000)
        ok, length = validate_snake(m, seq)
        if not ok:
            continue
        best = max(best, int(length))
        if int(length) < min_len:
            continue
        base = min(seq)
        sig = frozenset(v ^ base for v in seq)
        if sig in seen:
            continue
        seen.add(sig)
        catalog.append(seq)
        log(f"  catalog {len(catalog)}/{count}: len {int(length)} (best seen {best})")
    if not catalog:
        log(f"  (no snake reached min_len {min_len}; best seen {best})")
    return catalog


def catalog_from_pairs(lower: list[list[int]], n: int, perm_samples: int,
                       count: int, min_len: int, seed: int,
                       log=lambda s: None) -> list[list[int]]:
    """Build a Q_n catalog by gluing pairs from a Q_(n-1) catalog (Wynn's
    recursive combining). Collects up to `count` distinct glued snakes of
    >= min_len edges rather than stopping at the first."""
    rng = random.Random(seed)
    out: list[list[int]] = []
    seen: set[frozenset[int]] = set()
    cat = sorted(lower, key=len, reverse=True)
    best = 0
    for i in range(len(cat)):
        for j in range(i, len(cat)):
            glued = try_glue(cat[i], cat[j], n, perm_samples, rng)
            if not glued:
                continue
            best = max(best, len(glued) - 1)
            if len(glued) - 1 < min_len:
                continue
            base = min(glued)
            sig = frozenset(v ^ base for v in glued)
            if sig in seen:
                continue
            seen.add(sig)
            out.append(glued)
            log(f"  Q{n} catalog {len(out)}/{count}: "
                f"{len(cat[i])-1}+{len(cat[j])-1} -> {len(glued)-1}e")
            if len(out) >= count:
                return out
    log(f"  Q{n} catalog: {len(out)} snakes, best glue {best}e")
    return out


def _apply(seq: list[int], m: int, xor: int, perm: list[int] | None) -> list[int]:
    if perm is None:
        return [v ^ xor for v in seq]
    out = []
    for v in seq:
        w = 0
        for b in range(m):
            if (v >> b) & 1:
                w |= 1 << perm[b]
        out.append(w ^ xor)
    return out


def try_glue(a: list[int], b: list[int], n: int, perm_samples: int,
             rng: random.Random) -> list[int] | None:
    """Search automorphic placements of Q_(n-1) snake b that glue to a across
    one matching edge into a validated Q_n snake, or None. a and b live in
    Q_(n-1); the split bit is 1<<(n-1).

    SAMPLED, FIRST-HIT search (audit 2026-07-04): translations are swept
    exhaustively but permutations are randomly sampled, and the FIRST valid
    glue per pair is returned — other distinct glues for the same pair are
    not enumerated. Returning None therefore never proves non-existence, and
    catalogs built from these glues under-count the distinct-glue space.

    Hot path is the feasibility prefilter: vertex sets as arbitrary-precision
    int bitmasks, so "exactly one shared label, and it is an endpoint of a"
    is an AND + popcount instead of building Python sets per placement."""
    m = n - 1
    top_bit = 1 << m
    a_mask = 0
    for v in a:
        a_mask |= 1 << v
    a_ends_mask = (1 << a[0]) | (1 << a[-1])
    perms: list[list[int] | None] = [None]
    for _ in range(perm_samples):
        p = list(range(m))
        rng.shuffle(p)
        perms.append(p)
    # translating a vertex-set mask by e_k swaps bit-blocks of stride 2^k:
    # lo_k = positions whose k-th coordinate is 0
    lo = []
    for k in range(m):
        mk = 0
        for v in range(1 << m):
            if not (v >> k) & 1:
                mk |= 1 << v
        lo.append(mk)

    for perm in perms:
        bt = _apply(b, m, 0, perm)
        b_mask = 0
        for v in bt:
            b_mask |= 1 << v
        xor = 0
        for i in range(1 << m):
            if i:  # Gray-code walk: flip exactly one translation bit
                k = (i & -i).bit_length() - 1
                s = 1 << k
                b_mask = ((b_mask & lo[k]) << s) | ((b_mask >> s) & lo[k])
                xor ^= s
            inter = a_mask & b_mask
            # exactly one shared label — the crossing; any other coincidence
            # is a cross-layer chord through the matching — and it must be an
            # endpoint of a
            if inter == 0 or (inter & (inter - 1)) or not (inter & a_ends_mask):
                continue
            s = inter.bit_length() - 1
            placed = [v ^ xor for v in bt]
            for bi, be in ((0, placed[0]), (-1, placed[-1])):
                if be != s:
                    continue
                snake = _assemble(a, placed, s, bi, top_bit)
                if snake is None:
                    continue
                ok, length = validate_snake(n, snake)
                if ok and int(length) == len(a) + len(placed) - 1:
                    return snake
    return None


def _assemble(a: list[int], b_placed: list[int], shared: int,
              b_join_idx: int, top_bit: int) -> list[int] | None:
    """Bottom snake a (ending at `shared`) + crossing edge + top snake
    b_placed (starting at `shared`, lifted by top_bit)."""
    a_ord = a if a[-1] == shared else a[::-1]
    if a_ord[-1] != shared:
        return None
    b_ord = b_placed if b_join_idx == 0 else b_placed[::-1]
    bottom = list(a_ord)
    top = [v | top_bit for v in b_ord]
    if (bottom[-1] ^ top[0]) != top_bit:
        return None
    return bottom + top


def run(catalog: list[list[int]], n: int, perm_samples: int, seed: int,
        log=lambda s: None) -> list[int] | None:
    rng = random.Random(seed)
    cat = sorted(catalog, key=len, reverse=True)
    for i in range(len(cat)):
        for j in range(i, len(cat)):
            glued = try_glue(cat[i], cat[j], n, perm_samples, rng)
            if glued:
                length = int(validate_snake(n, glued)[1])
                log(f"  GLUE {len(cat[i])-1}+{len(cat[j])-1} -> Q{n} length {length}")
                return glued
        log(f"  paired snake {i} ({len(cat[i])-1}e) against {len(cat)-i} others")
    return None


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(
        description="Recursive meet-in-the-middle: mine Q7 snakes, glue -> Q8, "
                    "glue -> Q9 (Wynn's combining method).")
    ap.add_argument("--catalog", type=int, default=120,
                    help="Q7 base-catalog size")
    ap.add_argument("--budget", type=int, default=20_000_000)
    ap.add_argument("--polish", type=int, default=16,
                    help="exact-completion polish depth for base Q7 snakes")
    ap.add_argument("--q7-min", type=int, default=47)
    ap.add_argument("--q8-min", type=int, default=94)
    ap.add_argument("--q8-count", type=int, default=60)
    ap.add_argument("--perm-samples", type=int, default=96)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args(argv)

    if not fastcore.CORE_AVAILABLE:
        print(f"C core required: {fastcore.LOAD_ERROR}", file=sys.stderr)
        return 1
    p = lambda s: print(s, flush=True)  # noqa: E731

    p(f"[1/3] base Q7 catalog: mining {args.catalog} snakes >= {args.q7_min}e "
      f"(Kinny + polish {args.polish}) ...")
    q7 = mine_catalog(7, args.catalog, args.budget, args.q7_min, args.seed,
                      polish=args.polish, log=p)
    if not q7:
        return 1
    p(f"  Q7 catalog: {len(q7)} snakes, best {max(len(s)-1 for s in q7)}e")

    p(f"[2/3] Q8 catalog by gluing Q7 pairs (target >= {args.q8_min}e) ...")
    q8 = catalog_from_pairs(q7, 8, args.perm_samples, args.q8_count,
                            args.q8_min, args.seed, log=p)
    if not q8:
        p("  no Q8 snake reached the target — recursion stalls at Q7->Q8")
        return 0
    best8 = max(len(s) - 1 for s in q8)
    p(f"  Q8 catalog: {len(q8)} snakes, best {best8}e (proven optimum 98)")

    p(f"[3/3] Q9 gluing over {len(q8)} Q8 snakes ...")
    glued = run(q8, 9, args.perm_samples, args.seed, log=p)
    if glued is None:
        p("  no Q9 gluing in the sampled automorphisms (not a theorem)")
        return 0

    length = int(validate_snake(9, glued)[1])
    out = Path("data/snake_in_box/certificates") / f"q9_len{length}_paired.json"
    out.write_text(json.dumps({"n": 9, "length": length, "seq": glued,
                               "method": "recursive pairing Q7->Q8->Q9"}))
    verdict = "NEW WORLD RECORD" if length > 190 else "below record 190"
    p(f"\nQ9 length {length} — {verdict}\nwritten to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
