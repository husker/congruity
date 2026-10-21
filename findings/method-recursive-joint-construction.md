# Recursive joint construction for snake-in-the-box lower bounds

*Draft for review — 2026-07-04. Status: Q7 = 50 (proven optimum,
recovered), Q8 = 95 (optimum 98), Q9 = 185 (best known 190). Every
artifact machine-validated twice (pack validator + independent
no-shared-code verifier).*

## 1. Problem and positioning

The snake-in-the-box problem asks for the longest induced (chord-free)
path in the hypercube Q_n. Dimensions ≤ 8 are solved by exhaustive
computation (Q7 = 50, Q8 = 98, Östergård–Pettersson); d9+ records are
lower bounds only (190/376/737/1465/2898 for d9–13 per the live
minortriad table, which supersedes the lagging Wikipedia/OEIS values).
Method history: evolutionary search held records until 2012; nested
Monte-Carlo (Kinny) displaced it; the current d9 record (190) came from
Wynn's structured construction (permuted initial transition sequences,
arXiv:1201.1647); d10–13 came from Meta-NRPA (AAAI-23) and later
priming/seeded search.

This document describes the method that reached 185 on a laptop in one
day, its verified structural by-products, and its measured dead ends.

## 2. The engine

Four components, each forced by a measured failure of something simpler:

1. **C bitboard inner loop** (`snake_core.c` via ctypes): rollout,
   exhaustive completion, and complement-constrained search at ~50x
   Python speed. Correctness anchored by node-for-node equality with a
   pure-Python reference.
2. **Complement-constrained exact search** (`sib_complete_blocked`):
   longest induced path avoiding a frozen fragment (blocked vertices
   excluded like visited, contributing no neighbor counts). Introduced
   after rejection pairing of independently mined fragments produced 0
   glues in 82M placements — expected label overlap of two ~50-vertex
   sets in Q7 is ~19; demanding exactly one shared label by sampling is
   hopeless. Fragments must be built jointly, not matched.
3. **Bilevel fragment selection**: fragments scored by
   `len(fragment) + exact-complement-value`, not length. The optimal
   Q7-50 as a bottom half caps its complement at a proven 43; a 48-edge
   fragment selected bilevel admits 46 — "optimality hogs the cube."
4. **Four-subcube assembly**: Q9 = 4 Q7 subcubes (bits 7, 8). Bottom Q8
   = bilevel Q7 fragment + its exact complement; top Q8 = two more
   complement-constrained exact searches against the bottom's shadow;
   single-bit junctions; final object validated end-to-end.
   Result: 185 = 95 + 41 + 47 + 2 junctions.

Scoping note (review-corrected): the *leaves* are exact searches; no
internal node of the assembly is jointly exact. Per-bottom top values
are lower bounds except where explicitly exhausted.

## 3. Verified structural results

- **Wynn-190 anatomy**: transcribed from arXiv:1201.1647 and validated;
  crossing profile (1, 2, 8 on axes 8, 7, 6) is the fossil record of
  recursive doubling. Our 185 doubles on axes 8, 7, 4 (1, 2, 5) — same
  construction family, different nesting, more block-diagonal.
- **End rigidity of the record**: exhaustive completion proves no snake
  > 190 shares Wynn's first ~121 vertices from the tail (unwind 70,
  1.03B nodes) or ~131 from the head (unwind 60, 155M).
- **Mid-body rigidity of the record**: no +1 interior rewrite exists —
  156 windows (all positions, widths 16–40), all exhausted
  (`window_repair.py`; positive control rediscovers the known 49→50).
- **Our 185 is equally locked** (+1 and +2, 304 windows exhausted), and
  640 unwind-diverge-recomplete neighbors found no move.
- Consequence: any 191 differs from both known long snakes wholesale —
  supporting quality-diversity search over local improvement.

## 4. Measured negative results (do not re-run)

| lane | verdict |
|---|---|
| Kinny playout heuristic | deterministic caps: Q7 46, Q8 88, Q9 ~154 |
| FunSearch priority-function evolution | 18 candidates, none beat the Kinny seed |
| NRPA (+ primer warm-start, forced exploration) | plateaus at/below Kinny level |
| rejection pairing of independent fragments | 0 glues / 82M placements |
| randomized DFS restarts | worse and 3.5x slower than deterministic |
| random bottom diversity (71 scored) / neighborhood walk (640 steps) | no improvement on 185 |

## 5. Provenance discipline

Every scored bottom is ledgered (`bottom_ledger.jsonl`) with its
canonical form (transition-sequence relabeling — invariant under
coordinate permutation, XOR translation, reversal; `canon.py`), exact
top value, and search budgets. Window sweeps ledger per-window node
counts and exhaustion flags. Certificates carry method + parameters.
This bookkeeping makes completed enumerations impossibility theorems —
"no 191 with doubling architecture (1, 2, ≤k)" is provable, not
anecdotal, if the bottom class is enumerated.

## 6. Open lanes

1. **Fragment assembly at branch-and-price generality** (multi-crossing,
   multi-fragment stitching over canonical catalogs) — hunts a 191 and
   accumulates impossibility coverage as its failure mode.
2. **Quality-diversity search** (archive over structural descriptors) —
   the only live hypothesis for a non-basin 191.
3. **Symmetric growth** (joint (S, π) search — Wynn's own family):
   full 255-translation sweep in progress at write time.
4. Upper-bound side (SDP/Terwilliger) — untouched by anyone recently;
   out of scope for this machine, flagged as a research direction.

## Sources

- Wynn, *Constructing circuit codes by permuting initial sequences*,
  arXiv:1201.1647 (190-snake transcribed from §"other").
- Dang et al., *Warm-starting NRPA*, AAAI-23 (Meta-NRPA records).
- Östergård & Pettersson (exhaustive Q7/Q8).
- Live record table: https://minortriad.com/snake/ (snapshot in
  `computable_packs/snake_in_box/record_sources/`).
- Independent verifier: audit 2026-07-04 (naive O(L²) checks; all
  certificates pass).
