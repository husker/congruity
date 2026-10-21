# Spread-k circuit codes — campaign day 1: records verified, machinery built + validated, search-strength wall

**Date:** 2026-07-07. Records-lane campaign (chosen over transfer-win hunting
after the hit-rate study). Target: spread-3 circuit codes (k-coils), leading
with the smallest open cell **(9,3)=64**. Reuses the snake-in-the-box campaign's
approach (validator-first, records-table discipline, chunked mining).

## M0 — records verified (authoritative, cross-checked)

A **(d,k) circuit code / k-coil** is an induced cycle in Q_d with
Hamming(x,x') >= min(cycle_distance(x,x'), k). Spread 2 = coil-in-the-box.
**Snake-in-the-box is the spread-2 OPEN-path case** — the machinery generalizes.

Records compiled + cross-checked (Byrnes 2017 Table 3 == Hood et al. 2015 on
all k>=3 cells; closed-form theorems confirm the proven-optimal frontier), in
`experiments/circuit_codes/records.json`. Key facts:

- **Proven-optimal frontier for k=3 stops at d=8 (=36).** First OPEN cell:
  **(9,3)=64** — "the smallest unproven circuit-code cell in existence," open
  since Hood 2015.
- k=4 proven through d=10 (first open (11,4)=70); k=5 through d=11 ((12,5)=60).
- **CAVEAT (load-bearing):** the k=2 column of Byrnes is STALE — do not use it.
  k>=3 is current. And k>=3 records are effectively single-lineage (Hood +
  Byrnes agree; not multiply-confirmed) — any claim needs a hand-check.
- Both outcomes per open cell are publishable: beat the LB (record) or prove it
  optimal (cell closure).

## M1 — spread-k validator: built + VERIFIED

`experiments/circuit_codes/validator.py`: `validate_circuit_code(n, k, seq,
cyclic)` re-derives spread-k validity + length. Correctness anchors
(`tests/test_circuit_code_validator.py`, 12 tests):

- **k=2 open-path reproduces `snake_in_box.baselines.validate_snake` EXACTLY**
  on real snakes from both baselines across n=5,6,7 (the load-bearing
  equivalence — a real snake validates identically under both).
- coil (cyclic) validation correct on a known length-6 spread-2 coil; the
  reflected Gray-code cycle correctly REJECTED (not spread-2); the
  open-vs-cyclic convention distinction tested; spread-3 local-window rejection.

## M2 — search: prune/closure CORRECT, search-strength is the wall

`experiments/circuit_codes/search.py`. The load-bearing derivation was the
**incremental coil prune** (the naive open-path prune over-rejects at the wrap:
a coil's closing edge v_{N-1}->v_0 is Hamming-1 at large path-distance, which
the open-path condition forbids). Correct prune, in transition space:
- LOCAL: transition coordinate not repeated within the last k-1 (=> Hamming =
  index-distance for d<k).
- GLOBAL: Hamming(v_m, v_i) >= k for i in [k-1, m-k] — the first k-1 vertices
  are SKIPPED (they become cyclically within k at closure); full-cyclic
  re-validation catches the wrap pairs exactly at closure.

**Calibration gate (advisor's design — recover PROVEN cells, not the unproven
64, so a shortfall is unambiguous):**

| cell | proven | found | verdict |
|---|---|---|---|
| (6,3) exact | 16 | **16** | PASS (exact, correct) |
| (7,3) beam/exact | 24 | **24** | PASS (beam 30s; exact 90s) |
| (8,3) | 36 | 32 (beam), 6 (naive DFS) | **search too weak** |

The prune + closure + validator are **correct** (two proven cells recovered
EXACTLY). The wall is SEARCH STRENGTH: coils must CLOSE (return to 0), and
neither the naive fixed-order DFS (pathological past n=6) nor the
longest-partial stochastic beam (caps at 32; doesn't steer toward closure)
recovers (8,3)=36 — let alone (9,3). This mirrors the snake campaign, which
needed NRPA / Wang-Landau / routers to go deep; the analogous investment is the
next step here.

## Verdict + next

**Near-certain deliverable so far (per the pre-agreed honest framing):**
authoritative records + a validated spread-k coil validator + a correct
(prune/closure) search, calibrated on the proven cells it can reach. The
campaign is real but gated on search strength.

Next (in order):
1. **A closure-steering search** — the missing piece. Options: bias the beam
   toward returning to 0 (distance-to-start in the score); a proper restart+
   MCTS/NRPA over coil-paths; or reuse `snake_in_box` NRPA/windowed-WL
   generalized to the spread-k coil legal-move + closure. Gate: recover
   (8,3)=36 EXACTLY before trusting any (9,3) number.
2. Once (8,3)=36 recovered -> push **(9,3)**: reproduce 64, then probe >64
   (best length achieved is the near-certain deliverable; a >64 coil would be a
   record -> hand-check vs Hood/Byrnes).
3. **Closure of (9,3)** is a LONG-SHOT with a hard compute stop, NOT a fallback
   (proven frontier sits exactly one dimension below; 11 years open; a
   timed-out exhaustive proves nothing) — attempt only with orderly generation
   over the hyperoctahedral group and a firm cutoff.

## Gotchas
- Records k=2 column stale (use minortriad/OEIS); k>=3 single-lineage (hand-check
  any claim vs Hood 2015 / Byrnes 2017).
- Never trust a solver's length — `validate_circuit_code` re-derives everything.
- Env: `uv run` (uv reinstalled to ~/.local/bin; prepend to PATH), PYTHONUTF8=1.
