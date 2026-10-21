# Snake-in-the-box record campaign — day 2 (2026-07-03)

Milestone A gate unchanged: recover the proven Q7 optimum (50) before
spending wall-clock on Q9. Day-1 close: budget-mined 48, hypothesis that
the plateau is a knowledge gap (canonical forms / isomorph rejection /
symmetry machinery absent from candidates).

## What ran

1. **Budget lever exhausted.** +60 x 10M-eval attempts on g0_c2 (fresh
   seeds 101-160): zero improvement, best stays 48. Stint distribution
   46x18 / 47x38 / 48x4. Day-1's "+1 edge per ~5x compute" did NOT
   repeat — 48 is a method-class ceiling for g0_c2, not a budget line.
2. **OpenAlex /works recovered** (503-throttled all of day 1). Paper-direct
   compose run `20260703-142800`: 8 problem-anchored + symmetry queries,
   year-min 1990, 141 papers fetched, 44 passed the gate (vs 13 in day-1's
   topic-mode run). 6 compositions, all L3=weak (no signal per calibration
   policy). Top: C1 conf 8 — orderly generation (P27 MAYGEN) + SAT
   circuit-code oracle (P7) + seed snakes from Q_{n-1} (P12) + warm-started
   NRPA rollouts (P11). Cost: 29 codex calls, 40.6 codex-min.
3. **Deeper abstract-only evolution on C1** (pop 6, gens 4, elites 3, seed
   g0_c2, record mode): best 47 (g1_e1), 19 candidates. Third independent
   method family to stall at 47.
4. **Full-text mining shipped** (`autosearch --fulltext`, commit 8e9519b).
   pipeline/fulltext.py: OA PDF resolution (pdf_url -> .pdf landing ->
   arXiv rewrite -> Zenodo records API), pdftotext extraction, method-cue
   paragraph excerpts, per-run cache. C1 coverage: P27, P11, and P12 (the
   seed-snakes paper, recovered via Zenodo); P7's only PDF mirror is a
   dead ETH host.
5. **A/B: same C1 evolution with --fulltext** (identical budget, fresh
   codegen; abstract arm preserved at `search_record7_abstract_only/`):

   | arm | gen-0 fresh best | best overall | candidates at 47 |
   |---|---|---|---|
   | abstract-only | 46 | 47 (g1_e1, gen 1) | 2 |
   | full-text | **47 (g0_c3, gen 0)** | 47 | 4 |

   Full-text did not break the in-search 47 at this population scale, but
   codegen now *starts* where the abstract arm finished after refinement,
   and 2x as many candidates reach the plateau. The knowledge transfer is
   real; the in-search eval budget may be the binding constraint.

## Ladder after day 2 (Q7, proven optimum 50)

| method | Q7 best |
|---|---|
| abstract-bred evolution (3 families, day 1+2) | 47 |
| full-text-bred evolution (in-search) | 47 |
| budget-mined g0_c2 (160 attempts cumulative) | 48 |
| g0_c3 (full-text-bred) long mining | in flight |

## g0_c3 mining result (evening)

48 in 60 x 10M attempts (45x12 / 46x27 / 47x19 / 48x2) — matched the
ceiling in 1/3 of g0_c2's attempts at ~3.6x lower cost per attempt, but
did not break it. Q7 stands at 48 everywhere; gap -2.

## Diagnosis: fitness can't see the tail

`record_command` evaluated candidates at **2 seeds x 2M evals**, but 48s
surface ~2-4 times per 60 attempts at 10M. In-search fitness literally
cannot distinguish a 47-capable method from a 48-capable one — selection
was blind exactly where it mattered. This explains the A/B: full-text
lifted the population (gen-0 47, 2x the 47s) and selection couldn't rank
the top of it.

## Changes made in response

- `record_command` budget: 2x2M -> **3x5M** (15M evals per fitness call;
  900s bench timeout now doubles as a speed filter — slow candidates FAIL,
  which is the right bias for record mining).
- P7 full text recovered via a companion paper: the ISITA'08 mirrors are
  dead/blocked, so `fulltext/P7.txt` holds "Finding Lean Induced Cycles in
  Binary Hypercubes" (Chebiryak, Wahl, Kroening, Haller) — same group,
  same SAT-encoding machinery. Provenance note: not the exact P7 work.

## Night session: the ceiling broke — Q7 = 49

Scaled full-text evolution (pop 8 / gens 3 / elites 3, seed g0_c3,
--fulltext 4/4 excerpts incl. recovered P7, 15M-eval fitness) produced
**g1_e0: the first candidate to score 48 in-search** (refined from the
g0_c3 seed; 18 candidates evaluated). Mining g1_e0 (72 attempts x 10M):

- **Q7 = 49** (seed 60, certificate `q7_len49.json`) — first result past
  48 in the campaign, margin -1 to the proven optimum.
- Tail distribution: 45x7 / 46x27 / 47x31 / **48x8 / 49x1** — an ~13%
  48-rate vs 3-7% for g0_c2/g0_c3, plus the 49.

The full chain that did it: paper-direct corpus -> full-text method
excerpts (incl. seed-snakes P12 + SAT-encoding companion P7) -> fitness
budget able to rank the tail (3x5M) -> refinement of the full-text-bred
seed. Every link was necessary: abstracts-only bred 47-cap candidates,
and the old 2x2M fitness couldn't see 48-capable ones.

Extended stint (180 more attempts) hunting 50 at close. Also this
session: records.json verified against Wikipedia/OEIS (all 9 entries
exact); disk-full incident traced to a 20GB codex marketplace cache
(deleted; atomic state writes survived ENOSPC as designed).

- Ops note: long background runs on this laptop die on sleep — record_run
  and autosearch both resume cleanly (re-run the same command), plan
  around it. (Resolved later in the day: power settings changed, lid up.)

## Late session: cross-field corpus and the pivot method class

James supplied a field map of SIB cousins (SAW/pivot from polymer physics,
HP folding, transfer-matrix, circuit codes, SAT-symmetry record hunts,
nested MC). Compose run `20260703-221128` with the problem reframed as
"constrained embedding of a self-avoiding chain in a symmetric space":
76/199 papers passed the gate; 6 compositions mapping ~1:1 onto the field
list (C1 atmosphere-MCTS conf 8, C2 pivot rearrangements conf 7, C3
core-directed growth, C4 transfer-matrix, C5 replica exchange, C6 ACO).

- **C2 evolution produced the missing operator.** Fresh gen-0 `g0_c7`
  (in-search 48 — best-ever fresh opening): five global moves on the
  coordinate sequence (segment reversal = pivot, block relocation = pull
  move, substitution, span-wise coordinate relabeling = partial
  automorphism, block insertion) + a chord-repair routine. Exactly the
  "validity-preserving global move" the L3 reviewer flagged as the
  critical missing component of the SAW transfer.
- **Mining g0_c7: Q7=49 in 120 attempts** (49 by attempt ~17 vs ~60 for
  g1_e0). Two independent method classes now mine to 49; 50 unbroken.
- **C1 hybrid (atmosphere-MCTS seeded with g0_c7): no advance** — fresh
  best 47, seed remains best at 48. Atmosphere prior adds nothing over
  the pivot machinery (consistent with the L3 "atmosphere = counting
  legal moves" critique).
- records.json verified against Wikipedia/OEIS (all 9 entries exact);
  OPENALEX_API_KEY installed; arXiv fulltext upgrades shipped (f639f7b):
  e-print LaTeX preferred, title-search fallback, both live-verified.

Overnight closers:

- g0_c7 extended mining: 360 total attempts, best 49 (49x3, no 50). The
  pivot class saturates at 49 within this budget scale, like the
  extension class before it.
- C5 replica exchange seeded with g0_c7: fresh best 46, seed stays best
  at 48 in-search — ensemble overhead without tail gain, empirically
  confirming the L3 budget-accounting objection.

## MILESTONE A PASSED — Q7 = 50 (proven optimum recovered)

After the day-2 close, James redirected to options outside the
pipeline's own recommendations. Three lanes: exhaustive completion of
stored certificates, a deep-research sweep, and warm-start/polish.

The completion lane closed the campaign gate within the hour:

- `complete_certificate.py`: unwind a stored snake's last k vertices,
  exhaustively enumerate all chord-free completions (neighbor-count DFS).
- g1_e0's 49: **dead to unwind 36** (262M nodes exhausted) — no 50
  shares even its first 14 vertices. Deeply isolated local optimum.
- **g0_c7's 49: completion at unwind 24 found length 50** (46k nodes,
  0.1s) — the pivot-class snake's first 26 vertices lie on a true
  optimum. Certificate `q7_len50_completed.json`, independently
  re-validated (unique vertices, all edges, zero chords by a
  no-shared-code recompute).

Method story: stochastic pivot-class search located the neighborhood
(49, 24 vertices from an optimum); exact completion finished the job
deterministically. Neither alone sufficed — extension mining saturated
at 49 over 730 attempts, and the *other* 49's basin was proved dead.
Also shipped: `record_run --polish K` (two-ended exact polish of every
attempt, since path reversal is a hypercube symmetry — James's
observation).

**Campaign gate satisfied → graduated to Q9** (best-known 190). First
stint: g0_c7 + --polish 12, 40 x 20M attempts.

## Q9 day 1 (2026-07-04) — three levers converge on 154

Deep-research report (findings/deep-research-snake-methods-20260704.json,
24/25 claims verified 3-0) set the strategy: Meta-NRPA primer loops set
the d10-13 records but reached only 188 at d9; Wynn's permuted-copy
construction holds d9=190; FunSearch's recipe (evolve the priority
function, not whole solvers) is the validated LLM-harness design.

Ran all three cheap levers:

| lever | Q9 best | notes |
|---|---|---|
| from-scratch mining (g0_c7 + 2-ended polish, 42 x 20M) | 154 | baseline |
| primer loop (--primer, Meta-NRPA design, 120 attempts) | 154 | replays hit 154 x12; divergent DFS never exceeded — the missing half is the *adaptive* rollout policy, not the warm start |
| Wynn-construction evolution (compose 20260704-070330 C1 conf 8, 18 fresh candidates) | 154 | best fresh candidate g2_e0 tied the g0_c7 seed |
| best known (Wynn construction) | **190** | |

Diagnosis mirrors Q7's: a method-class ceiling, and the literature names
the classes above it. Next levers, in evidence order: (a) real
policy-adaptation rollouts (NRPA-class) inside the primer loop — the
documented 154->188 mechanism; (b) FunSearch-style narrowing of
autosearch's evolvable surface to the extension priority function;
(c) serious compute on a construction-template search (Wynn's 190 was
not a laptop-scale result). Seed-lift of the optimal Q7 snake still
running at close. All are feature/compute investments — James's call.

## Day-2 close

**Q7 = 49** (two independent method classes, certificates on disk),
from a starting ceiling of 48. The full causal chain, each link
verified necessary: paper-direct corpus -> full-text method excerpts ->
fitness budget that can rank the tail -> refinement (g1_e0, extension
class) and cross-field pivot transfer (g0_c7, rearrangement class).

The 49->50 step resisted: ~730 total mining attempts across both
classes, two hybrid compositions (atmosphere-MCTS, replica exchange)
that failed to beat their seeds. 50 was proven optimal by exhaustive
search; a stochastic method finding it may need orders more attempts or
per-candidate machinery (e.g. true canonical enumeration) that one-shot
codegen has not produced at this scale.

**Open decision for James:** the Milestone A gate reads "recover the
proven Q7 optimum (50) before spending wall-clock on Q9." Q7=49 is 98%
of optimum via methods the pipeline bred autonomously — either relax the
gate and graduate to Q9 (`--record-dim 9`, bar = best-known 190, where
the same machinery hunts open records rather than a proven ceiling), or
keep grinding the last edge (next levers: much longer mining, larger
populations, or a compose targeting exact/canonical enumeration
literature specifically).

Postscript: the Q9 seed-lift closed at 147 (4h of capped DFS extending
the optimal Q7-50 snake) — below the 154 stochastic baseline. Optimality
does not lift: P12's record priming selects lower-dim seeds for
*extensibility*, and an optimal subcube snake appears to over-constrain
its neighborhood one dimension up. Full Q9 day-1 ladder: lift 147 |
from-scratch 154 | primer 154 | construction-bred 154 | record 190.

## Day 3 (2026-07-04): C core, Kinny ceiling, Wynn rigidity, recursion attack

- **C bitboard inner loop shipped** (snake_core.c via ctypes, 53x on
  exhaustive completion; correctness anchored by node-for-node equality
  with the Python reference). Per James: compiled inner loop > any
  hardware upgrade for record hunting.
- **Wynn-190 rigidity theorem (C-scale):** both ends exhausted — tail to
  unwind 70 (1.03B nodes, 32s), head to unwind 60 (155M). No 191 shares
  the record's first ~120 vertices from either end.
- **Kinny heuristic is deterministically capped:** Q7=46, Q8=88 across
  hundreds of seeds — a method ceiling, not budget. It alone matches Q9
  154 in seconds (the earlier all-methods-at-154 convergence was the
  Kinny level).
- **FunSearch arm verdict:** 18 evolved priority functions, none beat
  the Kinny seed (153 unbeaten; best evolved 146) — "a good human
  heuristic is stiff competition" confirmed empirically.
- **Record attack pivoted to recursive pairing** (Wynn's own method):
  pairing_search generalized to glue Q_(n-1) catalogs into Q_n. Naive Q8
  pairing is dead (2x88 = 177 < 190); the recursion needs polished 48-50
  Q7 snakes with diverse label sets. Diagnostic in flight at write time.

## Day 3, PM: recursive joint construction — Q9 = 185, gap 5

The engine that emerged (each piece forced by a measured dead end):

1. **complement-constrained exact search** (`sib_complete_blocked`):
   longest induced path avoiding an existing fragment — rejection
   pairing found 0 glues in 82M placements; joint construction works.
2. **Q8 = 94** (optimal Q7-50 + proven-max 43e complement), then
   **Q8 = 95** by bilevel fragment shaping (48e A chosen for complement
   richness beats the optimal 50 — optimality hogs the cube).
3. **Q9 lifts**: 171 (94-bottom), 172, 176 (deeper caps), then the
   **four-subcube construction** — top Q8 built as two
   complement-constrained Q7 fragments across the bit-7 seam —
   **Q9 = 185** (95 + 41 + 47 + 2 junctions), all validated.
4. Trim sweeps, bottom bit7-flips: 185 stable; topA reaches exhausted
   (43/45 exact vs this bottom). Binding variable = bottom shape.
5. **bottom_bilevel.py**: diverse Q8 bottoms scored by exact
   four-subcube Q9 value, checkpointed, parallel, chunk-runnable.
   ~40 bottoms scored so far; 185 stands.

Ladder: 154 (bred) -> 165 (skeleton@100M) -> 171 -> 176 -> **185**
(joint construction) -> 190 (Wynn record). Levers if the sweep stalls:
local search in bottom-space (pivot-mutate the best bottom + rescore),
symmetric full-depth construction (all four cells jointly), deeper caps.

## Day 3 review corrections (external feedback on the 185)

- **Retracted: the 70/186 label-overlap statistic.** Vertex labels are
  not invariant under the hyperoctahedral group; the number is
  meaningless as basin evidence. Crossing profile + periodicity are the
  legitimate comparanda and independently support "same species,
  different specimen."
- **Precision on "exact":** in the four-subcube assembly the LEAVES are
  exact (each complement-constrained search); no internal node is. The
  top layer composes sequentially (topA then topB given topA) — a
  per-bottom top value is a lower bound, not a joint-top certificate.
  The bottom's "95e + proven-max complement" is likewise exact only
  per-piece.
- **The 5-edge deficit localized:** our bottom layer matches the record
  (96v); the whole gap is the top (90v vs 95v). Since the top search is
  (leaf-)exact given the shadow, the bottom is shadow-hostile — the
  bilevel criterion optimized the bottom in isolation, one level too
  shallow for a four-level assembly. Fix: enumeration over a
  Pareto-diverse canonical bottom pool scored by downstream (top)
  value, with a residual-free-subgraph surrogate to triage.
- **Structural signal:** ours is more block-diagonal (third-axis
  crossings 5 vs Wynn's 8; fragments 45/43/27/25). The single-junction
  schema is a constraint to relax — sweep deep-axis crossing counts.

## Mid-body rigidity theorems (window-repair sweeps, 2026-07-04 evening)

Machinery: freeze prefix+suffix, exact-DFS all chord-free window
replacements at length +delta (control verified: extra=0 recovers the
original everywhere). Ledgers with per-window node counts + exhaustion:
data/snake_in_box/{wynn190,ours185}_window_ledger.json.

- **Wynn-190 is mid-body locked:** 156 windows (all positions, widths
  16-40), all exhausted, no +1 repair exists. With the end-rigidity
  results (dead to unwind 60-70 both ends), the record is proven locally
  optimal under end extension AND any <=40-wide window rewrite. A 191,
  if it exists, is far from the record in edit distance.
- **Ours-185 likewise:** 304 windows (+1 and +2), all exhausted.
- Open scope for the theorem write-up (task #21): wider windows (C port
  makes 48-56 feasible), +2 on Wynn, multi-window simultaneous repair.

## Symmetric-growth sweep: complete, negative (2026-07-04 late)

All 255 XOR-translations swept (x identity + 3 sampled coordinate
permutations each, 4 assembly orientations, 5M-node growth budget):
1,002 configurations, none produced a glue above 185. Scope caveat:
permutations sampled (3/40320 per translation) and growth node-capped —
this bounds the *searched slice* of the permuted-copy family, not the
family. Verdict: at laptop budget, Wynn's own construction family
yields nothing above our four-subcube 185; the basin-native lane is
closed. Live lanes: branch-and-price fragment assembly, QD for the
irregular region.

## Fragment-assembly lane v1 (router): built, validated, plateaus at 155

Beam-routed 2-layer assembly with exact fragment growth
(fragment_router.py; finds Q5 optimum 13/13 after the invariant fix).
Q9 result: 155 — and the failure is the finding: multi-crossing at the
TOP axis is geometrically wrong (an occupied Q8 layer forecloses
re-entry; measured — beam collapses to 1 by stage 3 even with dense
stop points + occupancy-stratified beam). Wynn's 8 crossings live at
his THIRD axis; the top has one. v2 design: hierarchical recursion —
top split k=1-2, each layer fragment itself a routed Q8 object with
frozen/blocked context threaded down (true branch-and-price shape).

## Hierarchical router v2: architecture validated, 152 at first scale

Recursive multi-crossing assembly with clean context projection
(frozen -> frozen + twin-block per layer). Valid at Q7/Q8/Q9 after the
crossing-vs-target-blocked fix (a parent-blocked label could previously
be entered via a deep crossing — caught by the pack validator, located
by chord autopsy). Q9 first scale: 152 (ks 9:1,8:2,7:6, beam 6).

Gap analysis vs our 185: the router is greedy at every level; the 185
came from bilevel fragment-shaping (score fragments by what they LEAVE).
Next integration: stop-point scoring by downstream potential (task #28's
surrogate, or exact complement probes at coarse stops) inside the
recursion. That single change makes the router a superset of the
four-subcube construction; parameter sweeps then explore the full
1-2-k architecture family the record lives in.

## Day 4 (2026-07-05): probe scoring lands, router 152 -> 178

Bilevel stop-point scoring integrated into hierarchical_router.py
(HANDOFF todo #1) and swept (#2). Every step measured, ledger-logged:

- **Probe scoring alone was inert**: instrumentation showed ~3 cands/
  stage against beam 6 — pruning never bound, so no scoring rule could
  matter. Candidate pressure first: dense stops (step 5->1) 152->166,
  beam 12/24 -> 168/169. With pressure, probes give +1 reliably
  (169@12, 170@24 vs greedy 168/169).
- **Exact-leaf family is the unlock** (ks 9:1,8:1 — Q7 leaves become
  complement-constrained exact searches; the four-subcube k=1 case):
  174 in 19s at leaf cap 20M.
- **Uniform deep caps HURT** (150M: 168 vs 20M: 175): greedy-long
  early leaves hog the cube — the "optimality hogs the cube" effect
  at leaf scale.
- **Leaf pools** (K seeded fragments/state, stop points pooled — the
  diverse-bottom-pool move): 175 -> 176.
- **Cap schedule** (shallow early / deep final complements, no pool on
  finals): 176 -> **178** (beam 6, final 300M, 8.8 min).
- Flat vs recursive downstream probes (--probe-deep): no difference at
  178. ks 9:1,8:2 = 178; 9:2,8:1 = 176; deep-axis ks in the routed-7
  family unused (identical trajectories). Seed portfolio worse on
  average (171-174 vs 175 deterministic), thin tail only.

Verdict: router family saturates ~178 at laptop-stint scale, 7 below
our four-subcube 185 — breadth alone doesn't buy the pipeline's
per-bottom depth. Next lever: two-phase route-then-deepen (take router
top-k assemblies, re-run their final complements at 1G+ caps), or
concede the router lane matches-but-not-beats and route compute back
to bottom-pool enumeration (review's Pareto-diverse pool, task #4/#5).

## Day 5 (2026-07-05 late): conflict-core prototype — scoped verdict

James's ideas memo (8 techniques, tagged) triaged: adopted #2 conflict
cores (prototype now) + #1 column generation (staged, LP-first, C
surgery only on measured traction); declined #3-#8 this stint with
reasons logged. Prototype results (core_extract_prototype.py):

- The 185-bottom's topA refutation ("<= 44 from junction 13") is
  carried by **23 of its 48 shadow labels** — greedy deletion, 50
  probes, 20 min, every kept deletion exhaustion-proven at 1G nodes.
- **Transfer is zero** across the router bottom pool: core coverage
  4-10/23, junction never matches (9 bottoms). Diverse pools are too
  spread in label space for 23-label conjunctions to recur.
- Scoped verdict: cores are a prune for LOCAL bottom-space moves
  (pivot-mutations share ~46/48 labels) — attach to
  bottom_neighborhood-style walks and to column-generation pricing,
  NOT to global diversity sweeps. Cost model: ~20 min/core, junction-
  conditioned.

Also: intrinsic tension measured — smaller cores mean more-relaxed
probes mean harder exhaustion proofs (baseline exhausts in 8.6s at 4G;
singles pass took 17 of the 20 min). Core minimization pays refutation
strength back as compute.

Seed portfolio through seed 6: {174, 176, 177, 177, 178, 179, 181};
181 (seed 3) still the router best. Balance gate measured negative
(cells 48/44/46/41 — the ungatable final cell absorbs the damage).

## Day 5 close: portfolio complete; rigidity NOT universal; two 181 specimens

Full 13-draw portfolio at the frozen config (ks 9:1,8:1, beam 6,
step 1, caps 20M/300M, probe 5M/24, pool 4):
det=178, seeds 1-12 = 177,179,181,177,174,176,179,180,179,174,174,174.
Median 177, max 181 (seed 3). Route-then-deepen over the pool:

- **Seed 8's 180 polished to 181** — first deepen gain; the earlier
  "local rigidity is universal" read was premature (5 rigid assemblies
  were a biased early sample). Gain came from end-polish, not windows.
- The two 181s are DISTINCT specimens (canonical forms differ) — both
  feed the master-LP pool (70 cell-fragments now; LP bound stays 191.0,
  integral at Wynn's cells).

Router lane final: 152 -> 181 in one day of measured upgrades. Next
program (funded, see HANDOFF): dual-guided bottom generation vs
Pareto-selection A/B, cores as local-move prunes.

## Day 5, small hours: dual-guidance A/B/C — duals NOT funded

Pre-registered gate: fund weighted-DFS C pricing iff dual-guided
selection beats control on exact Q9 yield. First A/B passed (guided
180/177.2 vs control 177/174.8) but the guided arm's bottoms were
also longer — so a third arm (top-K by edges, same 40-candidate pool,
same budgets) ran the confounder down:

  guided : max 180  mean 177.2
  control: max 177  mean 174.8
  length : max 180  mean 178.3   <- beats guided

5/6 picks overlap; on the differing picks the dual's choice scored
172 (run worst), the length arm's 179. **Verdict: at this pool scale
the LP duals carry no externality signal beyond bottom length; the C
retrofit is NOT funded.** Redirect: taxonomy (#4) / Wang-Landau (#5
on the todo) tier, letter-polarization gate first (cheap).

Also measured: core hits 0/40 on make_bottom diversity (transfer
still zero); three distinct fresh bottoms score 180 at reduced
budgets (1G/300M) — the 185 needed 4G/1G, consistent with prior
sweeps. Master LP itself stays: the bound (191.0, calibrated) and
the pool infrastructure feed the impossibility census regardless.

## Day 5: polarization arm — adopted as tail-finder; the 181 shelf

Arm D (word-shape: L1 distance of sorted bottom letter-multiplicities
to Wynn's [20,18,17,16,12,6,4,1]) on the same 40-candidate pool:

  guided 180/177.2 · control 177/174.8 · length 180/178.3 ·
  **polar 181/177.2**

Polarization does NOT lift the mean but found the experiment's best
bottom (seed 6003, 92e — invisible to length and dual selection).
Verdict: adopted as a TAIL-FINDER alongside length selection, not a
mean-improver. Full-budget rescore (4G/1G): still 181 — finals were
already exhausting. Deepen: rigid.

**The 181 shelf:** three canonically distinct 181s now exist from
three lanes (router seed-3; router-180 + polish; polar make_bottom +
four-subcube). Method-independent convergence at exactly 181 mirrors
the 154 convergence of from-scratch methods. Shelf ladder:
154 (from-scratch) -> 181 (stint-budget assemblies) -> 185 (shaped
bottom + heavyweight budgets) -> 190 (Wynn). Strengthens the
entropy-collapse hypothesis; raises the priority of the Wang-Landau
oracle (todo #5) relative to more assembly variants.

Cheap follow-up filed: bias make_bottom GENERATION toward polarized
words (generate-shaped, not select-shaped) before concluding the
word-gate lane.

## Day 5 close: 100-pool four-arm verdict — the 181 shelf is now statistical

k=8 arms on 100 fresh candidates (140 generated total across pools;
41 distinct bottoms exact-scored):

  guided 181/179.6 · length 181/179.4 · polar 181/177.8 ·
  control 177/174.9

- Duals == length (179.6 vs 179.4): C-pricing stays dead, confirmed
  at 2.5x the sample.
- Polarization demoted: mean below both, tail-find did NOT replicate
  (its 40-pool 181 was luck). Length is the selector; polar_dist is
  a weak (-0.25) secondary signal at best.
- Informed selection of any kind is worth ~4.5 edges over random.
- **The shelf: 6 distinct bottoms score exactly 181; zero score 182+**
  across everything ever exact-scored. With three deepen-rigid 181
  assemblies from three lanes, the 154 -> 181 -> 185 -> 190 shelf
  structure is now statistically solid, not anecdotal.

Cheap-lane conclusion: stint-budget assembly generation is measured
out — selection tuning cannot cross the shelf. Remaining moves are
structural: symmetry taxonomy (#4 on the todo) or Wang-Landau oracle
(#5), i.e. James's hunt-vs-census allocation.

## Day 6 (2026-07-05 late): Wang-Landau oracle built + calibrated

James's allocation call: lead with Wang-Landau (memo #6). Built
(`wang_landau.py`): flat-histogram walk over all directed snakes,
grow/shrink both ends (ergodic, no constraint softening needed),
Hastings-corrected, anchored at N(0)=2^n, kill-safe checkpoints.

Trust chain, all committed:
- Exact enumerator == independent brute force at Q3.
- WL vs exact at Q4: <=0.35 nats everywhere, optimum found unaided.
- WL vs exact at Q5 (1.3M paths, 0.1s enumeration): worst 0.51,
  mid-range ~0.2, optimum found.
- Q6 exact = 30,001,185,088 directed paths (6x first-move symmetry,
  cross-validated vs full enumeration; saved to
  wl_q6_exact_counts.json). ENTROPY CLIFF measured: S falls 21.9 ->
  10.7 over the last 6 lengths; 25->26 alone is -4.6 nats (~100x).
- WL vs exact at Q6: mid-tail 0.2-0.35 nats; ~1 nat at the last
  reachable bins; NEVER reached L=26 (extendable 25-snakes are 1-in-
  100 — the blind chain converges f first). Oracle precision at the
  extreme tail is ~1 nat and coverage stops short of the optimum.

Q9 production: chunk 1 banked 1.77B moves (1.2M/s pure Python),
f=2.4e-4, best_seen=138 — converged over [0,138] only. Implication
from Q6: the raw chain will NOT reach the 180-190 region unaided.
**Next build for the oracle: windowed WL** (overlapping L-windows,
e.g. [120,160]/[150,190], stitched; standard deep-tail practice) —
possibly seeded with known snakes (start window walks FROM the 181/
185 certificates; legal since the chain is ergodic within a window).
That is the piece that makes the 182-190 tail measurable.

## Day 6: dimension-landscape assessment, corrected against the live table

An external "soft middle at d10-13" pitch was fact-checked against
minortriad (re-verified 2026-07-06): its record values were STALE
(370/712/1373/2687 vs real 376/737/1465/2898 — the exact trap our
2026-07-04 verification documented). Corrected verdicts:

- d10-13: contested, improved twice in a decade, held by Meta-NRPA-
  class methods. Density decay gentle (0.371 -> 0.354). d10 is NOT an
  easy pivot: doubling our 185 tops out ~372 < 376.
- SURVIVES: (a) d14-20 derived-from-coil bounds, never directly
  searched — near-guaranteed discounted record via 2x2898-k, pending
  verification of the deduced-bounds claim; (b) spread-k (k>=3)
  circuit-code grid, under-attacked, verify tables first; (c) upper-
  bound polarity — dimension-specific upper bounds stagnant ~30
  years, zero competition, SDP/Lasserre route.

All three filed as HANDOFF todo #10 (a/b/c) with verification steps
built in. The serious games remain: d9 hunt, census/upper bounds, and
the WL oracle arbitrating between them.

## Day 6: pivot move (polymer MC) — measured frozen at campaign densities

The untried-ideas memo's flagship (Madras-Sokal pivot, never ported to
SIB) prototyped same-hour (`pivot_prototype.py`): pivot at joint i =
apply sigma in S_n to the suffix in vertex form w -> v_i ^ sigma(w ^
v_i); only prefix<->suffix interactions need checking (word-view
factor legality is permutation-invariant within each side — the memo's
theorem checks out).

Measured acceptance (200 trials/joint, uniform S_9):
- ours-185 / wynn-190 / router-181: 0.13-0.14% overall; legal pivots
  at only 5-6 of ~185 joints, ALL at the extreme ends — except one:
  the 185's position 154, its topA<->topB construction seam.
- Q8 bottoms (90e, the diversity-engine target regime): 0.28-0.45%,
  3-5 of 89 joints. Same saturation (~0.36 density), same freeze.

Verdicts:
1. As a mutation/diversity engine: DEAD at campaign-relevant
   densities. Polymer intuition fails — SAW sampling is dilute; SIB
   record objects are constraint-saturated.
2. As an INSTRUMENT: real. The acceptance profile independently
   reproduces the rigidity theorems (mid-body frozen, ends loose) and
   LOCATES CONSTRUCTION SEAMS (the 185's only mid-body flexibility is
   exactly its assembly junction) — no prior SIB diagnostic does this.
3. Follow-up filed, not built: shrink-pivot-regrow composite (shrink
   to ~0.25 density where pivots breathe, pivot, exact-regrow) as a
   pool mutator — the only form in which the move class could still
   pay.
