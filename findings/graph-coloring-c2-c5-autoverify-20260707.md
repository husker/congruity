# Graph coloring C2–C5 autoverify — Windows portability fixed; all candidates time out

**Date:** 2026-07-07
**Run:** `20260611-022616` (graph_coloring, compositions C1–C5)
**Pack:** `computable_packs/graph_coloring` (candidate command `--seeds 3`,
home-field SOTA = descending-k TabuCol seeded from DSATUR)
**Verdict:** **the pipeline now runs end-to-end on Windows; C2–C5 all
grade `invalid` (timeout) — zero new valid labels.**

## Context

Only C1 had ever been run through the autoverify one-shot (it tied
TabuCol; `findings/graph-coloring-domain-20260611.md`). C2 had an
autosearch partial; C3–C5 were never run. The task was to run C2–C5.
The autoverify path (candidate codegen via codex → pack bench → grade →
ledger) had never executed on this Windows box.

## Windows portability — four blockers fixed (the real deliverable)

The autoverify + pack path was full of POSIX-only assumptions that
crashed or mis-ran on Windows. All fixed:

1. **`import fcntl`** (top-level in `pipeline/autoverify.py`) crashed
   test collection and any import → portable `_exclusive_lock`
   (fcntl / msvcrt.locking / no-op fallback). *(committed 675412d)*
2. **`shlex.split(command)`** ran in POSIX mode and mangled `C:\` paths
   as escapes → on Windows hand the string to CreateProcess. Same bug in
   `pipeline/pack_tools.py`. *(675412d)*
3. **`os.killpg`/`SIGKILL`/`start_new_session`** (timeout path) are
   POSIX-only → Windows uses `CREATE_NEW_PROCESS_GROUP` + `taskkill /F
   /T`. *(675412d)*
4. **codex on Windows** (`pipeline/llm_io.py`): `subprocess.run(["codex",
   …])` does not resolve `codex.CMD` → `shutil.which("codex")`; and
   codex (Rust) tears down its MCP worker with `taskkill`, whose SUCCESS
   line **leaks onto codex's own stdout** and corrupted every generated
   candidate (line-1 SyntaxError) → strip it in `_invoke_codex`.
   *(uncommitted — llm_io.py holds unrelated in-flight edits; see below)*

Also added `CONGRUITY_BENCH_TIMEOUT_S` (env override for the pack-bench
wall-clock cap; default 900s) so a non-terminating candidate resolves
faster. *(81ca5cb)*

Effect: test suite 442→450 pass (the 4 collection errors + the
autoverify/autosearch/computable/harness failures are gone; 8 remaining
failures are pre-existing test-fixture POSIX assumptions). codex auth
had also lapsed (401) and was re-established by the user.

## Outcome — C2–C5 all time out

With every blocker cleared and codex authenticated, the four
compositions generated real candidates and ran against the bench:

| cid | grade | reason | valid label? |
|---|---|---|---|
| C2 | invalid | timeout (even at the full 900s) | no |
| C3 | invalid | timeout (420s cap) | no |
| C4 | invalid | timeout (420s cap) | no |
| C5 | invalid | timeout (420s cap) | no |

The generated candidates are consistently **over-engineered hybrids** —
e.g. C2's module (validated, ~470 lines) layers qudit-inspired
simulated annealing over exact backtracking (`_bounded_k_color` with
budgets up to 60000 nodes) descending k across all instance families ×
3 seeds. That does not respect the pack's practical budget (TabuCol
finishes the same instances in seconds) and blows the wall-clock cap.
C2 timing out even at 900s confirms genuine non-viability, not a
too-short cap.

**Result: zero new valid labels.** All four are quarantined
(`provenance=auto`, `confidence=quarantined`, `valid=false`) and do not
advance the calibration ledger. The earlier `invalid: SyntaxError` rows
in the ledger for C2/C3 are artifacts of blockers #4 above (codex not
found → empty candidate; taskkill-stdout leak → line-1 syntax error),
now fixed and superseded by the timeout rows.

## Reading

The flywheel behaved correctly — it ran, validated, and quarantined
non-viable candidates without inventing labels. But the autoverify
**one-shot codegen** for this pack has a systematic failure mode:
codex produces maximal, exact-search-heavy candidates that ignore the
"equal budget, TabuCol is the bar, keep it cheap" gotchas the pack
states, so they time out rather than compete. This mirrors the
autoverify-flywheel verdict (`findings/autoverify-flywheel-
verdict-20260530.md`): cheap one-shot codegen rarely yields a viable
home-field-SOTA-beating candidate. The productive path for graph
coloring remains **autosearch** (the C2 partial came from the
evolutionary search arm), not the one-shot.

The durable win here is the **Windows portability** of the entire
autoverify/pack path — every future flywheel run on this box was
blocked before and works now.

## Autosearch rerun (C3) — same TabuCol-parity ceiling, plus a runaway confirm

Reran C3 through `pipeline autosearch` (population 4, generations 3,
elites 2; search-fitness cap `CONGRUITY_BENCH_TIMEOUT_S=420`) — the
evolutionary path that produced C2's only valid label. Its selection
does weed out the timeout-prone candidates (they score `fitness=FAILED`
and are dropped), so this is a fairer test than the one-shot. Full
search trace:

```
gen 0: g0_c0 -0.125  g0_c1 -0.125  g0_c2  0.0   g0_c3 -0.375
gen 1: g1_e0  0.0    g1_e1 -0.125
gen 2: g2_e0  0.0    g2_e1 FAILED
gen 3: g3_e0 FAILED  g3_e1 FAILED
```

(fitness = mean colors_used delta vs TabuCol at equal budget; 0.0 = tie,
negative = worse.) **Best achievable = 0.0 = TabuCol parity** — no
candidate beats the bar, and the refinements that tried to do more
timed out and were selected out. This is the identical ceiling C1 hit
("added nothing over DSATUR-seeded TabuCol") and that the one-shot hit.

**A second, practical finding:** autosearch's holdout confirm uses a
hardcoded **3-hour backstop** (`autosearch.py` `timeout_s=10800`; the env
cap governs only the search fitness evals). The best parity candidate
over-engineers the heavy holdout instances (gnp125, planted80, queen10
at 100k iters) with exact backtracking, so the confirm ran away — ~20
CPU-minutes and climbing when I killed it (it would have run toward the
3-hour cap). So even autosearch's fitness-viable parity candidates don't
respect the practical budget on large instances.

**Verdict across all methods:** graph coloring does **not** beat TabuCol
here — C1 (one-shot, tied), C2 (autosearch, tied → the lone valid
"partial"), C2–C5 (one-shot, all timeout), and C3 (autosearch, parity).
It is a robust **TabuCol-parity ceiling**, consistent with the gotcha
the pack itself states ("most published improvements lose to plain
TabuCol at equal budget"). Burning ~40 min of search + a multi-hour
confirm each on C4/C5 would only re-demonstrate the same parity; not
run.

## Follow-ups

- **Commit the codex fixes** in `pipeline/llm_io.py` (currently
  uncommitted alongside unrelated in-flight edits): the
  `shutil.which("codex")` resolution and the taskkill-stdout strip.
- Autosearch was tried (above) and also saturates at parity; the
  practical blocker for running it at scale here is the hardcoded
  3-hour confirm backstop colliding with over-engineered candidates —
  make `autosearch.py`'s confirm `timeout_s` env-configurable (like
  `execute_pack_bench` now is) so a runaway confirm resolves to invalid
  instead of burning hours.
- A per-candidate iteration/step guard (so an over-engineered candidate
  self-limits instead of timing out) would let both paths grade the
  candidate's quality rather than its wall-clock.
- The 8 remaining `test_pack_forge`/`test_pack_tools` failures are
  pre-existing test-fixture POSIX assumptions (hardcoded `python3`,
  single-quoted shell) — a separate, test-only cleanup.
