# RUNBOOK — end-to-end operation for agents

Step-by-step procedure for running congruity's full cycle:
**compose → adversarially review → cheaply verify with code → record.**
Written for an AI agent or a human at the terminal. Policy background lives
in `AGENTS.md`; this file is the *how*.

Standing invariants (violating any of these poisons the catalog):

1. **Prioritize, don't veto.** Never select or filter candidates by L3
   verdict or `purely_testable`. Rank by `verify_priority` / confidence.
2. **Never generate the measuring stick.** Benches, baselines, SOTA bars,
   and record values are human-written or forge-checked, never freehand.
3. **Never fabricate record values.** `records.json` entries come from
   cited published sources only.
4. **The holdout is spent once.** Never evaluate a holdout split during
   search; never re-run a confirm to get a better number.
5. **Never edit `data/pipeline/outcomes.jsonl` or `findings/INDEX.md`
   retroactively** — append only.

---

## Phase 0 — Orient and preflight

```bash
uv run pytest -q && uv run ruff check .        # repo healthy?
uv run python -m pipeline pack doctor --no-smoke   # packs healthy?
```

Read, in order: `findings/INDEX.md` (track record), `OVERVIEW.md` (architecture), `docs/roadmap-discovery-2026-06.md` (open threads).

Environment (`.env`, template in `.env.example`):

| Var | Effect |
|---|---|
| `OPENALEX_MAILTO` | polite-pool rate limits (recommended) |
| `OPENAI_API_KEY` | embedding relevance gate (else LLM-judge/codex fallback) |
| `ANTHROPIC_API_KEY` | anthropic relevance judge (optional) |
| `CONGRUITY_MAX_CONCURRENCY` | parallel codex calls, default 4; set 1 for serial |
| `CONGRUITY_EFFORT_<STAGE>` | per-stage reasoning effort, e.g. `CONGRUITY_EFFORT_VERIFY_L1=medium` |

The `codex` CLI must be authenticated (it is the primary LLM transport);
`codex login status` to check.

---

## Phase 1 — Choose the corpus mode (decision point)

- **Topic-sampled** (default): the problem maps to one or more mainstream
  OpenAlex topics ("audio source separation", "cache replacement").
- **Paper-direct** (`--paper-search`): the problem is narrower than any
  topic (single algorithm, sub-topic). Symptom if you guess wrong: the run
  aborts at the Step 1.5 relevance gate with most papers dropped
  (merge sort: 0/71; graph coloring: 5/90 via topics).

Topic-sampled needs a harvested list first:

```bash
uv run python -m pipeline.topics harvest \
  --queries "term1,term2,...,term20" \
  --papers-per-query 100 --year-min 2010 --min-works 50 \
  --save data/pipeline/topic_lists/<name>.json
```

Use `--year-min 2010` for mature fields (2018+ starves the corpus).
Paper-direct skips harvesting entirely — anchor every query on the
problem term: `--paper-search "graph coloring local search;tabu search graph coloring"`.

---

## Phase 2 — Compose

```bash
uv run python -m pipeline compose \
  --topic-list data/pipeline/topic_lists/<name>.json \   # OR --paper-search "q1;q2"
  --problem "$(cat /tmp/problem.txt)" \
  --max-papers 200 --year-min 2010 --target 6 \
  --relevance-mode auto \
  --with-recipes \
  --computable-pack <pack_id> \        # only if a pack covers the domain
  > /tmp/compose_<name>.log 2>&1 &
```

Runs 5–40 min. Monitor with `tail -f /tmp/compose_<name>.log`. Output:
digest at `findings/compose-<run_id>.md`, full state at
`data/pipeline/compositions/<run_id>/`.

**Failure handling:**

| Symptom | Action |
|---|---|
| `aborted_corpus_gate`, most papers dropped | wrong corpus mode → switch to `--paper-search`, or widen `--year-min` |
| 0 compositions produced | correct refusal behavior, not a bug — corpus too shallow; broaden queries |
| crash / kill mid-run (any stage) | **do not start a fresh run** — `uv run python -m pipeline compose --resume <run_id> --problem "..." [same flags]`; finished stages (corpus, dimensions, scores, compositions, per-composition verification) are reused from disk |
| codex transient failure | one retry is automatic; a persistent failure fails the stage — resume after fixing auth |

**Cost checkpoint:** after completion read
`data/pipeline/compositions/<run_id>/llm_usage.json` (also printed at end
of the log): calls, codex-minutes, estimated tokens per stage. If
verification dominates, consider `CONGRUITY_EFFORT_VERIFY_L1=medium` and
`CONGRUITY_EFFORT_VERIFY_L4=medium` for future runs.

---

## Phase 3 — Select candidates

Open `findings/compose-<run_id>.md`. Candidates flagged
`verify_recommended` (top-K by priority score) carry verification recipes.

- Rank by priority score / LLM confidence. **Ignore L3/L4 verdicts as
  filters** — use L3's specific objections as experiments to run, never as
  a veto (the project's best result was L3=INVALID).
- Six high-confidence candidates from one run is suspicious (vague problem
  statement), not good news.

---

## Phase 4 — Verify (decision point)

**Path A — domain has a candidate-ready pack** (check: `pipeline pack
doctor <pack_id>` shows `candidate_ready` / `search_ready`; currently
`graph_coloring` and `small_points_optimization`):

```bash
# One-shot: generate one candidate, run the trusted bench, record a label
uv run python -m pipeline autoverify --run <run_id> --cid C2 --pack <pack_id>

# Or bounded evolutionary search (train split), single holdout confirm
uv run python -m pipeline autosearch --run <run_id> --cid C2 --pack <pack_id> \
  --population 4 --generations 3 --elites 2
```

Both append a graded row to `data/pipeline/outcomes.jsonl`
(`provenance=auto`, confidence capped at `medium`). A killed autosearch
resumes automatically from `experiments/auto/<run>-<cid>/search/evals.json`
— just re-run the same command. Delete the search dir only for a
deliberate fresh start.

The full compose→search chain for one pack is also wrapped:
`uv run python scripts/flywheel.py --pack <pack_id> --problem "..." --paper-search "q1;q2"`.

**Path B — no pack; verify by hand.** Build
`experiments/<domain>/bench.py` on the shared harness:

```python
from experiments._harness import cap_threads
cap_threads()                     # BEFORE importing numpy — non-negotiable
import numpy as np                # noqa: E402
from experiments._harness import budget_loop, emit_result_contract, make_rng  # noqa: E402
```

Test against the recipe's named baselines and home-field SOTA. Emit the
`RESULT_CONTRACT` (`--json` mode) if the domain might ever join the
flywheel.

**Path C — new domain worth automating.** Two options:

```bash
# Hand-authored pack from templates (you supply datasets/baselines/SOTA):
uv run python -m pipeline pack new <pack_id>

# Evidence-first automation (codex drafts, machine checks):
uv run python -m pipeline pack forge <pack_id> --domain "<description>"
```

Forge stages: draft (cited) → pin (git SHA) → **reproduce** (pinned
implementation must match its published number within tolerance) →
battery (validator attack cases) → records (source snapshots) → evidence.
Output stays in `data/pipeline/forge/<pack_id>/` and cannot touch the
ledger. Review `evidence.md`, then — human signature required:

```bash
uv run python -m pipeline pack promote <pack_id> --approve "<your name>"
```

Promotion runs a final doctor smoke gate and rolls back on failure. If a
stage failed, fix and re-run `pack forge` (passed stages are skipped;
`--redo <stage>` to force). Never `--waive` reproduce without writing down
why in the findings doc. After promotion, hand-transcribe any record
values from `data/pipeline/forge/<pack_id>/records/` snapshots into
`records.json` with sources.

---

## Phase 5 — Record results

For every verified composition, regardless of outcome:

1. Write `findings/<domain>-verification-<date>.md`: which L3 attacks were
   falsified, which confirmed, a plain verdict (positive / partial /
   negative). Clean negatives are the engine working.
2. Append the row to `findings/INDEX.md` — the gold-standard track record.
3. If a bench emitted `record_claims`, `apply_record_claims` already
   compared them to the pack's record table. A `new-record` grade must be
   hand-checked against the published source before celebrating;
   `exceeds-proven-optimal` means your objective is miswired — treat as a
   bug, and the outcome is auto-quarantined.

---

## Phase 6 — Periodic reporting

```bash
uv run python -m pipeline web                      # regenerate findings/web/
uv run python scripts/funnel_report.py             # stage pass-rates + codex-min per valid label
uv run python scripts/outcome_calibration.py       # signal-vs-outcome calibration
uv run python scripts/ledger_calibration.py        # auto-ledger calibration (informative at n>=10)
```

Nightly automation exists but is opt-in: `docs/flywheel-scheduling.md`
(`scripts/flywheel_nightly.sh` + launchd/cron template). Nothing
self-installs.

---

## Quick failure index

| Situation | Fix |
|---|---|
| compose crashed anywhere | `compose --resume <run_id>` with the same flags |
| autosearch killed | re-run the same command (evals.json cache) |
| bench hangs ~forever on one eval | thread oversubscription — bench must call `cap_threads()` before numpy |
| bench timeout leaves CPU burning | run through `execute_pack_bench`/doctor (process-group kill), not bare shell |
| OpenAlex 429s | automatic retry w/ Retry-After; set `OPENALEX_MAILTO` |
| pack behaves oddly | `pipeline pack doctor <id>` (add `--no-smoke` for structure-only) |
| forge reproduce fails | the cited SOTA didn't reproduce — that's the system working; find a better citation, don't waive |
| two flywheel runs at once | safe: ledger appends are flock-protected |
