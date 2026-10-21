# pipeline — the congruity engine

The package has two generations of machinery:

- **Current: the compose + verify flywheel.** `compose.py` (corpus →
  dimensions → scoring → compose → L1-L4 verification), `autoverify.py`
  (one-shot pack-bench verify), `autosearch.py` (bounded evolutionary
  search + single holdout confirm), `computable.py` (computable-pack
  loader/preflight/records), `calibration.py`, `recipe.py`, `novelty.py`,
  `web_ui.py` (static dashboard via `python -m pipeline web`). The current
  workflow, flags, and selection policy are documented in the **root
  `README.md`** and `AGENTS.md` — start there.
- **Legacy: the v1 batch bridge-discovery flow** (`discover.py`,
  `pair_filter.py`, `rank.py`, `digest.py` via `python -m pipeline run`),
  documented below for reference.

## What's here

```
pipeline/
├── __init__.py
├── _openalex.py        shared HTTP plumbing + abstract reconstruction
├── env.py              .env loading (idempotent, never overrides shell)
├── topics.py           OpenAlex topic harvest
├── corpus.py           corpus assembly (cached per-topic / paper-direct)
├── corpus_gate.py      Step 1.5 quality gate (relevance/encoding/structure)
├── relevance.py        relevance judge backends (anthropic/openai/codex)
├── compose.py          compose + L1-L4 verification orchestrator
├── llm_io.py           codex subprocess layer + artifact persistence
├── schemas.py          pydantic schemas
├── recipe.py           per-composition verification recipes
├── autoverify.py       one-shot pack-bench verify (candidate codegen only)
├── autosearch.py       evolutionary candidate search + holdout confirm
├── computable.py       computable-pack loader / preflight / record tables
├── calibration.py      outcome-calibration machinery
├── novelty.py          post-L3 novelty check (experimental)
├── domain_pack.py      compose-time hint packs
├── identity.py         model/family identity helpers
├── web_ui.py           static findings dashboard
├── pair_filter.py      [legacy v1] cross-citation density across pairs
├── discover.py         [legacy v1] Codex bridge discovery + citation check
├── rank.py             [legacy v1] weighted score per bridge
├── digest.py           markdown digest renderer
├── runner.py           orchestrator + CLI (all subcommands)
├── __main__.py         entry: `python -m pipeline`
└── README.md           this file
```

---

# Legacy: v1 batch discovery

End-to-end batch that takes ~100 OpenAlex CS topics, filters by cross-citation density, runs Codex bridge-discovery on the most-separated pairs, and produces a ranked markdown digest.

## Run it

Always do a `--dry-run` first to validate the pipeline shape without spending on Codex:

```bash
uv run python -m pipeline run --top-n-topics 20 --max-pairs 5 --dry-run
```

For a real batch:

```bash
uv run python -m pipeline run --top-n-topics 50 --max-pairs 30
```

Step-at-a-time:

```bash
uv run python -m pipeline topics                  # fetch + cache CS topics
uv run python -m pipeline filter --sample-size 50 # compute density matrix
uv run python -m pipeline run --refilter          # full batch, recomputing layer 2
```

## Outputs

- `data/pipeline/topics_cs.json` — Layer 1 cache
- `data/pipeline/topic_samples/{topic_id}_*.json` — Layer 2 paper samples per topic
- `data/pipeline/density_matrix.json` — Layer 2 result
- `data/pipeline/runs/{pair_id}/` — per-pair corpus, prompt, codex response, citation check
- `data/pipeline/runs.jsonl` — full run log (one line per batch run)
- `findings/digest-{date}.md` — Layer 4 digest

## Budget

Per-batch cost dominated by Codex calls. With `--max-pairs 30` and high reasoning effort, expect ~$10–25.

## Scoring (Layer 4a)

| Signal | Weight | Source |
|---|---|---|
| Codex confidence (1-10 self-rating) | 0.30 | parsed from response |
| Citation grounding | 0.25 | citation_check pass/fail |
| Cross-cite density (inverse) | 0.20 | density matrix |
| Mechanism specificity | 0.15 | keyword count in response |
| Corpus completeness | 0.10 | papers per side |

Weights are starting guesses. Tune after the first batch shows what correlates with "actually interesting."

## What's NOT in v1

- Multi-model verification (Claude + Codex + Gemini cross-check) — landed
  in the compose pipeline (generation vs verification split across families)
- Daily cron / digest emails — still unbuilt; the flywheel is run manually
- Cross-domain (math, bio, etc.) topic pairs — landed in the compose
  pipeline (7+ domains verified)
- Web UI — shipped after all: `python -m pipeline web` writes a static
  dashboard to `findings/web/`
- Database — JSON files are sufficient at this scale
- Per-domain oracles — that's `oracle/`, separate package, manual

## Caches

All intermediate artifacts are cached. Re-running with same flags re-uses caches. Force regeneration:

- `topics --force` — re-fetch CS topics
- `run --refilter` — recompute density matrix
- Delete `data/pipeline/topic_samples/{topic_id}_*.json` for a single topic to re-sample it
