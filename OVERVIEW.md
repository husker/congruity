# Congruity — Overview

_Core architecture write-up as of 2026-05-22, with a status update below.
Supersedes the pre-pivot CS-equivalence framing. For current goals and
policy see `docs/PUBLIC_SNAPSHOT.md` and `docs/roadmap-discovery-2026-06.md`; the
track record lives in `findings/INDEX.md`._

## Status update — 2026-07-01

Everything below about the compose pipeline and L1-L4 stack still holds
mechanically, but three later shifts recalibrated how the outputs are used:

1. **Success bar: home-field SOTA.** (2026-05-30, `docs/PUBLIC_SNAPSHOT.md`) A win is a
   technique from a *distant* field that beats the destination field's own
   best-known method — ideally a certifiable record (Tier B). Two clean
   transfer wins so far: Heilbronn C1 and short-string C2
   (`findings/transfer-catalog-20260530.md`).
2. **Prioritize, don't veto.** (2026-05-29,
   `findings/outcome-calibration-2026-05-29.md`) L3 verdicts and the
   `purely_testable` flag have no outcome-predictive power and are no
   longer used as filters — the old gate wrongly rejected 84% of
   compositions, including short-string C2. Candidates are ranked by
   priority score / LLM confidence. Standing rule: no new gates.
3. **The pack-bench flywheel.** (2026-06,
   `docs/roadmap-discovery-2026-06.md`) Human-written computable packs
   (`computable_packs/`) supply datasets, baselines, SOTA bars, and record
   tables; `pipeline autoverify` (one-shot) and `pipeline autosearch`
   (bounded evolutionary search + single holdout confirm) generate only the
   candidate solver. First valid autonomous labels landed on graph coloring
   (2026-06-11). Auto-labels append to `data/pipeline/outcomes.jsonl`.

## What it is

A **verification-first compositional synthesis engine** that takes a problem
statement and a corpus of recent scientific papers, identifies candidate
research directions composed of 2–5 papers whose combined contributions
solve the problem better than any single paper, and stress-tests each
candidate through a four-level verification stack.

The lineage is Swanson's 1986 literature-based discovery (LBD), where
the cure for one disease was hiding in the union of two unconnected
literatures. The pivot from generation-heavy tools (Iris.ai, Elicit,
Connected Papers) is the emphasis on **verification**: every surfaced
candidate carries a traceable evidence chain, an adversarial falsification
attempt, and a list of testable predictions classified as fundamental
vs empirical.

The engine is **domain-agnostic** — exercised on 7 domains across CS,
biomedicine, materials science, and physics. The interesting findings
emerge precisely at cross-community boundaries where domain experts
on each side don't read each other's reviews.

## What it does (in one example)

Given the problem `"identify critical engineering blockers for
commercializing nuclear fusion energy"` and a 150-paper corpus across
15+ disciplines, the engine produced (among other compositions) a
candidate combining:

- A 2024 paper on Rydberg-atom microwave receivers (96–170 GHz, µV/m/√Hz
  sensitivity)
- A 2024 paper on high-power microwave fault diagnostics
- A bridge mechanism: directional-coupler attenuation to reconcile the
  10⁶ V/m gyrotron field with the Rydberg sensor's ~230 V/m ceiling

The composition survived L1 (internal consistency), L2 (every claim
traceable to a verbatim abstract quote), L3 (adversarial reviewer
challenge), and L4 (8 empirical objections / 0 fundamental). An external
four-agent deep-research pass then independently confirmed: no published
deployment, no formal proposal, no patent, no review-paper mention. The
gap is real; the engineering recipe is known. **First externally-verified
novel composition** the engine has produced.

## How it works

```
INPUTS                Step 0
  ├─ problem statement (free-form)
  ├─ topic list (harvested from OpenAlex)
  └─ post-cutoff year (default 2024)

CORPUS                Step 1
  └─ assemble_corpus():  sample papers/topic, dedupe, tag source

QUALITY GATE          Step 1.5  [NEW, 2026-05-19]
  ├─ check_relevance      (OpenAI text-embedding-3-small)
  ├─ check_encoding       (mojibake, HTML, control chars)
  └─ check_structure      (W-id regex, year, title/abstract length)
                          → soft-drop on individual fail
                          → abort if fail_rate > threshold

DIMENSION EXTRACTION  Step 2
  └─ Codex/GPT identifies 5–8 orthogonal problem dimensions in the corpus

SCORING               Step 3
  └─ Codex/GPT scores each paper 0–5 on each dimension + extracts
     one-line contributions

COMPOSITION           Step 4
  ├─ cluster_papers_by_dimension()  ← programmatic, not LLM
  │     bounds the search from C(n,5) to compositions-across-clusters
  └─ Codex/GPT produces ≤5 candidate compositions, each
     pulled from DIFFERENT clusters, each with verbatim quotes

VERIFICATION          Step 5
  ├─ L1 internal consistency      (Codex)
  ├─ L2 source grounding          (programmatic substring + token overlap)
  ├─ L3 adversarial falsification (Codex with falsification prompt)
  └─ L4 testability classification (Codex: FUNDAMENTAL vs EMPIRICAL
                                   per L3 objection)

OUTPUTS
  ├─ data/pipeline/compositions/<timestamp>/state.json   (full structured run)
  ├─ data/pipeline/compositions/<timestamp>/corpus_gate.json
  └─ findings/compose-<timestamp>.md                     (human digest)
```

## Key design decisions

1. **Post-cutoff papers (year ≥ 2024).** Limits the risk that L3
   adversarial verification just recalls the composition from the
   verifier's training data. Bar creeps forward as model cutoffs do.

2. **Three model families for independence.** Generation runs in Claude;
   verification (L1/L3/L4) runs in Codex/GPT; manual novelty checks use a
   third family (Gemini-class deep-research). The 95.9% empirical / 9
   fundamental aggregate across 26 compositions is a cross-family signal,
   not a self-graded one.

3. **Cluster-then-compose.** At 100+ paper corpora, C(n,5) is 75M+
   subsets. We cluster papers programmatically by primary dimension
   (highest score, ≥3) and ask the LLM only to compose across clusters.
   This is what made the engine work at scale.

4. **L4 testability classifier.** "L3 WEAK" doesn't mean "wrong" — it
   often means "the composition makes specific predictions that haven't
   been tested yet." L4 splits each L3 objection into FUNDAMENTAL vs
   EMPIRICAL (this is exactly the kind of test that would settle it).
   Across 26 compositions, 95.9% of objections were empirical, which is
   exactly what an unsolved-problem tool should produce.
   _Definition updated 2026-05-30 (HANDOFF Fix 2): FUNDAMENTAL now means
   "precludes beating baselines on realistic data," not "no experiment can
   rescue this." Neither L4 flavor is used as a filter — see the status
   update above._

5. **Verification before novelty.** The stack runs to completion on every
   candidate before any judgment about whether the finding is new. Cheap
   filters first; the expensive novelty check is only worth running on
   compositions that survive L1–L4.

## Validation so far

### Domains exercised (chronological)

| Domain | Compositions | L3 mix | Notable |
|--------|:---:|--------|--------|
| Matrix multiplication | 3 | mixed | broad problem |
| HTS (broad mechanism) | 4 | 2 WEAK / 2 INVALID | same-community |
| Hydride SC (narrow) | 5 | 5 WEAK | hardest problem, all testable |
| Carbon capture | 4 | mixed | broad-statement drift |
| Colchicine repurposing | 5 | 4 WEAK / 1 INVALID | drug-repurpose shape |
| Nuclear fusion | 5 | 5 WEAK | **C5 externally-verified novel** |
| TSP (graph-based) | 5 | 2 WEAK / 3 INVALID | saturation visible |

**Aggregate:** 26 compositions, 217 objections, 95.9% empirical, 9
fundamental. One externally-confirmed novel finding (Rydberg-fusion).

### Cross-run patterns

Three distinct INVALID failure modes are now diagnosable from the
output, not just blanket rejections:

1. **Saturated-field rediscovery.** TSP produced 3/5 INVALIDs because the
   field is mature enough that compositions the LLM finds are usually
   published. Same risk in classical mechanics, organic chemistry, CV.
2. **Broad-statement sidestep.** Carbon-capture compositions drifted into
   adjacent problems because "capture CO₂" admits too many valid
   interpretations.
3. **Same-community no-surprise.** Most HTS / hydride compositions stayed
   within one community's vocabulary — no cross-citation gap to bridge.

### The Step 1.5 corpus quality gate (added 2026-05-19)

Catches three pre-LLM corruption modes:
- Off-topic papers (embedding cosine similarity < threshold)
- Encoding corruption (mojibake, undecoded HTML entities, JATS tags
  leaking from XML)
- Malformed structure (bad W-id, missing year, short abstract)

Empirical validation against the TSP corpus: 7/150 papers flagged for
real encoding issues (HTML entities like `&gt;`, JATS tags like
`<sup xmlns:mml=...>` — these had been silently feeding into Codex
prompts). Below the 30% abort threshold; soft-dropped and continued.

## What works vs what doesn't yet

### Works
- **Mechanically across all 7 domains.** The pipeline doesn't break on
  any field; corpus assembly through L4 completes in 20–40 min per run.
- **L1 + L2 verification.** Programmatic, deterministic, catches real
  inconsistencies and ungrounded quotes.
- **L3 adversarial.** Produces genuinely useful pushback; the verdicts
  correlate with field saturation in interpretable ways.
- **L4 testability split.** Turns "weak" verdicts into actionable lists
  of empirical predictions with method / tooling / effort estimates.
- **Cluster-then-compose.** Scales to 150+ paper corpora without
  prompt-size blowup.

### Doesn't yet
- **No expert hit-rate study.** The framework's load-bearing unknown.
  We don't know what fraction of surfaced compositions a domain expert
  would call "interesting and I hadn't thought of this."
- **No automated novelty check.** Rydberg-fusion was checked manually
  with a four-agent deep-research pass. `pipeline/novelty.py` exists
  behind `--with-novelty` but its S2/OpenAlex search primitives are
  still stubs — treat it as experimental.
- **No saturation index or cross-community distance metric in the
  pipeline.** (`scripts/transfer_distance.py` measures transfer distance
  after the fact; nothing predicts it pre-compose.)
- **Domain bits partially extracted.** `domain_packs/` (compose-time
  hints) and `computable_packs/` (executable verification) exist, but
  prompts and thresholds are still tuned in code, not config.
- **No cost target per composition.** Codex burn at 150-paper scale is
  real but unmeasured — no token/cost tracking exists yet. First funnel
  reading: ~137 codex-minutes per valid label (roadmap #6).

## Avenues to explore

### A. Framework-level (highest payoff)

1. **Define falsifiable success.** Recruit 3 domain experts. Show each
   30 random compositions, blind. <5% "interesting" = engine produces
   smooth noise; >15% = framework works. Single biggest unknown.

2. **Pre-flight diagnostic layer.** Pure compute on the corpus:
   - Saturation index (publication rate × citation density ×
     review-paper count, trailing 5y)
   - Cross-community distance (cluster-pair cross-citation rate) — this
     is the metric that *predicts* Rydberg-fusion-shape wins
   - Problem-statement specificity score (LLM rubric, gate broad
     statements)

3. **Automated post-L3 novelty check.** Run the four-agent web search as
   a built-in stage for any L3 PASS/WEAK survivor.

4. **Engine + domain pack.** Pull domain-specific bits (topic-harvest
   queries, problem-statement templates, adjacency hints) into a config
   layer so the engine is a clean reusable core.

5. **Generalized adapter-overfit guard.** Every composition has a
   bridging mechanism with some complexity tax. Flag compositions where
   the bridge is more complex than either source cluster — likely
   over-engineered.

6. **Hit-rate feedback loop.** Track per-domain, per-dimension,
   per-composition-shape conversion across runs. Feed forward into next
   run's prompt tuning.

### B. Single-finding follow-through

- **Rydberg-fusion (C5 from fusion run).** Concept paper to *Review of
  Scientific Instruments* (Notes) or HTPD biennial proceedings. SBIR
  proposal with Rydberg Technologies as small-business lead and PPPL or
  IPP Greifswald (W7-X) as research partner. Engineering deliverable:
  ~120 dB directional-coupler tap + vapor-cell pickup; SI-traceable
  ECRH stray-field detector.
- **Other compositions.** Most other L3 WEAK / 0-fundamental
  compositions still need their independent novelty check before any
  follow-through; the manual four-agent process is too expensive to run
  on each one until step A.3 lands.

### C. Productize (de-prioritized vs framework work)

- Pharma rare-disease drug repurposing
- Tech-transfer offices and IP scouts at R1 universities
- Verification-stack-as-service for other research-AI tools (Elicit,
  Consensus, SciSpace) as their "is this actually novel?" layer

### D. Engineering / cost

- Per-composition cost target (<$5 inference)
- Result browsing UI (web) over CLI + markdown digests
- Cheaper L1/L4 via smaller models (current pipeline runs everything at
  Codex `model_reasoning_effort=high`)
- Replace OpenAlex direct HTTP with batched, cached, and resumable
  fetcher for very large corpora

## Open questions

1. **Hit rate.** Unknown until A.1 is run.
2. **Pre-cutoff drift.** As model cutoffs creep past 2024, the
   training-contamination floor rises. Probably need to push the cutoff
   forward annually.
3. **Ground-truth latency.** Real validation of a surfaced composition
   takes 2–5 years (publication, follow-up, attempted replication). The
   framework's evaluation has to use proxies (expert hit rate, novelty
   check, internal verification consistency).
4. **Verification-stack moat.** Composition is what an LLM does well;
   *filtering* compositions is where the value is. Defensibility likely
   tracks the verification stack, not the engine.
5. **Domain shape limits.** The engine works mechanically across fields,
   but output quality depends on field maturity, problem-statement
   specificity, and community-overlap structure. Some fields may
   structurally not have Swanson-shape gaps.

## Repository layout

```
congruity/
├── pipeline/                          # the engine
│   ├── _openalex.py                   # HTTP plumbing
│   ├── corpus.py                      # sample_topic, cached
│   ├── corpus_gate.py                 # Step 1.5 quality gate (NEW)
│   ├── topics.py                      # topic harvest (subcommand)
│   ├── pair_filter.py                 # cross-citation density (legacy)
│   ├── discover.py                    # single-pair discovery (legacy)
│   ├── compose.py                     # current primary entry point
│   ├── autoverify.py                  # one-shot pack-bench verify
│   ├── autosearch.py                  # evolutionary search + holdout confirm
│   ├── computable.py                  # pack loader / preflight / records
│   ├── rank.py / digest.py            # post-processing
│   └── runner.py                      # CLI dispatch
├── oracle/                            # differential-test framework
│   ├── policies.py                    # LRU, ARC, paged-eviction
│   └── workload.py                    # synthetic LLM-serving workload
├── computable_packs/                  # executable verification packs (flywheel)
├── domain_packs/                      # compose-time hint packs
├── experiments/                       # per-composition verification harnesses
├── scripts/                           # step0_*.py through step4_*.py + reports
├── findings/                          # digests + write-ups; INDEX.md = track record
├── ground_truth/                      # YAML candidate lists (legacy)
├── data/pipeline/
│   ├── topic_lists/                   # harvested topic lists
│   ├── topic_samples/                 # per-topic paper cache
│   ├── compositions/<timestamp>/      # per-run state
│   ├── outcomes.jsonl                 # auto-label ledger (autoverify/autosearch)
│   └── research/                      # manual novelty-check reports
├── tests/                             # pytest suite (279 tests)
└── plans/                             # design / planning artifacts
```

## Useful commands

```bash
# Harvest topics for a new domain
uv run python -m pipeline.topics harvest \
  --queries "fusion plasma" "ECRH gyrotron" "tokamak diagnostic" \
  --papers-per-query 50 --year-min 2024 \
  --out data/pipeline/topic_lists/fusion.json

# Run a compose (top-level `pipeline compose` and module `pipeline.compose`
# are interchangeable; both expose the same flag surface).
uv run python -m pipeline compose \
  --topic-list data/pipeline/topic_lists/fusion.json \
  --problem "identify critical engineering blockers for commercializing nuclear fusion" \
  --max-papers 150 --target 5 \
  --relevance-mode auto --with-recipes

# Dry-run (no LLM calls, just corpus assembly preview)
uv run python -m pipeline compose --topic-list ... --problem ... --dry-run

# Step 1.5 gate tuning (and backend selection)
uv run python -m pipeline compose ... \
  --relevance-mode anthropic \
  --relevance-threshold 0.35 --max-fail-rate 0.20

# Bypass the relevance gate
uv run python -m pipeline compose ... --skip-gate

# Debug-only: continue past Stage 3 even with degraded scoring coverage
uv run python -m pipeline compose ... --no-strict-scoring

# Add L4 testability classification to an older run
uv run python -m pipeline.compose reprocess \
  --run-dir data/pipeline/compositions/20260518-204512/

# Run the test suite
uv run pytest -q
```

## Reading further

- `docs/PUBLIC_SNAPSHOT.md` — current goals, open decisions, and the June-pivot update
- `docs/roadmap-discovery-2026-06.md` — the discovery roadmap
- `findings/INDEX.md` — gold-standard track record of every verified run
- `findings/outcome-calibration-2026-05-29.md` — why L3/L4 verdicts are
  recorded but never used as filters
- `findings/transfer-catalog-20260530.md` — which findings are true
  transfer wins
- `findings/compose-20260519-153720.md` — the fusion run that produced
  the Rydberg-ECRH composition (C5)
- `data/pipeline/research/rydberg_fusion_synthesis.md` — independent
  four-agent novelty-check report on C5
- `findings/compose-20260519-162024.md` — TSP run, demonstrates
  domain-saturation detection in mature fields
