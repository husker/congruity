# Project deep analysis - 2026-05-21

## Bottom line

Congruity is genuinely promising. The core loop is strong:

```text
compose -> adversarial review -> cheap empirical verification
```

The most encouraging signal is that the project does not only confirm its own
outputs. Across five verified compose-pipeline outputs, the current track record
is:

| Outcome type | Count |
|---|---:|
| Partial positive | 1 |
| Clean positive | 1 |
| Clean negative | 3 |

L4 has correctly identified five of five cases as testable in one session with
a clear signal. L3 has been directionally correct in four of five WEAK verdicts
and empirically wrong in one. That is the right shape for a useful research
triage system: the adversary is informative but not final.

The next quality jump should come from tighter corpus gates, stronger
auditability, stricter benchmark harnesses, and better product reporting around
calibration.

## Validation run

Commands run during this audit:

```bash
uv run pytest
uv run ruff check .
uv run --with brotli --with zstandard --with hpack python -m experiments.c1_short_string.bench --small
uv run python -m oracle.runner --n-requests 50 --capacity 50 --policies lru,arc,paged_stub
uv run python -m experiments.cache_replacement.bench
```

Results:

- `uv run pytest`: 30 passed.
- `uv run ruff check .`: failed with 20 lint issues.
- C1 short-string small benchmark matched the written conclusion: C1 has real signal on JSON and real HTTP headers, but loses to stateful HPACK on HTTP.
- Cache benchmark confirmed Succ-SIEVE is effectively tied with SIEVE.
- `git status --short`: clean after the audit.

## What works

The project has a strong epistemic posture. The docs and verification writeups
are honest about wins, losses, caveats, and baseline strength. That is unusual
and valuable.

The compose pipeline is operational across multiple domains. Saved compose
state shows 14 complete runs, 64 surfaced compositions, 34 WEAK and 30 INVALID
L3 verdicts, plus a useful zero-composition "refuse to fabricate" mode.

The empirical verification arm is the project's strongest asset. The harnesses
are small enough to inspect and rerun, and the findings documents give clear
claims, baselines, results, and verdicts.

The project has learned from prior failures. The Step 1.5 corpus quality gate
was added after corrupted OpenAlex metadata caused misleading early results.
That is exactly the right kind of system improvement.

## Main accuracy risks

### 1. Corpus relevance is currently optional

In saved `corpus_gate.json` files, the `relevance` check was skipped because no
OpenAI API key was present. The code treats missing or failed embeddings as a
non-fatal warning:

- `pipeline/corpus_gate.py:133`
- `pipeline/corpus_gate.py:141`
- `pipeline/corpus_gate.py:287`

That means the advertised Step 1.5 gate often only checks structure and
encoding, not whether papers are actually relevant to the problem.

Recommendation: make relevance mandatory for non-dry-run compose runs, or add a
local fallback embedding model. At minimum, surface "relevance skipped" in the
digest as a risk badge.

### 2. Saved corpus can diverge from the actual run corpus

`compose_one` writes `corpus.json` before Step 1.5 drops bad papers, then
reindexes survivors. Later reprocessing loads the pre-drop corpus.

Relevant code:

- `pipeline/compose.py:814`
- `pipeline/compose.py:850`
- `pipeline/compose.py:1074`

This can make later L4 reprocessing or audit work map `P1`, `P2`, etc. to the
wrong papers.

Recommendation: after Step 1.5, write both:

- `corpus_raw.json`
- `corpus_validated.json`

Then store a stable `paper_uid` alongside display IDs so reindexing cannot
break references.

### 3. LLM stages are under-auditable

Successful LLM calls mostly save parsed JSON, but not the full prompt and raw
response for each stage. `run_codex` returns output, but there is no consistent
artifact layer around it.

Relevant code:

- `pipeline/compose.py:51`
- `pipeline/compose.py:166`
- `pipeline/compose.py:229`
- `pipeline/compose.py:322`
- `pipeline/compose.py:737`

Recommendation: every LLM stage should save:

- prompt text
- raw stdout
- raw stderr
- parsed JSON
- parse status
- model/reasoning config

Also replace the regex/brace parser with strict schema validation and retry once
with a repair prompt when JSON is invalid.

### 4. Source grounding is too permissive for "verbatim quote"

The compose prompt asks for verbatim quotes, but the compose prompt only includes
truncated abstracts. L2 also accepts 70% token overlap as `PARAPHRASED_MATCH`.

Relevant code:

- `pipeline/compose.py:257`
- `pipeline/compose.py:493`

Recommendation: require exact spans from full abstracts for PASS. Keep
paraphrase matching, but classify it as WARN rather than grounded. The digest
should distinguish exact support from approximate support.

### 5. L4 testability can over-recommend weak compositions

L4 correctly identifies many objections as empirical, but "empirical" is not the
same as "worth testing." Some INVALID compositions are still labeled PURELY
TESTABLE, which is useful but easy to misread as endorsement.

Recommendation: split L4 into two scores:

- `testability`: can this be tested?
- `investment_priority`: should this be tested first?

Investment priority should include L3 verdict, baseline clarity, expected effect
size, implementation cost, novelty risk, and whether the problem statement was
actually answered.

## Benchmark harness issues

### Embed quant

Issues found:

- `exact` is computed and unused.
- `calib_db` is assigned and unused.
- exact top-k is recomputed inside every loop.
- `np.argpartition(-sims, k)` should use `k - 1` for top-k.
- C3 methods labeled as 128b/256b sometimes use fewer bits after allocation rounding.

Relevant files:

- `experiments/embed_quant/bench.py:20`
- `experiments/embed_quant/bench.py:47`
- `experiments/embed_quant/bench.py:78`
- `experiments/embed_quant/c3_pca_alloc.py:145`
- `experiments/embed_quant/c3_pca_alloc.py:231`

The negative result likely still holds, but the harness should be tightened
before treating the table as canonical.

### Multitrack stems

The residual encoder clips overflow and records `overflow_count`, but does not
encode a true lossless fallback for overflow samples.

Relevant file:

- `experiments/multitrack_stems/predictor.py:95`

Because the method loses even with this favorable clipping, the negative
conclusion likely holds. But the codec should not be described as lossless until
overflow fallback is implemented and round-trip tests exist.

### C1 short-string compression

C1 uses Python's randomized `hash()` for minimizer signatures, making candidate
lookup non-reproducible across processes.

Relevant file:

- `experiments/c1_short_string/c1.py:70`

Recommendation: replace with a stable hash such as BLAKE2 or xxhash-style
integer hashing. Also add round-trip tests over all corpora and a hierarchy
ablation, since the composition named hierarchy but the prototype uses flat
templates.

### Cache replacement

The cache benchmark is coherent and the negative result is convincing for the
tested mechanism. The main product improvement is to add real traces.

Recommendation: add SNIA/MSR Cambridge-style public block traces, plus an
optional prefetcher variant, because the current finding correctly diagnoses
that successor prediction is a prefetching mechanism rather than a replacement
mechanism.

### Heilbronn

The positive result is real against the implemented baselines, but not against
published SOTA. The writeup says this honestly.

Recommendation: add stronger baselines before broader sharing:

- CMA-ES
- differential evolution
- tuned basin hopping
- longer SA sweeps
- multiple seeds with confidence intervals
- schedule ablations for C1

## Product improvements

### 1. Add a calibration dashboard

The product should expose:

- number of compose runs
- L3 verdict mix
- L4 testability mix
- verified outcomes
- false-positive and true-positive rates
- skipped gates
- cost per run and per surfaced composition
- top recurring L3 failure modes

The track record is the product's trust engine. It should be first-class.

### 2. Generate a verification recipe per candidate

Every WEAK + PURELY_TESTABLE composition should emit:

- named baseline(s)
- benchmark dataset
- metric
- minimum meaningful effect size
- expected runtime
- likely failure mode
- stop rule

This turns the system from "idea generator" into "research triage and test
planner."

### 3. Automate novelty checks after L3

The strongest current gap is novelty. The Rydberg-fusion novelty check was
manual, and the early CS bridge novelty checks were partial.

Recommendation: add a post-L3 novelty stage for all WEAK/PASS survivors:

- search exact synthesis terms
- search constituent-paper pairings
- inspect references/citations
- classify as known, adjacent-known, not-found, or unverified
- save evidence links and queries

### 4. Add domain packs

The engine is domain-agnostic, but useful runs depend heavily on domain-specific
query design and baseline knowledge.

A domain pack should include:

- topic-harvest queries
- problem statement templates
- known baselines
- evaluation datasets
- gotchas
- recommended transient dependencies

### 5. Run an expert hit-rate study

This is the biggest unresolved product truth test. Blind domain experts should
rate random surfaced compositions:

- interesting and novel
- interesting but known
- obvious/folklore
- wrong
- not actionable

The framework is much stronger if expert "interesting and not obvious" rate is
above roughly 15%. If it is under 5%, the system is mostly producing smooth
noise with good verification hygiene.

## Prioritized remediation plan

### Phase 1: correctness and auditability

1. Fix `ruff` failures.
2. Save post-gate `corpus_validated.json` and preserve stable paper IDs.
3. Persist prompt/raw-response artifacts for every LLM stage.
4. Add schema validation for dimensions, scores, compositions, L1, L3, and L4.
5. Make skipped relevance checks visible and preferably fatal for non-dry-run compose.

### Phase 2: benchmark hardening

1. Add round-trip tests for compression experiments.
2. Fix embed-quant top-k, unused calibration split, and bit-budget labeling.
3. Add stronger Heilbronn baselines and seed sweeps.
4. Add real cache traces.
5. Add test cases for full compose with fake OpenAlex and fake Codex outputs.

### Phase 3: product layer

1. Generate verification recipes automatically.
2. Add novelty-check stage.
3. Build a run dashboard or at least a structured markdown calibration report.
4. Introduce domain packs.
5. Run expert hit-rate study.

## Verdict

The project should not become "more generation." It should become more
calibrated, more auditable, and more benchmark-tight.

The key product promise is not that Congruity always finds winners. It is that
it produces falsifiable research directions, quickly separates the weak from the
interesting, and records its own calibration honestly.

That promise is already visible. The code and product can improve most by making
every stage traceable, every benchmark reproducible, and every recommendation
grounded in the growing verification track record.
