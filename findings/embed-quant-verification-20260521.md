# Embedding quantization — verification (clean negative)

**Source compositions:** `compose-20260521-055931.md` (0 surfaced) and
`compose-20260521-060508.md` (6 surfaced, C3 WEAK)
**Run date:** 2026-05-21
**Prototype:** `experiments/embed_quant/` (Python + sklearn)

## Two-stage outcome

**First compose run** (year-min=2018): Step 3 scored only 10 papers; Step 4
produced 0 candidate compositions. The engine **declined to fabricate** —
this is itself a useful calibration point. The corpus didn't have enough
foundational quantization papers (PQ 2011, OPQ 2013, JPEG quantization
1990s) under the year-min filter.

**Second compose run** (year-min=2010): Step 3 scored 185 papers. Step 4
produced 6 compositions. C3 was the only WEAK one (rest INVALID, plus one
L1 consistency failure).

## C3 composition

L3 verdict: WEAK. L4: PURELY TESTABLE (10/0).

C3 claim:
> Divide embeddings into PCA bands; estimate each band's local retrieval
> utility on calibration neighborhoods; allocate unequal bits per band
> (more bits to more useful bands) so the integrated compressed vector
> exceeds a uniform allocation.

L3 reviewer's strongest attacks (all proved right):
- *"Already folklore"* — JPEG quantization tables, OPQ, Matryoshka, variable-rate PQ all do unequal allocation
- *"Recall objective mismatch"* — utility ≠ marginal value
- *"Low-bit regime likely breaks it"* — at 32 bits one band gets 0 bits

## Verification setup

`experiments/embed_quant/`:
- 5000 sentence-transformers/all-MiniLM-L6-v2 embeddings (384d) of 20 Newsgroups documents, L2-normalized
- 4500 db / 500 queries (held-out); 200 separate calib queries for C3-utility
- Baselines: scalar-int8, scalar-int4, binary-1bit, PQ(m=4,8,16,32, k=256)
- C3 variants:
  - **C3-variance**: bits ∝ log(band eigenvalue) — folklore allocation
  - **C3-utility**: bits ∝ per-band recall@10 contribution from calibration — C3-specific

Recall@10 = mean overlap of approx top-10 with exact top-10 over queries.

## Results — recall@10 vs bits/vector, lower bits = harder

| method | bits | recall@10 |
|---|---|---|
| float32 | 12288 | 1.0000 |
| scalar-int8 | 3072 | 0.9952 |
| scalar-int4 | 1536 | 0.9364 |
| binary-1bit | 384 | 0.7042 |
| PQ-32b | 32 | **0.2590** |
| C3-var-32b | 32 | 0.2614 (~tie, but band allocation 2/1/1/0 has dead band) |
| C3-util-32b | 32 | 0.1942 (-25% vs PQ) |
| PQ-64b | 64 | **0.4086** |
| C3-var-64b | 64 | 0.3844 (-6%) |
| C3-util-64b | 64 | 0.3624 (-11%) |
| PQ-128b | 128 | **0.5558** |
| C3-var-128b | 112 | 0.5104 (-8%, also fewer bits) |
| C3-util-128b | 120 | 0.4986 (-10%) |
| PQ-256b | 256 | **0.6870** |
| C3-var-256b | 224 | 0.6490 (-6%, also fewer bits) |
| C3-util-256b | 240 | 0.6366 (-7%) |

PQ wins at every bit budget. C3-variance is folklore-equivalent (and the
allocation rounding even hurts it slightly). C3-utility is *worse than*
C3-variance — the supposedly novel allocation criterion is empirically bad.

## Why utility-based allocation fails

The L3 reviewer's *"recall objective mismatch"* attack was the right call.
Per-band recall@10 contribution measures **total** retrieval value when that
band is kept alone. But for retrieval after quantization, you want bits
where **marginal** quantization noise hurts the ranking most.

The high-variance / high-utility band keeps most of its retrieval power
even with few bits, because the angular structure is dominated by its
biggest eigenvectors. Giving it more bits has diminishing returns.

Low-variance / low-utility bands, conversely, can't be saved by extra
bits because their information is mostly noise. Extra bits there encode
noise more precisely, which doesn't help recall.

Result: the "obvious" utility-based allocation puts bits in the wrong
place. PQ's uniform allocation is closer to optimal than the C3 heuristic.

## What this falsifies

| L3 attack on C3 | Verdict |
|---|---|
| Already folklore | Confirmed — variance allocation = uniform PQ in this regime |
| P171 (heterostructured materials) analogy has no mechanism | Confirmed — analogy is metaphorical |
| P73 (single-cell modalities) doesn't transfer | Confirmed — PCA bands aren't modalities |
| Side-information budget ignored | Implementation issue — bit budget rounding caused some bands to get 0 m |
| Missing distance computation | We sidestepped via reconstruction; full asymmetric LUT would not have helped |
| Recall objective mismatch | **Confirmed empirically** — utility allocation is worse than variance, both worse than uniform |
| Baselines under-specified | Partially — we compared to PQ only; OPQ would likely also beat C3 |
| Low-bit regime breaks it | Confirmed — at 32 bits one band gets m=0 |
| No cross-subspace correlation recovery | Confirmed by construction — banded PQ can't model cross-band correlation |

## What this is

A clean true-negative. The composition surfaced by congruity (PCA bands +
utility-allocated bits) is **empirically falsified** as a method to beat
uniform PQ on the recall@10/bits Pareto frontier. The L3 adversarial
reviewer's recall-objective-mismatch attack was the correct diagnosis.

## Compose-verification track record (now n=5)

| run | L3 verdict | Verification outcome |
|---|---|---|
| short-string compression | WEAK + "equivalent to Brotli+dict / Zstd+dict / HPACK" | Partial positive |
| multitrack stems | WEAK + "high-level correlation ≠ sample-level predictability" | Clean negative |
| cache replacement | WEAK + "successors ≠ residents" | Clean negative |
| Heilbronn triangle | WEAK + "folklore + no evidence vs SA" | Clean positive |
| **embedding quantization** | WEAK + "folklore + recall objective mismatch" | **Clean negative** |

Track record: 1 partial-positive + 1 clean-positive + 3 clean-negatives.
L3 directionally correct in 4 of 5 WEAK verdicts. L4 PURELY_TESTABLE
classifier 5-for-5 on "testable in one session, clear signal."

Also notable: **the engine has a refuse-to-fabricate mode**. Run 5a
(year-min=2018) produced 0 compositions because Step 3 only scored 10
papers, signalling the corpus lacked depth on the topic. Re-running with
year-min=2010 surfaced compositions, but they all failed adversarial
review or empirical test.

## Reproducing

```
uv run --with numpy --with sentence-transformers --with scikit-learn python \
  -m experiments.embed_quant.bench
```

## Files

- `experiments/embed_quant/corpus.py` — MiniLM embeddings of 20 Newsgroups
- `experiments/embed_quant/baselines.py` — scalar/binary/PQ baselines
- `experiments/embed_quant/c3_pca_alloc.py` — C3 prototype (PCA-banded PQ)
- `experiments/embed_quant/bench.py` — runner

## Verdict

C3 is **falsified** as a path to beating uniform PQ. The L3 reviewer's
"folklore + recall objective mismatch" diagnosis is empirically vindicated.
Embedding quantization remains a Pareto frontier where uniform-PQ-with-
calibration is hard to dislodge with simple allocation tweaks. A
genuinely novel method would need to (a) model cross-band correlation
(C3 can't), (b) use a ranking-loss objective (not utility), or (c) jointly
train embedding + quantization (matryoshka territory). None of those is
what C3 proposed.
