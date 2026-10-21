# Cache replacement — verification (clean negative)

**Source composition:** `compose-20260521-023514.md` C2
(successor-aware replacement: Markov n-gram + decay protecting likely-soon-followed residents)
**Run date:** 2026-05-21
**Prototype:** `experiments/cache_replacement/` (pure Python)

## Composition under test

L3 verdict on C2: WEAK. L4: PURELY TESTABLE (9 empirical / 0 fundamental).

C2's central claim:
> Maintain a small per-key successor sketch (top-k recent followers with decay).
> On access(K), pre-set the SIEVE "visited" bit on K's likely-soon-followed
> residents. Predicted successors are thus protected from eviction.

The L3 reviewer's strongest attacks:
- *"Not Novel: Markov/correlation-based prefetch is long-known"*
- *"Wrong Objective: predicting successors ≠ ranking residents for eviction"*
- *"Metadata Explosion at small cache sizes"*
- *"Missing admission policy"*

## Verification harness

Implemented LRU, FIFO, SIEVE, S3-FIFO, ARC (~300 LOC pure Python). Hit rates
on Zipf s=0.99 at 1% cache: LRU 0.40, SIEVE 0.51, S3FIFO 0.50, ARC 0.50 —
match published numbers from the SIEVE OSDI'24 paper. Baselines validated.

Then implemented **Successor-SIEVE** (~80 LOC) — SIEVE with a bounded-size
`source_key → top-k successor counts` table. On access(K), pre-set visited
bit on resident keys appearing as frequent successors of K.

## Results — hit rate, all policies × all traces × all cache sizes

| trace | cap | LRU | SIEVE | S3FIFO | ARC | **Succ-SIEVE** |
|---|---|---|---|---|---|---|
| zipf_s099 | 0.1% | 0.155 | 0.289 | 0.311 | 0.284 | 0.293 |
| zipf_s099 | 1% | 0.397 | 0.511 | 0.499 | 0.502 | 0.511 |
| zipf_s099 | 10% | 0.663 | 0.717 | 0.710 | 0.715 | 0.717 |
| zipf_s080 | 1% | 0.171 | 0.285 | 0.276 | 0.278 | 0.285 |
| zipf_s080 | 10% | 0.441 | 0.516 | 0.510 | 0.514 | 0.516 |
| scan_hot | 1% | 0.140 | 0.234 | 0.225 | 0.230 | 0.235 |
| scan_hot | 10% | 0.625 | 0.629 | 0.626 | 0.629 | 0.629 |
| **paired_zipf** † | 0.1% | 0.067 | 0.181 | 0.345 | 0.190 | 0.188 |
| **paired_zipf** † | 1% | 0.329 | 0.455 | 0.445 | 0.445 | 0.459 |
| paired_zipf | 10% | 0.637 | 0.707 | 0.693 | 0.695 | 0.707 |

† `paired_zipf` is a sanity-check workload where every odd-indexed access
strictly follows its paired even-indexed access (P(2k+1 | 2k) = 1.0). If
Successor-SIEVE works at all, it should win here.

**Succ-SIEVE is at-best tied with SIEVE.** Gains are <1% even on the paired
workload designed to favor it.

## Why it fails — exactly as the L3 reviewer predicted

When access(K) triggers protection of K's predicted successor B:
1. If B is resident: the *next* access (which IS B, since we predicted it
   well) sets B.visited=1 anyway. The pre-protection is redundant.
2. If B is non-resident: setting visited bit on a non-resident key does
   nothing — we can't protect what isn't there.
3. The window between "we predict B" and "B is actually requested" is
   essentially zero accesses on workloads where the predictor has signal.
   No eviction happens in that window.

This is the *"Wrong Objective: predicting successors ≠ ranking residents"*
critique materializing exactly. Successor prediction is useful for **prefetching**
(loading non-resident objects ahead of time, paying bandwidth) — not for
**replacement** (choosing among residents to evict).

S3-FIFO at 0.1% on paired_zipf scores **0.345** vs Succ-SIEVE 0.188 — almost
2× — because S3-FIFO's Ghost queue *does* solve a different problem: it
filters newly-arrived objects that don't show repeat behavior, which is
precisely the small-cache challenge. Successor-SIEVE doesn't have that
mechanism.

## Compose-verification track record (now n=3)

| run | L3 verdict | Verification outcome |
|---|---|---|
| short-string compression | WEAK + "equivalent to Brotli+dict / Zstd+dict / HPACK" | Partial true positive (LZ-family attack falsified; HPACK attack partially upheld) |
| multitrack stems | WEAK + "high-level correlation ≠ sample-level predictability" | **Clean true negative** (variance reduction <0.3%) |
| cache replacement | WEAK + "predicting successors ≠ ranking residents" | **Clean true negative** (Succ-SIEVE ≈ SIEVE everywhere) |

The L3 adversarial reviewer has been **directionally right in 2 of 3 cases**
when it gave WEAK verdicts. The L4 testability classifier has been correct
in all 3 cases (every test took ~1 session and produced clear empirical
signals). This is exactly the calibration shape you want from a research-finding
system: it surfaces ideas, you cheaply test them, and most don't survive —
but the ones that do (short-string C1 generalizing HPACK) are real.

## What would make C2 work?

The mechanism *could* work if reframed as a **prefetcher**: when K is accessed
and successor[K] → B has high confidence, *insert B into cache* before B is
actually requested. That changes the semantics from replacement to prefetching
and requires admission control to avoid pollution.

But the problem statement asked for a replacement policy. As a replacement
policy, C2 is empirically falsified by this benchmark.

## Caveats and what could change the result

1. **Synthetic traces only** (Zipfian, paired, scan_hot, loop). Real CDN /
   block / Twitter traces might have richer successor structure that
   showed signal. SNIA MSR Cambridge has historical block traces (~hours
   of OS-level I/O) that are publicly available — worth re-testing.
2. **Implementation knobs** are at defaults (budget=1024, top_k=4, min_count=2).
   A grid search might find a setting that wins — but the paired-workload
   ceiling suggests the mechanism's headroom is small.
3. **Larger cache sizes** weren't tested above 25%; some applications run
   at much larger ratios. But the L3 reviewer specifically asked about
   small-cache regimes (0.1-10%), which we tested.

## Reproducing

```
uv run python -m experiments.cache_replacement.bench
```

## Files

- `experiments/cache_replacement/traces.py` — Zipf/loop/scan_hot/paired_zipf
- `experiments/cache_replacement/policies.py` — LRU/FIFO/SIEVE/S3-FIFO/ARC
- `experiments/cache_replacement/successor_sieve.py` — C2 prototype
- `experiments/cache_replacement/bench.py` — runner

## Verdict

C2 successor-aware replacement is **falsified as a replacement policy** at
the linear-protection level on standard workloads. The L3 reviewer's
*"Wrong Objective"* attack was the correct diagnosis. The mechanism could be
revived as a *prefetcher* with admission control — different problem, not
what the compose pipeline was asked to solve.

This is the third compose run verified; tally now 1 partial-positive, 2
clean-negatives. The L3 adversarial filter is doing useful work, and the L4
classifier remains 3-for-3 on "this is worth testing in one session."
