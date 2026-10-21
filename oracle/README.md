# oracle — differential testing for the KV-cache ↔ cache-replacement bridge

This package operationalizes the v0 bridge claim from `findings/01-kvcache-cache-replacement.md`: **LLM serving's KV-cache eviction policies are operationally equivalent to adaptive cache replacement in databases/systems.** If true, applying ARC (Megiddo & Modha 2003) to a KV-cache should be competitive with or beat current LLM-serving eviction strategies on real workloads.

The oracle is the cheapest experiment that can refute that claim. It runs identical workloads through multiple cache policies and compares hit rates. If ARC fails to beat LRU on LLM-shaped workloads, the bridge is wrong. If ARC wins decisively, that's evidence the bridge is real and the LLM-serving community is leaving performance on the table.

## What's here

```
oracle/
├── __init__.py
├── workload.py        # Synthetic LLM-serving access-pattern generator
├── policies.py        # LRU, ARC, PagedEvictionStub
├── runner.py          # Differential-test harness
└── __main__.py        # CLI entry: python -m oracle
```

## Run it

```bash
uv run python -m oracle --n-requests 200 --capacity 500 --policies lru,arc,paged_stub
```

Or with a JSON dump:

```bash
uv run python -m oracle --n-requests 500 --capacity 1000 --output data/oracle/run1.json
```

Vary capacity to see how the spread between policies behaves under pressure. ARC's advantage is typically biggest in the squeezed regime (capacity ≈ working-set size).

## What v0 proves, and doesn't

**Proves (if ARC wins on synthetic workloads):**
- ARC's adaptive-recency-vs-frequency split is a real signal even on LLM-shaped access patterns.
- The bridge claim has at least the basic shape correct.

**Does NOT prove (limits of synthetic workloads):**
- That ARC wins on REAL vLLM traces. Real workloads carry attention scores; the synthetic version doesn't.
- That ARC beats specifically PagedEviction's policy. The PagedEvictionStub is currently identical to LRU — that's documented in policies.py.

The honest claim a positive synthetic result supports: "this is worth running on real vLLM traces." Not: "the bridge is verified."

## Open work (TODO before this is a real validation oracle)

1. **Capture real KV-cache access traces** from vLLM running ShareGPT-style or production-like prompts. The bench harness from vLLM has telemetry hooks; we want per-block access timing.
2. **Extend `AccessEvent` to carry per-block attention scores** so PagedEvictionStub can be replaced by a real implementation of the Zipage/PagedEviction policy.
3. **Re-implement `PagedEvictionStub` as `PagedEviction`** using captured attention scores.
4. **Re-run the differential test on real traces.** That's the validation that the bridge claim is real-world correct.

Each of these is a 1–3 day task. The package structure is set up so swapping policies is a one-file edit, and the runner is workload-agnostic.

## Tests

```bash
uv run pytest tests/test_oracle.py -v
```

Tests cover: LRU correctness, ARC invariants (|T1|+|T2| ≤ capacity, etc.), ARC vs. LRU on a deliberately skewed workload where ARC's adaptive split should help.

## Reproducibility

The workload is seeded (`--seed`, default 42). Same flags → same events → same results. Save the JSON output (`--output`) to attach to a findings doc.
