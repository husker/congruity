"""Correctness tests for the oracle's cache policies and workload generator."""

from __future__ import annotations

import random

from oracle.policies import ARC, LRU, PagedEvictionStub
from oracle.workload import WorkloadConfig, generate_workload


def test_lru_basic_hit_after_insert():
    c = LRU(2)
    assert c.access("a") is False  # miss (insert)
    assert c.access("a") is True   # hit
    assert c.hits == 1
    assert c.misses == 1
    assert c.evictions == 0


def test_lru_evicts_oldest():
    c = LRU(2)
    c.access("a")        # cache: [a]
    c.access("b")        # cache: [a, b]
    c.access("a")        # hit; cache: [b, a] (b is now LRU)
    c.access("c")        # miss; evicts b; cache: [a, c]
    assert c.evictions == 1
    # Confirm what was evicted vs kept
    # (note: accessing "b" here would re-insert it and evict "a", so we
    # check stats rather than continuing to access)
    assert c.misses == 3
    assert c.hits == 1


def test_arc_basic_hit_after_insert():
    c = ARC(2)
    assert c.access("a") is False
    assert c.access("a") is True


def test_arc_invariants_under_random_workload():
    rng = random.Random(7)
    c = ARC(10)
    for _ in range(1000):
        c.access(str(rng.randint(0, 30)))
    # ARC invariants from the paper
    assert len(c.t1) + len(c.t2) <= c.capacity
    assert len(c.t1) + len(c.b1) <= c.capacity
    assert len(c.t2) + len(c.b2) <= 2 * c.capacity
    assert len(c.t1) + len(c.t2) + len(c.b1) + len(c.b2) <= 2 * c.capacity


def test_arc_stats_includes_internal_sizes():
    c = ARC(5)
    for k in "abcdefgh":
        c.access(k)
    s = c.stats()
    assert "p" in s
    assert "t1_size" in s and "t2_size" in s and "b1_size" in s and "b2_size" in s


def test_paged_stub_currently_matches_lru():
    # Until real attention-trace integration, PagedEvictionStub should be
    # bit-identical to LRU on the same key sequence.
    keys = ["a", "b", "a", "c", "d", "b", "a", "e", "f", "c"]
    lru = LRU(3)
    stub = PagedEvictionStub(3)
    for k in keys:
        assert lru.access(k) == stub.access(k)
    assert lru.hits == stub.hits and lru.misses == stub.misses


def test_workload_is_deterministic_with_seed():
    cfg = WorkloadConfig(n_requests=5, seed=42)
    events_1 = list(generate_workload(cfg))
    events_2 = list(generate_workload(cfg))
    assert events_1 == events_2


def test_workload_has_shared_prefix_keys():
    cfg = WorkloadConfig(n_requests=3, shared_prefix_blocks=5,
                         avg_prompt_blocks=2, avg_generate_blocks=2,
                         prompt_jitter=0, generate_jitter=0, seed=1)
    events = list(generate_workload(cfg))
    shared_keys = {e.key for e in events if e.key.startswith("shared_")}
    # 5 shared prefix blocks
    assert len(shared_keys) == 5
    # And each request reads each shared block at least once
    # Total shared reads = n_requests * shared_prefix_blocks
    shared_reads = sum(1 for e in events if e.key.startswith("shared_"))
    assert shared_reads == 3 * 5


def test_differential_test_runs_end_to_end():
    """Integration smoke test — the full pipeline must run and produce
    sensible numbers without crashing."""
    from oracle.runner import differential_test
    cfg = WorkloadConfig(n_requests=20, seed=1)
    result = differential_test(cfg, capacity=50, policy_names=["lru", "arc"])
    assert result["n_events"] > 0
    assert len(result["results"]) == 2
    for r in result["results"]:
        # Some events should have occurred for each policy
        assert r["hits"] + r["misses"] == result["n_events"]
        # Hit rate must be in [0, 1]
        assert 0.0 <= r["hit_rate"] <= 1.0


def test_arc_beats_lru_on_mixed_recency_frequency():
    """Sanity check: ARC implementation must beat LRU on a workload with
    both recency and frequency components. If this fails, ARC is broken.

    Pattern: a small hot set (accessed frequently throughout), interleaved
    with cyclic cold scans that flush LRU but should not flush ARC's T2.
    """
    keys = []
    hot = ["H1", "H2", "H3"]
    # Repeated cycles: read hot, then a long cold scan, then hot again
    for cycle in range(20):
        for _ in range(5):
            keys.extend(hot)  # hot set, accessed many times
        # Cold scan that exceeds cache capacity, defeats pure LRU
        for i in range(50):
            keys.append(f"cold_{cycle}_{i}")
    # Final hot reads
    for _ in range(10):
        keys.extend(hot)

    lru = LRU(10)
    arc = ARC(10)
    for k in keys:
        lru.access(k)
        arc.access(k)
    # ARC should retain hot items in T2 through cold scans; LRU evicts them.
    assert arc.hit_rate > lru.hit_rate, (
        f"ARC failed to beat LRU on mixed workload: "
        f"LRU={lru.hit_rate:.1%}, ARC={arc.hit_rate:.1%}"
    )
