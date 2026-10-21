"""Cache replacement policies for differential testing.

Policies in this module are plug-compatible: same `access(key) -> bool`
contract, same stats interface. The runner can swap them and compare hit
rates on identical workloads.

Three policies ship in v0:
- `LRU`: standard least-recently-used baseline.
- `ARC`: Megiddo & Modha (2003) Adaptive Replacement Cache — the candidate
  ported from the database/OS literature.
- `PagedEvictionStub`: placeholder for the current LLM-serving SOTA.
  Currently behaves as LRU; needs real attention-trace integration before
  the differential test against ARC means anything for the actual bridge
  claim.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Hashable


class CachePolicy(ABC):
    """Plug-compatible cache replacement policy."""

    def __init__(self, capacity: int):
        if capacity <= 0:
            raise ValueError("capacity must be > 0")
        self.capacity = capacity
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    @abstractmethod
    def access(self, key: Hashable) -> bool:
        """Read `key`. Returns True on hit, False on miss. Evicts on full
        miss. Implementations must update self.hits / self.misses /
        self.evictions."""

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return (self.hits / total) if total else 0.0

    def stats(self) -> dict:
        return {
            "policy": self.name,
            "capacity": self.capacity,
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "hit_rate": self.hit_rate,
        }


class LRU(CachePolicy):
    """Standard least-recently-used cache."""

    def __init__(self, capacity: int):
        super().__init__(capacity)
        self._od: OrderedDict[Hashable, None] = OrderedDict()

    def access(self, key: Hashable) -> bool:
        if key in self._od:
            self._od.move_to_end(key)
            self.hits += 1
            return True
        self.misses += 1
        if len(self._od) >= self.capacity:
            self._od.popitem(last=False)
            self.evictions += 1
        self._od[key] = None
        return False


class ARC(CachePolicy):
    """Megiddo & Modha (2003) Adaptive Replacement Cache.

    Maintains two LRU lists (T1 for recency-once, T2 for frequency) and two
    ghost lists (B1, B2) tracking recently-evicted keys from each side. An
    adaptive parameter `p` ∈ [0, capacity] sets the target size of T1; ghost
    hits on B1 or B2 nudge `p` toward the side that just made a mistake.

    Reference: N. Megiddo & D. Modha, "ARC: A Self-Tuning, Low Overhead
    Replacement Cache," USENIX FAST 2003.

    Invariants (asserted in tests):
        |T1| + |T2| ≤ capacity
        |T1| + |B1| ≤ capacity
        |T2| + |B2| ≤ 2 * capacity
        |T1| + |T2| + |B1| + |B2| ≤ 2 * capacity
    """

    def __init__(self, capacity: int):
        super().__init__(capacity)
        self.c = capacity
        self.p = 0  # target size for T1
        self.t1: OrderedDict[Hashable, None] = OrderedDict()
        self.t2: OrderedDict[Hashable, None] = OrderedDict()
        self.b1: OrderedDict[Hashable, None] = OrderedDict()
        self.b2: OrderedDict[Hashable, None] = OrderedDict()

    def _replace(self, key: Hashable) -> None:
        """REPLACE subroutine — evict from T1 or T2 based on `p`."""
        t1_len = len(self.t1)
        if t1_len > 0 and (
            (key in self.b2 and t1_len == self.p) or t1_len > self.p
        ):
            old, _ = self.t1.popitem(last=False)
            self.b1[old] = None
        else:
            if self.t2:
                old, _ = self.t2.popitem(last=False)
                self.b2[old] = None
            elif self.t1:
                old, _ = self.t1.popitem(last=False)
                self.b1[old] = None
        self.evictions += 1

    def access(self, key: Hashable) -> bool:
        # Case I: hit
        if key in self.t1:
            del self.t1[key]
            self.t2[key] = None
            self.hits += 1
            return True
        if key in self.t2:
            self.t2.move_to_end(key)
            self.hits += 1
            return True

        # Miss
        self.misses += 1

        # Case II: in B1 — recency-list eviction was wrong, shift toward recency
        if key in self.b1:
            delta = max(len(self.b2) // max(len(self.b1), 1), 1)
            self.p = min(self.c, self.p + delta)
            self._replace(key)
            del self.b1[key]
            self.t2[key] = None
            return False

        # Case III: in B2 — frequency-list eviction was wrong, shift toward frequency
        if key in self.b2:
            delta = max(len(self.b1) // max(len(self.b2), 1), 1)
            self.p = max(0, self.p - delta)
            self._replace(key)
            del self.b2[key]
            self.t2[key] = None
            return False

        # Case IV: full miss, key not seen before (or long ago)
        if len(self.t1) + len(self.b1) == self.c:
            if len(self.t1) < self.c:
                self.b1.popitem(last=False)
                self._replace(key)
            else:
                self.t1.popitem(last=False)
                self.evictions += 1
        elif len(self.t1) + len(self.b1) < self.c:
            total = len(self.t1) + len(self.t2) + len(self.b1) + len(self.b2)
            if total >= self.c:
                if total >= 2 * self.c and self.b2:
                    self.b2.popitem(last=False)
                self._replace(key)

        self.t1[key] = None
        return False

    def stats(self) -> dict:
        s = super().stats()
        s.update({
            "p": self.p,
            "t1_size": len(self.t1),
            "t2_size": len(self.t2),
            "b1_size": len(self.b1),
            "b2_size": len(self.b2),
        })
        return s


class PagedEvictionStub(CachePolicy):
    """Placeholder for LLM-serving SOTA (Zipage / IceCache / PagedEviction class).

    Current behavior: LRU. The real PagedEviction policy evicts based on per-
    token attention scores, which our synthetic workload doesn't carry.

    TODO before this is a meaningful baseline:
        1. Capture per-token attention scores from a vLLM trace on a real
           workload (e.g. ShareGPT-style prompts).
        2. Extend `AccessEvent` to carry per-block attention score.
        3. Re-implement this policy to evict the lowest-scored block first.
        4. Re-run the differential test.

    Until that work lands, an ARC-vs-PagedEvictionStub result IS NOT EVIDENCE
    for or against the bridge claim — it's just ARC-vs-LRU twice.
    """

    def __init__(self, capacity: int):
        super().__init__(capacity)
        self._lru = LRU(capacity)

    def access(self, key: Hashable) -> bool:
        hit = self._lru.access(key)
        self.hits = self._lru.hits
        self.misses = self._lru.misses
        self.evictions = self._lru.evictions
        return hit


POLICIES: dict[str, type[CachePolicy]] = {
    "lru": LRU,
    "arc": ARC,
    "paged_stub": PagedEvictionStub,
}
