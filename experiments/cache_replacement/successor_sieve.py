"""Successor-SIEVE: SIEVE + bounded successor-counter table.

Mechanism (C2 from compose-20260521-023514):
    Maintain a small per-key successor sketch: how often, recently, was each
    other key requested *next* after K. On access(K), look up K's top-N
    successors and pre-set the SIEVE "visited" bit on any of those that are
    currently resident. The standard SIEVE eviction logic then naturally
    protects predicted-soon objects.

Metadata bound: at most `successor_budget` distinct source keys × `top_k`
successor entries. The successor table is itself LRU-bounded (oldest key
dropped when budget exceeded).

Two knobs to sweep:
    successor_budget: cap on number of source-key entries
    min_count: minimum observed count before a successor is considered "likely"
"""

from __future__ import annotations

from collections import OrderedDict

from experiments.cache_replacement.policies import SIEVE


class SuccessorSIEVE(SIEVE):
    """SIEVE + bounded successor sketch.

    `top_k` per-source caps the per-key memory: each source key tracks at
    most top_k distinct successors with frequency counts.
    """

    def __init__(
        self,
        capacity: int,
        successor_budget: int = 1024,
        top_k: int = 4,
        min_count: int = 2,
    ):
        super().__init__(capacity)
        self.successor: OrderedDict[int, dict[int, int]] = OrderedDict()
        self.successor_budget = successor_budget
        self.top_k = top_k
        self.min_count = min_count
        self.last_key: int | None = None
        self.name = f"Succ-SIEVE(b={successor_budget},k={top_k},m={min_count})"

    def _record_edge(self, prev_key: int, key: int) -> None:
        d = self.successor.get(prev_key)
        if d is None:
            if len(self.successor) >= self.successor_budget:
                self.successor.popitem(last=False)
            d = {}
            self.successor[prev_key] = d
        self.successor.move_to_end(prev_key)
        d[key] = d.get(key, 0) + 1
        if len(d) > self.top_k:
            # Drop the smallest-count entry
            victim = min(d, key=lambda k: d[k])
            del d[victim]

    def _protect_likely_successors(self, key: int) -> None:
        d = self.successor.get(key)
        if not d:
            return
        for next_key, count in d.items():
            if count >= self.min_count and next_key in self._visited:
                self._visited[next_key] = True

    def access(self, key: int) -> bool:
        # Record edge before standard SIEVE access
        if self.last_key is not None and self.last_key != key:
            self._record_edge(self.last_key, key)
        self.last_key = key

        hit = super().access(key)

        # Pre-protect K's likely successors that are resident
        self._protect_likely_successors(key)
        return hit


class SuccessorSIEVE_Default(SuccessorSIEVE):
    """Convenience class with default knobs for the bench."""

    def __init__(self, capacity: int):
        super().__init__(capacity, successor_budget=1024, top_k=4, min_count=2)
        self.name = "Succ-SIEVE"
