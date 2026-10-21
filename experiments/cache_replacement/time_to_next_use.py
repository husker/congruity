"""Time-to-Next-Use (TTNU): cheap online Belady approximation.

Mechanism (C5 from compose-20260521-023514, an L3=invalid / "not novel"
composition we are verifying to test gate precision):

    "Predict each cached object's next request window, then evict the object
    whose scheduled future contribution is weakest under capacity. Approximates
    Belady-style replacement using cheap predictors and decayed next-use
    estimates rather than neural inference."

L3 rejected this as not novel relative to reuse-distance prediction / Hawkeye /
learned replacement, and objected that the cited papers "do not supply the
missing cache-specific machinery: an online, cheap, accurate next-use predictor
under tiny metadata budgets." So the implementation supplies exactly that and
nothing fancier:

    - A virtual clock `t` (increments per access).
    - Per-key EWMA of observed inter-arrival gaps (the reuse-distance estimate),
      kept in a bounded LRU `gap_table` so the metadata footprint is fixed
      regardless of object-id space — this is the "tiny metadata budget" the
      reviewer said was missing.
    - Predicted next-use time = last_access_time + ewma_gap. First-seen keys
      use the running global-mean gap as their estimate.
    - Eviction = the resident key with the LARGEST predicted next-use time
      (farthest future = weakest contribution), via a lazy max-heap with
      version tags for O(log n) amortized eviction.

No neural inference, no offline planning. Pure Python, integer keys, hit-rate
only — same interface as the other policies in policies.py.
"""

from __future__ import annotations

import heapq
from collections import OrderedDict


class TimeToNextUse:
    """Belady-approximating eviction via cheap EWMA next-use prediction."""

    def __init__(
        self,
        capacity: int,
        alpha: float = 0.5,
        metadata_budget: int = 8192,
    ):
        self.capacity = capacity
        self.alpha = alpha
        self.metadata_budget = metadata_budget
        self.t = 0
        self.resident: set[int] = set()
        self.last: dict[int, int] = {}
        # Bounded LRU of per-key EWMA inter-arrival gap (survives eviction).
        self.gap_table: OrderedDict[int, float] = OrderedDict()
        # Lazy max-heap of (-pred_next_use, version, key); stale entries skipped.
        self._heap: list[tuple[float, int, int]] = []
        self._version: dict[int, int] = {}
        # Running global-mean gap, used as the prior for first-seen keys.
        self._gap_sum = 0.0
        self._gap_n = 0
        self.name = f"TTNU(a={alpha},m={metadata_budget})"

    # -- metadata budget management -----------------------------------------
    def _set_gap(self, key: int, gap: float) -> None:
        if key in self.gap_table:
            self.gap_table.move_to_end(key)
        self.gap_table[key] = gap
        while len(self.gap_table) > self.metadata_budget:
            self.gap_table.popitem(last=False)

    def _global_mean_gap(self) -> float:
        if self._gap_n == 0:
            # No repeats observed yet: assume one capacity-worth of distance.
            return float(max(1, self.capacity))
        return self._gap_sum / self._gap_n

    # -- heap management -----------------------------------------------------
    def _push(self, key: int, pred_next: float) -> None:
        ver = self._version.get(key, 0) + 1
        self._version[key] = ver
        heapq.heappush(self._heap, (-pred_next, ver, key))

    def _evict(self) -> None:
        while self._heap:
            _, ver, key = heapq.heappop(self._heap)
            if key not in self.resident or self._version.get(key) != ver:
                continue  # stale entry
            self.resident.discard(key)
            self.last.pop(key, None)
            return

    # -- main interface ------------------------------------------------------
    def access(self, key: int) -> bool:
        self.t += 1
        t = self.t

        if key in self.resident:
            gap = t - self.last[key]
            prev = self.gap_table.get(key)
            ewma = float(gap) if prev is None else self.alpha * gap + (1 - self.alpha) * prev
            self._set_gap(key, ewma)
            self.last[key] = t
            self._gap_sum += gap
            self._gap_n += 1
            self._push(key, t + ewma)
            return True

        # Miss: insert (evicting the farthest-future resident if full).
        if len(self.resident) >= self.capacity:
            self._evict()
        known = self.gap_table.get(key)
        ewma = known if known is not None else self._global_mean_gap()
        self.resident.add(key)
        self.last[key] = t
        self._set_gap(key, ewma)
        self._push(key, t + ewma)
        return False


class TimeToNextUse_Default(TimeToNextUse):
    """Convenience class with default knobs for the bench."""

    def __init__(self, capacity: int):
        super().__init__(capacity, alpha=0.5, metadata_budget=8192)
        self.name = "TTNU"
