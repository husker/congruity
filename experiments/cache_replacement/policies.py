"""Cache replacement policy implementations.

Each policy implements:
    .access(key) -> bool   # True = hit, False = miss (key is inserted on miss)
    .capacity                # int, max number of objects

Pure Python, integer keys, no value storage (we measure hit rate only).
"""

from __future__ import annotations

from collections import OrderedDict, deque
from typing import Protocol


class Policy(Protocol):
    name: str
    capacity: int

    def access(self, key: int) -> bool: ...


class LRU:
    """Classic Least-Recently-Used: O(1) per access via OrderedDict."""

    name = "LRU"

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.od: OrderedDict[int, None] = OrderedDict()

    def access(self, key: int) -> bool:
        if key in self.od:
            self.od.move_to_end(key)
            return True
        if len(self.od) >= self.capacity:
            self.od.popitem(last=False)
        self.od[key] = None
        return False


class FIFO:
    """Plain FIFO: no recency tracking. Useful as a floor baseline."""

    name = "FIFO"

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.set: set[int] = set()
        self.q: deque[int] = deque()

    def access(self, key: int) -> bool:
        if key in self.set:
            return True
        if len(self.set) >= self.capacity:
            old = self.q.popleft()
            self.set.discard(old)
        self.set.add(key)
        self.q.append(key)
        return False


class SIEVE:
    """SIEVE (Zhang et al., OSDI 2024).

    Single FIFO queue + 1-bit "visited" per entry. A "hand" pointer walks
    from tail toward head; on visited=1 it clears the bit and advances; on
    visited=0 it evicts. Hand wraps when it reaches head.
    """

    name = "SIEVE"

    def __init__(self, capacity: int):
        self.capacity = capacity
        # Use a doubly-linked list via dict-of-prev/next for O(1) removal
        self._next: dict[int, int | None] = {}
        self._prev: dict[int, int | None] = {}
        self._visited: dict[int, bool] = {}
        self._head: int | None = None
        self._tail: int | None = None
        self._hand: int | None = None
        self._size = 0

    def _insert_head(self, key: int) -> None:
        self._visited[key] = False
        self._prev[key] = None
        self._next[key] = self._head
        if self._head is not None:
            self._prev[self._head] = key
        else:
            self._tail = key
        self._head = key
        self._size += 1

    def _remove(self, key: int) -> None:
        p = self._prev[key]
        n = self._next[key]
        if p is not None:
            self._next[p] = n
        else:
            self._head = n
        if n is not None:
            self._prev[n] = p
        else:
            self._tail = p
        del self._prev[key]
        del self._next[key]
        del self._visited[key]
        self._size -= 1

    def _evict(self) -> None:
        if self._hand is None:
            self._hand = self._tail
        while self._hand is not None and self._visited[self._hand]:
            self._visited[self._hand] = False
            self._hand = self._prev[self._hand]
            if self._hand is None:
                self._hand = self._tail
        if self._hand is None:
            return
        victim = self._hand
        self._hand = self._prev[victim]
        self._remove(victim)

    def access(self, key: int) -> bool:
        if key in self._visited:
            self._visited[key] = True
            return True
        if self._size >= self.capacity:
            self._evict()
        self._insert_head(key)
        return False


class S3FIFO:
    """S3-FIFO (Yang et al., SOSP 2023).

    Three queues: Small (~10% of capacity), Main (~90%), Ghost (track recent
    evictions from Small, names only).
    - New key → Small FIFO with visited=0.
    - On hit: set visited=1 (in whichever queue it lives).
    - Small full → evict head: if visited, promote to Main; else move name
      to Ghost.
    - Main full → SIEVE-style eviction within Main (1-bit visited rotation).
    """

    name = "S3FIFO"

    def __init__(self, capacity: int, small_frac: float = 0.10):
        self.capacity = capacity
        self.small_cap = max(1, int(capacity * small_frac))
        self.main_cap = capacity - self.small_cap
        self.ghost_cap = capacity  # keep names of recent evictions
        # Each queue: ordered deque + presence dict[key->visited_bit]
        self.small_q: deque[int] = deque()
        self.small_set: dict[int, bool] = {}
        self.main_q: deque[int] = deque()
        self.main_set: dict[int, bool] = {}
        self.ghost_q: deque[int] = deque()
        self.ghost_set: set[int] = set()

    def _evict_main(self) -> None:
        # SIEVE-style: rotate visited=1 entries to back, evict first visited=0
        for _ in range(len(self.main_q)):
            head = self.main_q[0]
            if self.main_set[head]:
                # Clear bit, move to back
                self.main_q.popleft()
                self.main_set[head] = False
                self.main_q.append(head)
            else:
                # Evict
                self.main_q.popleft()
                del self.main_set[head]
                return

    def _evict_small(self) -> None:
        # Move head of Small: if visited, promote to Main; else add name to Ghost
        if not self.small_q:
            return
        head = self.small_q.popleft()
        visited = self.small_set.pop(head)
        if visited:
            # Promote: insert into Main
            if len(self.main_q) >= self.main_cap:
                self._evict_main()
            self.main_q.append(head)
            self.main_set[head] = False
        else:
            # Ghost
            if len(self.ghost_q) >= self.ghost_cap:
                old = self.ghost_q.popleft()
                self.ghost_set.discard(old)
            self.ghost_q.append(head)
            self.ghost_set.add(head)

    def access(self, key: int) -> bool:
        if key in self.small_set:
            self.small_set[key] = True
            return True
        if key in self.main_set:
            self.main_set[key] = True
            return True
        # Miss
        if key in self.ghost_set:
            # Promote: insert directly into Main
            self.ghost_set.discard(key)
            # (we leave it in ghost_q; harmless stale entry)
            if len(self.main_q) >= self.main_cap:
                self._evict_main()
            self.main_q.append(key)
            self.main_set[key] = False
            return False
        # Brand new: insert into Small
        if len(self.small_q) >= self.small_cap:
            self._evict_small()
        self.small_q.append(key)
        self.small_set[key] = False
        return False


class ARC:
    """ARC (Megiddo & Modha, FAST 2003) — Adaptive Replacement Cache.

    Maintains 4 lists:
      T1: pages seen once recently
      T2: pages seen 2+ times recently
      B1: ghost (recently evicted from T1)
      B2: ghost (recently evicted from T2)
    Adaptive parameter p balances |T1| vs |T2|; |T1|+|T2| <= c.
    """

    name = "ARC"

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.p = 0
        self.t1: OrderedDict[int, None] = OrderedDict()
        self.t2: OrderedDict[int, None] = OrderedDict()
        self.b1: OrderedDict[int, None] = OrderedDict()
        self.b2: OrderedDict[int, None] = OrderedDict()

    def _replace(self, key_in_b2: bool) -> None:
        c = self.capacity
        t1_len = len(self.t1)
        if t1_len > 0 and (t1_len > self.p or (key_in_b2 and t1_len == self.p)):
            # evict LRU of T1, move to B1
            k, _ = self.t1.popitem(last=False)
            self.b1[k] = None
            # Trim B1 to size c
            while len(self.b1) > c:
                self.b1.popitem(last=False)
        else:
            k, _ = self.t2.popitem(last=False)
            self.b2[k] = None
            while len(self.b2) > c:
                self.b2.popitem(last=False)

    def access(self, key: int) -> bool:
        c = self.capacity
        # Case I: hit in T1 or T2
        if key in self.t1:
            del self.t1[key]
            self.t2[key] = None
            return True
        if key in self.t2:
            self.t2.move_to_end(key)
            return True
        # Case II: miss but in B1 → grow T1
        if key in self.b1:
            delta = max(1, len(self.b2) // max(1, len(self.b1)))
            self.p = min(c, self.p + delta)
            self._replace(key_in_b2=False)
            del self.b1[key]
            self.t2[key] = None
            return False
        # Case III: miss but in B2 → grow T2
        if key in self.b2:
            delta = max(1, len(self.b1) // max(1, len(self.b2)))
            self.p = max(0, self.p - delta)
            self._replace(key_in_b2=True)
            del self.b2[key]
            self.t2[key] = None
            return False
        # Case IV: brand-new miss
        if len(self.t1) + len(self.b1) == c:
            if len(self.t1) < c:
                self.b1.popitem(last=False)
                self._replace(key_in_b2=False)
            else:
                self.t1.popitem(last=False)
        elif len(self.t1) + len(self.t2) + len(self.b1) + len(self.b2) >= c:
            total = len(self.t1) + len(self.t2) + len(self.b1) + len(self.b2)
            if total == 2 * c:
                self.b2.popitem(last=False)
            self._replace(key_in_b2=False)
        self.t1[key] = None
        return False


POLICIES = {
    "LRU": LRU,
    "FIFO": FIFO,
    "SIEVE": SIEVE,
    "S3FIFO": S3FIFO,
    "ARC": ARC,
}
