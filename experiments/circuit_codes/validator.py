"""Spread-k circuit code validator — generalizes snake-in-the-box (spread 2).

A circuit code of spread k in the hypercube Q_n is a sequence of distinct
vertices, consecutive vertices differing in exactly one coordinate, such that
for any two positions i, j the Hamming distance is at least min(d, k), where d
is the index-distance between them (cyclic distance for a coil/cycle, plain
|i-j| for a snake/open path). Equivalently:

  * LOCAL (d < k): any d <= k-1 consecutive transitions flip distinct
    coordinates -> Hamming(v_i, v_{i+d}) = d. (No transition coordinate repeats
    within a window of k.)
  * GLOBAL (d >= k): Hamming(v_i, v_j) >= k for all far-apart pairs.

Spread k=2, open path reproduces snake-in-the-box's chord-free "skin"
condition exactly (proved by test against baselines.validate_snake).

Length convention (matches the standard circuit-code / coil tables):
  * coil  (cyclic=True):  length = number of edges = number of vertices = L
  * snake (cyclic=False): length = number of edges = L - 1
The solver's claim is never trusted; callers re-validate every sequence.
"""

from __future__ import annotations


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def validate_circuit_code(
    n: int, k: int, seq: list[int], cyclic: bool = True
) -> tuple[bool, float]:
    """Re-derive spread-k validity + length from a raw vertex sequence.

    Returns (ok, length). O(L^2) — for validation, not the search inner loop.
    """
    if k < 2:
        raise ValueError("spread k must be >= 2")
    if not isinstance(seq, (list, tuple)):
        return False, 0.0
    seq = list(seq)
    L = len(seq)
    if L < (k if cyclic else 2):
        return False, 0.0
    if len(set(seq)) != L:
        return False, 0.0
    if any(not isinstance(v, int) or not (0 <= v < (1 << n)) for v in seq):
        return False, 0.0

    # consecutive unit-Hamming steps (wrap included for a coil)
    step_pairs = list(zip(seq, seq[1:]))
    if cyclic:
        step_pairs.append((seq[-1], seq[0]))
    for a, b in step_pairs:
        if hamming(a, b) != 1:
            return False, 0.0

    # spread: Hamming(v_i, v_j) >= min(index_distance, k) for all pairs
    for i in range(L):
        for j in range(i + 1, L):
            d = j - i
            if cyclic:
                d = min(d, L - d)
            need = k if d >= k else d
            if hamming(seq[i], seq[j]) < need:
                return False, 0.0

    length = float(L if cyclic else L - 1)
    return True, length
