"""Canonical form for snakes under the hyperoctahedral group + reversal.

Two snakes are isomorphic iff their transition sequences agree after
relabeling dimensions; the canonical relabeling maps each dimension to its
order of first appearance. Taking the lexicographic min over the sequence
and its reversal canonicalizes under coordinate permutation, XOR
translation (transition sequences are translation-invariant by
construction), and path reversal. O(L) — cheap enough to key every ledger
entry and dedup pools before paying for exact scoring."""

from __future__ import annotations


def transitions(seq: list[int]) -> list[int]:
    return [(seq[i] ^ seq[i + 1]).bit_length() - 1 for i in range(len(seq) - 1)]


def _first_use_relabel(trans: list[int]) -> tuple[int, ...]:
    label: dict[int, int] = {}
    out = []
    for t in trans:
        if t not in label:
            label[t] = len(label)
        out.append(label[t])
    return tuple(out)


def canonical_form(seq: list[int]) -> str:
    """Canonical key: min over orientation of the first-use-relabeled
    transition sequence, rendered compactly."""
    t = transitions(seq)
    a = _first_use_relabel(t)
    b = _first_use_relabel(t[::-1])
    best = min(a, b)
    return ",".join(map(str, best))
