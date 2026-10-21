"""Snake canonical form (experiments/snake_in_box/canon.py)."""

from experiments.snake_in_box.canon import canonical_form


def test_translation_invariant():
    seq = [0, 1, 3, 7, 6]
    assert canonical_form(seq) == canonical_form([v ^ 9 for v in seq])


def test_permutation_invariant():
    seq = [0, 1, 3, 7, 6]  # dims 0,1,2,0
    perm = {0: 2, 1: 0, 2: 1}
    permuted = []
    for v in seq:
        w = 0
        for b in range(3):
            if (v >> b) & 1:
                w |= 1 << perm[b]
        permuted.append(w)
    assert canonical_form(seq) == canonical_form(permuted)


def test_reversal_invariant():
    seq = [0, 1, 3, 7, 6]
    assert canonical_form(seq) == canonical_form(seq[::-1])


def test_distinguishes_different_structures():
    assert canonical_form([0, 1, 3, 7, 6]) != canonical_form([0, 1, 3, 2, 6])
