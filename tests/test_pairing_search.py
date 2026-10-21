"""Meet-in-the-middle pairing search (experiments/snake_in_box/pairing_search.py).

The gluer is exercised on a hand-built disjoint Q8 pair (no C core needed);
the catalog miner is only checked when the core is available."""

import pytest

from experiments.snake_in_box import fastcore, pairing_search
from experiments.snake_in_box.baselines import validate_snake
from experiments.snake_in_box.pairing_search import _apply, _assemble, try_glue


def test_apply_translation_and_permutation_stay_in_q8():
    seq = [0, 1, 3, 7]
    assert _apply(seq, 8, 0, None) == seq
    assert all(0 <= v < 256 for v in _apply(seq, 8, 5, [1, 0, 2, 3, 4, 5, 6, 7]))
    # translation is an involution by the same xor
    assert _apply(_apply(seq, 8, 9, None), 8, 9, None) == seq


def test_assemble_builds_valid_q9_from_disjoint_halves():
    # two short Q8 snakes sharing endpoint label 0, otherwise disjoint labels
    a = [3, 1, 0]          # ends at 0
    b = [0, 2, 6]          # starts at 0
    top_bit = 1 << 8
    snake = _assemble(a, b, shared=0, b_join_idx=0, top_bit=top_bit)
    assert snake is not None
    ok, length = validate_snake(9, snake)
    assert ok and int(length) == len(a) + len(b) - 1
    assert (snake[len(a) - 1] ^ snake[len(a)]) == top_bit


def test_assemble_rejects_non_matching_join():
    a = [3, 1, 0]
    b = [5, 4]  # top end label 5 != shared 0 => no single matching edge
    assert _assemble(a, b, shared=0, b_join_idx=0, top_bit=1 << 8) is None


def test_try_glue_finds_a_placement_for_separable_snakes():
    import random
    a = [3, 1, 0]
    b = [3, 1, 0]  # same labels as a; a translation must separate them
    glued = try_glue(a, b, n=9, perm_samples=8, rng=random.Random(0))
    assert glued is not None
    ok, _ = validate_snake(9, glued)
    assert ok


def test_try_glue_generalizes_to_q4_from_q3():
    import random
    a = [3, 1, 0]
    b = [3, 1, 0]
    glued = try_glue(a, b, n=4, perm_samples=6, rng=random.Random(1))
    assert glued is not None
    ok, _ = validate_snake(4, glued)
    assert ok


@pytest.mark.skipif(not fastcore.CORE_AVAILABLE,
                    reason=f"no C core: {fastcore.LOAD_ERROR}")
def test_catalog_mines_distinct_long_q7_snakes():
    cat = pairing_search.mine_catalog(7, count=5, budget=2_000_000,
                                      min_len=44, seed0=1)
    assert len(cat) >= 1
    sigs = {frozenset(v ^ min(s) for v in s) for s in cat}
    assert len(sigs) == len(cat)  # all distinct under translation
    for s in cat:
        ok, length = validate_snake(7, s)
        assert ok and int(length) >= 44
