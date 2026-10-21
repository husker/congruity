"""Wang-Landau oracle (experiments/snake_in_box/wang_landau.py).

Trust chain: the exact enumerator is verified against an independent
brute-force count at Q3; the sampler is then calibrated against the
enumerator at Q4. If S(L) from a short WL run tracks exact log-counts
within tolerance, the d9 production runs mean something.
"""

import itertools
import math

from experiments.snake_in_box.wang_landau import (WLState, exact_counts)


def _brute_q3() -> dict[int, int]:
    """Directed induced paths in Q3 by exhaustive sequence check."""
    def adj(a, b):
        return bin(a ^ b).count("1") == 1

    counts = {0: 8}
    for length in range(1, 5):
        c = 0
        for seq in itertools.permutations(range(8), length + 1):
            if any(not adj(seq[i], seq[i + 1]) for i in range(length)):
                continue
            ok = True
            for i in range(length + 1):
                for j in range(i + 2, length + 1):
                    if adj(seq[i], seq[j]):
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                c += 1
        if c:
            counts[length] = c
    return counts


def test_exact_counts_match_brute_force_q3():
    assert exact_counts(3) == _brute_q3()


def test_wl_calibrates_at_q4():
    exact = exact_counts(4)
    st = WLState(4, max(exact), seed=7)
    while st.f > 3e-4:
        st.run(100_000)
        if st.flat_enough(0.8):
            st.f /= 2.0
            st.h = [0] * (st.lmax + 1)
    s = st.entropy()
    for L, cnt in exact.items():
        assert abs(s[L] - math.log(cnt)) < 0.35, (
            f"L={L}: WL {s[L]:.2f} vs exact {math.log(cnt):.2f}")
    assert st.best_seen == max(exact)   # WL walk finds the Q4 optimum
