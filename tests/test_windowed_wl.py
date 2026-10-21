"""Windowed Wang-Landau oracle (spec:
findings/windowed-wl-oracle-design-20260706.md).

The windowed extension bounds the WL walk to a length band [lmin, lmax]
and seeds it from a known long snake, so the tail (182-190 at d9) that
the global chain cannot reach becomes measurable. These tests protect
the pieces the trust chain depends on: occupancy rebuild from a seed,
the lmin shrink floor, the calibration-seed generator, and the stitch.
"""



import json
import math
import os
import random
from pathlib import Path

import pytest

from experiments.snake_in_box.wang_landau import WLState, pivot
from experiments.snake_in_box.windowed_wl import (
    calibrate_window,
    find_snake,
    overlap_residual,
    run_window_checkpointed,
    run_windowed_campaign,
    stitch,
    tail_verdict,
)


def _grow_snake(n: int, length: int) -> list[int]:
    """A valid snake (induced path) of the requested length in Q_n,
    built greedily from vertex 0. Test fixture only."""
    size = 1 << n
    on = bytearray(size)
    nc = bytearray(size)
    path = [0]
    on[0] = 1
    for b in range(n):
        nc[1 << b] += 1
    while len(path) - 1 < length:
        e = path[-1]
        grew = False
        for b in range(n):
            w = e ^ (1 << b)
            if not on[w] and nc[w] == 1:
                path.append(w)
                on[w] = 1
                for c in range(n):
                    nc[w ^ (1 << c)] += 1
                grew = True
                break
        if not grew:
            raise RuntimeError(f"stuck at length {len(path) - 1} < {length}")
    return path


def _nc_from_set(n: int, occupied: set[int]) -> list[int]:
    """Independent recompute of the occupied-neighbor count array."""
    size = 1 << n
    nc = [0] * size
    for w in range(size):
        for b in range(n):
            if (w ^ (1 << b)) in occupied:
                nc[w] += 1
    return nc


def test_seed_from_rebuilds_occupancy():
    n, length = 5, 10
    snake = _grow_snake(n, length)

    st = WLState(n, lmax=length, seed=0)
    st.seed_from(snake)

    assert st.path == snake
    assert len(st.path) - 1 == length
    occupied = set(snake)
    assert [i for i, v in enumerate(st.on) if v] == sorted(occupied)
    assert list(st.nc) == _nc_from_set(n, occupied)


def test_lmin_clamps_shrink_floor():
    n, lmin, lmax = 5, 6, 12
    snake = _grow_snake(n, lmax)

    st = WLState(n, lmax=lmax, seed=1, lmin=lmin)
    st.seed_from(snake)
    st.run(200_000)

    # the walk never descends below the window floor...
    assert all(st.h[length] == 0 for length in range(lmin))
    # ...and it actually explores the band (does not freeze at lmax).
    assert any(st.h[length] > 0 for length in range(lmin, lmax + 1))


def test_lmin_survives_dump_load(tmp_path):
    n, lmin, lmax = 5, 6, 12
    snake = _grow_snake(n, lmax)

    st = WLState(n, lmax=lmax, seed=2, lmin=lmin)
    st.seed_from(snake)
    st.run(5_000)
    fp = tmp_path / "ck.json"
    st.dump(fp)

    st2 = WLState.load(fp)
    assert st2.lmin == lmin
    # a resumed windowed run still respects the floor.
    st2.run(50_000)
    assert all(st2.h[length] == 0 for length in range(lmin))


def test_p_pivot_and_moves_survive_dump_load(tmp_path):
    st = WLState(6, lmax=12, seed=2, lmin=6, p_pivot=0.3)
    st.seed_from(_grow_snake(6, 12))
    st.run(2000)
    assert st.moves == 2000  # run() tracks moves so a resume knows its progress
    fp = tmp_path / "ck.json"
    st.dump(fp)

    st2 = WLState.load(fp)
    assert st2.p_pivot == 0.3  # pivots survive resume (no silent re-biasing)
    assert st2.moves == 2000


def test_interrupted_dump_preserves_previous_checkpoint(tmp_path, monkeypatch):
    # A kill landing mid-write must not destroy the only checkpoint: the
    # previous dump has to survive an interrupted one intact.
    st = WLState(5, lmax=8, seed=3)
    st.run(1000)
    fp = tmp_path / "ck.json"
    st.dump(fp)
    good_moves = WLState.load(fp).moves

    st.run(1000)
    orig_write = Path.write_text

    def dying_write(self, text, *args, **kwargs):
        orig_write(self, text[: len(text) // 2], *args, **kwargs)
        raise OSError("killed mid-write")

    monkeypatch.setattr(Path, "write_text", dying_write)
    with pytest.raises(OSError):
        st.dump(fp)
    monkeypatch.undo()

    st2 = WLState.load(fp)
    assert st2.moves == good_moves


def _is_snake(n: int, path: list[int]) -> bool:
    """Validate an induced path (snake): consecutive vertices differ by one
    bit, all distinct, and no chords (each vertex's occupied neighbours are
    exactly its path neighbours)."""
    s = set(path)
    if len(s) != len(path):
        return False
    for i in range(len(path) - 1):
        if bin(path[i] ^ path[i + 1]).count("1") != 1:
            return False
    for i, v in enumerate(path):
        nbrs = {v ^ (1 << b) for b in range(n)}
        occ = nbrs & s
        expected = set()
        if i > 0:
            expected.add(path[i - 1])
        if i < len(path) - 1:
            expected.add(path[i + 1])
        if occ != expected:
            return False
    return True


def test_find_snake_returns_valid_snake_of_length():
    snake = find_snake(6, 26)
    assert len(snake) - 1 == 26
    assert _is_snake(6, snake)


def test_windowed_entropy_returns_raw_band():
    n, lmin, lmax = 5, 6, 12
    snake = _grow_snake(n, lmax)
    st = WLState(n, lmax=lmax, seed=3, lmin=lmin)
    st.seed_from(snake)
    st.run(50_000)

    s = st.entropy()
    # in-band bins only (no spurious N(0) anchor bin)...
    assert set(s) <= set(range(lmin, lmax + 1))
    assert lmax in s
    # ...and values are the raw g (offset is deferred to the stitch).
    assert all(s[length] == st.g[length] for length in s)


def test_stitch_recovers_known_offset():
    # An anchored reference curve S(L) = 2L, and a window measuring the same
    # curve but shifted by an unknown additive offset (+5.0), extending the
    # tail to L=15. stitch must recover the offset and align the extension.
    ref = {length: 2.0 * length for length in range(0, 11)}
    window = {length: 2.0 * length + 5.0 for length in range(8, 16)}

    merged = stitch(ref, [window])

    assert set(merged) == set(range(0, 16))
    for length in range(0, 16):
        assert abs(merged[length] - 2.0 * length) < 1e-9


def test_stitch_chains_two_windows_out_of_order():
    # ref covers [0,5]; w1 overlaps it on [4,5]; w2 overlaps w1 on [9,10].
    # Passed out of order, stitch must chain w1 then w2 back to the anchor.
    ref = {length: -0.5 * length for length in range(0, 6)}
    w1 = {length: -0.5 * length + 3.0 for length in range(4, 11)}
    w2 = {length: -0.5 * length - 2.0 for length in range(9, 16)}

    merged = stitch(ref, [w2, w1])

    assert set(merged) == set(range(0, 16))
    for length in range(0, 16):
        assert abs(merged[length] - (-0.5 * length)) < 1e-9


def test_stitch_raises_on_gap_in_ladder():
    ref = {length: float(length) for length in range(0, 5)}
    window = {length: float(length) for length in range(8, 12)}  # gap [5,7]

    with pytest.raises(ValueError):
        stitch(ref, [window])


def test_pivot_preserves_length_and_validity():
    n = 6
    snake = _grow_snake(n, 12)

    # identity permutation is a no-op: same-length valid snake.
    out = pivot(n, snake, 5, list(range(n)))
    assert out is not None
    assert len(out) == len(snake)
    assert _is_snake(n, out)

    # a non-trivial legal pivot yields a *different* valid snake, same length.
    rng = random.Random(0)
    found = None
    for _ in range(300):
        sigma = list(range(n))
        rng.shuffle(sigma)
        cand = pivot(n, snake, 5, sigma)
        if cand is not None and cand != snake:
            found = cand
            break
    assert found is not None
    assert len(found) == len(snake)
    assert _is_snake(n, found)


def test_pivot_move_preserves_length_and_occupancy():
    n, length = 6, 12
    st = WLState(n, lmax=length, seed=1, p_pivot=1.0)
    st.seed_from(_grow_snake(n, length))
    st.run(50_000)

    # p_pivot=1.0: every move is a fixed-length pivot -> length never changes,
    # only that length bin is visited.
    assert len(st.path) - 1 == length
    assert all(st.h[length_] == 0 for length_ in range(length))
    assert st.h[length] > 0
    # the incremental occupancy update keeps on/nc consistent with the pivoted
    # path, and the path stays a valid snake.
    occ = set(st.path)
    assert len(occ) == len(st.path)
    assert [i for i, v in enumerate(st.on) if v] == sorted(occ)
    assert list(st.nc) == _nc_from_set(n, occ)
    assert _is_snake(n, st.path)


def test_overlap_residual_measures_shape_mismatch():
    # identical shape shifted by a constant -> zero residual after offset fit.
    a = {L: 2.0 * L for L in range(10, 21)}
    b = {L: 2.0 * L + 4.0 for L in range(15, 26)}
    assert overlap_residual(a, b) < 1e-9
    # a real shape mismatch on the overlap -> nonzero residual.
    c = {L: 2.0 * L + 0.5 * (L - 15) ** 2 for L in range(15, 26)}
    assert overlap_residual(a, c) > 0.1


def test_tail_verdict_fat_vs_collapse():
    # fat tail: S still large near the record (extrapolates high) -> keep hunting.
    fat = {L: 30.0 - 1.0 * (L - 180) for L in range(180, 190)}
    v = tail_verdict(fat, our_best=185, record_len=190)
    assert v["classification"] == "fat_tail"
    assert v["S_at_best"] == fat[185]
    # collapse: steep cliff, few states extrapolated at the record -> census.
    coll = {L: 40.0 - 4.0 * (L - 180) for L in range(180, 190)}
    v2 = tail_verdict(coll, our_best=185, record_len=190)
    assert v2["classification"] == "collapse"
    assert v2["S_at_record"] < math.log(1000)


def test_run_window_checkpointed_resumes(tmp_path):
    ckpt = str(tmp_path / "w.json")
    seed_snake = find_snake(6, 13)
    # first call caps at 1M moves, checkpoints, and returns.
    st1 = run_window_checkpointed(
        6, 8, 13, seed_snake, ckpt, p_pivot=0.2,
        move_cap=1_000_000, f_stop=1e-12, moves_per_ckpt=500_000,
    )
    assert st1.moves >= 1_000_000
    assert os.path.exists(ckpt)
    # second call resumes from the checkpoint (does not restart from 0).
    st2 = run_window_checkpointed(
        6, 8, 13, seed_snake, ckpt, p_pivot=0.2,
        move_cap=2_000_000, f_stop=1e-12, moves_per_ckpt=500_000,
    )
    assert st2.moves >= 2_000_000
    assert st2.p_pivot == 0.2  # pivots survived the resume
    # decisive resume proof: tamper the checkpoint's move counter to a value
    # above the next cap. A true resume returns immediately with the tampered
    # counter; a silent restart-from-scratch would run fresh moves instead.
    d = json.loads(Path(ckpt).read_text())
    d["moves"] = 7_777_777
    Path(ckpt).write_text(json.dumps(d))
    st3 = run_window_checkpointed(
        6, 8, 13, seed_snake, ckpt, p_pivot=0.2,
        move_cap=2_000_000, f_stop=1e-12, moves_per_ckpt=500_000,
    )
    assert st3.moves == 7_777_777


def test_resume_rejects_mismatched_window(tmp_path):
    # A stale checkpoint for a different window must not silently continue:
    # the args and the loaded state have to agree on (n, lmin, lmax).
    ckpt = str(tmp_path / "w.json")
    run_window_checkpointed(
        6, 8, 13, find_snake(6, 13), ckpt,
        move_cap=500_000, f_stop=1e-12, moves_per_ckpt=500_000,
    )
    with pytest.raises(ValueError, match="refusing to resume"):
        run_window_checkpointed(
            6, 9, 13, find_snake(6, 13), ckpt,
            move_cap=1_000_000, f_stop=1e-12, moves_per_ckpt=500_000,
        )


def test_resume_keeps_checkpoint_p_pivot_and_warns(tmp_path, capsys):
    # The checkpoint's p_pivot wins on resume (documented behaviour), but a
    # mismatched request must be called out, not silently ignored.
    ckpt = str(tmp_path / "w.json")
    run_window_checkpointed(
        6, 8, 13, find_snake(6, 13), ckpt, p_pivot=0.2,
        move_cap=500_000, f_stop=1e-12, moves_per_ckpt=500_000,
    )
    st = run_window_checkpointed(
        6, 8, 13, find_snake(6, 13), ckpt, p_pivot=0.5,
        move_cap=1_000_000, f_stop=1e-12, moves_per_ckpt=500_000,
    )
    assert st.p_pivot == 0.2
    assert "p_pivot" in capsys.readouterr().out


def test_campaign_uses_distinct_window_seeds(tmp_path, monkeypatch):
    # Each window walker should get its own RNG seed rather than sharing the
    # campaign seed. Stub the window runner to capture the seeds it is given.
    import experiments.snake_in_box.windowed_wl as wwl

    seeds = []

    class FakeState:
        def __init__(self, lo, hi):
            self.lo, self.hi = lo, hi

        def entropy(self):
            return {L: -1.0 * L for L in range(self.lo, self.hi + 1)}

    def fake_run(n, lmin, lmax, seed_snake, ckpt_path, **kwargs):
        seeds.append(kwargs["seed"])
        return FakeState(lmin, lmax)

    monkeypatch.setattr(wwl, "run_window_checkpointed", fake_run)
    reference = {L: -1.0 * L for L in range(0, 6)}
    windows = [
        {"name": "lo", "lmin": 4, "lmax": 8, "seed_snake": []},
        {"name": "hi", "lmin": 7, "lmax": 10, "seed_snake": []},
    ]
    wwl.run_windowed_campaign(
        5, windows, reference, str(tmp_path), our_best=8, record_len=10, seed=7,
    )
    assert len(seeds) == 2
    assert len(set(seeds)) == 2


def test_default_exact_path_tracks_n():
    # CLI truth anchor must follow --n: the Q6 file exists in the repo, no Q4
    # file does (fall back to recomputing rather than anchoring to wrong truth).
    from experiments.snake_in_box.windowed_wl import _default_exact_path

    assert _default_exact_path(6).endswith("wl_q6_exact_counts.json")
    assert _default_exact_path(4) == ""


@pytest.mark.slow
def test_windowed_campaign_recovers_q5_tail(tmp_path):
    from experiments.snake_in_box.wang_landau import exact_counts

    exact = exact_counts(5)
    s_exact = {L: math.log(v) for L, v in exact.items()}
    reference = {L: s_exact[L] for L in range(8, 11)}  # trusted low anchor
    windows = [
        {"name": "lo", "lmin": 8, "lmax": 12, "seed_snake": find_snake(5, 12)},
        {"name": "hi", "lmin": 11, "lmax": 13, "seed_snake": find_snake(5, 13)},
    ]

    out = run_windowed_campaign(
        5, windows, reference, str(tmp_path), our_best=12, record_len=13,
        p_pivot=0.2, f_stop=1e-6, move_cap=25_000_000,
    )

    # the assembled tail matches exact truth, incl the near-max bins.
    for L in range(11, 14):
        assert abs(out["S"][L] - s_exact[L]) < 0.15, (L, out["S"][L], s_exact[L])
    # the re-measured low window agrees with the reference in shape.
    assert out["anchor_residual"] < 0.2


@pytest.mark.slow
def test_q5_window_calibration_with_pivot():
    # Truth-anchored gate: a clamped high window [10,13] above the Q5 peak is
    # biased without pivots (+0.70); with pivots it must recover the exact tail
    # (incl the near-max bins) to sampling accuracy.
    from experiments.snake_in_box.wang_landau import exact_counts

    exact = exact_counts(5)
    maxerr, errs = calibrate_window(
        5, 10, 13, exact, p_pivot=0.2, budget_moves=20_000_000
    )
    assert maxerr < 0.1, errs
