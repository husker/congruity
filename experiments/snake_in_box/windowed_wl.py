"""Windowed Wang-Landau orchestration for the d9 snake density oracle.

Spec: findings/windowed-wl-oracle-design-20260706.md

The validated flat-histogram sampler lives in wang_landau.py (WLState, now
with an lmin window floor and seed_from). This module drives it across the
tail: it supplies calibration seeds (find_snake), stitches overlapping window
g-curves into a single S(L) (stitch), and runs the Q5/Q6 calibration gate.
"""

from __future__ import annotations

import math
import os
from pathlib import Path

from experiments.snake_in_box.wang_landau import WLState




def find_snake(n: int, target_len: int, start: int = 0) -> list[int]:
    """Return the first snake (induced path) of length ``target_len`` found
    by depth-first descent from ``start`` in Q_n. Used to seed calibration
    windows (Q6 L=26, Q5 L=13) — a concrete long snake to start the walk
    inside the tail. Raises if no such snake exists from ``start``."""
    size = 1 << n
    on = bytearray(size)
    nc = bytearray(size)
    path = [start]
    on[start] = 1
    for b in range(n):
        nc[start ^ (1 << b)] += 1

    def dfs() -> bool:
        if len(path) - 1 == target_len:
            return True
        e = path[-1]
        for b in range(n):
            w = e ^ (1 << b)
            if not on[w] and nc[w] == 1:
                path.append(w)
                on[w] = 1
                for c in range(n):
                    nc[w ^ (1 << c)] += 1
                if dfs():
                    return True
                for c in range(n):
                    nc[w ^ (1 << c)] -= 1
                on[w] = 0
                path.pop()
        return False

    if not dfs():
        raise RuntimeError(
            f"no snake of length {target_len} in Q{n} from vertex {start}"
        )
    return list(path)


def stitch(
    reference: dict[int, float],
    windows: list[dict[int, float]],
) -> dict[int, float]:
    """Combine overlapping window log-density curves into a single anchored
    S(L).

    ``reference`` is an anchored curve (the global WL: L -> log N(L)); each
    entry of ``windows`` is a raw, unanchored window curve (L -> g[L]). Each
    window is aligned to the already-anchored region by the least-squares
    additive offset over their shared bins (for a pure additive shift this is
    the mean bin-wise difference), chained outward from the reference.
    Overlapping bins are averaged. A window that shares no bin with the
    anchored region (a gap in the ladder) is an error.

    Returns the merged L -> S(L).
    """
    merged = dict(reference)
    remaining = list(windows)
    while remaining:
        remaining.sort(key=min)
        for i, w in enumerate(remaining):
            overlap = set(w) & set(merged)
            if overlap:
                offset = sum(merged[L] - w[L] for L in overlap) / len(overlap)
                for L, g in w.items():
                    aligned = g + offset
                    merged[L] = (
                        0.5 * (merged[L] + aligned) if L in merged else aligned
                    )
                remaining.pop(i)
                break
        else:
            raise ValueError(
                "no window overlaps the anchored region; cannot stitch "
                "(gap in the window ladder)"
            )
    return merged


def calibrate_window(
    n: int,
    lmin: int,
    lmax: int,
    exact: dict[int, int],
    p_pivot: float = 0.3,
    flat: float = 0.85,
    f_stop: float = 1e-7,
    budget_moves: int = 50_000_000,
    seed: int = 7,
    chunk: int = 500_000,
) -> tuple[float, dict[int, float]]:
    """Run pivot-augmented windowed WL over ``[lmin, lmax]`` (seeded at lmax),
    stitch its raw g to the exact log-density at the low edge, and return the
    tail error against ``exact`` counts. This is the truth-anchored calibration
    gate and the same harness the d9 campaign uses, run here against known
    Q5/Q6 counts. All-bins flatness (every band bin must be visited before f
    halves) prevents a starved top bin from letting f run away.

    Returns ``(max |error| over [lmin+2, lmax], per-bin error dict)``.
    """
    S = {L: math.log(v) for L, v in exact.items()}
    st = WLState(n, lmax=lmax, lmin=lmin, seed=seed, p_pivot=p_pivot)
    st.seed_from(find_snake(n, lmax))
    total = 0
    while st.f > f_stop and total < budget_moves:
        st.run(chunk)
        total += chunk
        hs = [st.h[L] for L in range(lmin, lmax + 1)]
        if min(hs) > 0 and min(hs) >= flat * (sum(hs) / len(hs)):
            st.f *= 0.5
            st.h = [0] * len(st.h)
    g = st.entropy()
    merged = stitch({L: S[L] for L in range(lmin, lmin + 2)}, [g])
    errs = {L: merged[L] - S[L] for L in range(lmin, lmax + 1) if L in merged}
    maxerr = max(abs(errs[L]) for L in range(lmin + 2, lmax + 1))
    return maxerr, errs


def overlap_residual(a: dict[int, float], b: dict[int, float]) -> float:
    """RMS mismatch between two log-density curves over their shared bins after
    the best additive-offset alignment. Zero for curves that differ only by a
    constant. This is the anchor cross-check: a freshly re-measured low window
    and the trusted reference should agree in *shape* over their overlap; a
    large residual means the stitch seam is untrustworthy."""
    shared = sorted(set(a) & set(b))
    if not shared:
        raise ValueError("no overlap to compare")
    offset = sum(a[L] - b[L] for L in shared) / len(shared)
    return (sum((a[L] - (b[L] + offset)) ** 2 for L in shared)
            / len(shared)) ** 0.5


def tail_slope(S: dict[int, float], lengths: list[int]) -> float:
    """Least-squares slope (nats per unit length) of S over ``lengths``."""
    xs = list(lengths)
    ys = [S[L] for L in xs]
    m = len(xs)
    sx, sy = sum(xs), sum(ys)
    sxx = sum(x * x for x in xs)
    sxy = sum(x * y for x, y in zip(xs, ys))
    return (m * sxy - sx * sy) / (m * sxx - sx * sx)


def tail_verdict(
    S: dict[int, float],
    our_best: int,
    record_len: int,
    collapse_threshold: float | None = None,
    fit_bins: int = 4,
) -> dict:
    """Classify the record program from the measured tail S(L)=log N(L).

    ``fat_tail`` — many states remain above our best; the 181-shelf is a
    *search* failure, keep hunting. ``collapse`` — few states extrapolate at the
    record; it is a near-wall, pivot to census / upper bounds. The threshold on
    the (measured or slope-extrapolated) log-count at the record defaults to
    log(1000).
    """
    if collapse_threshold is None:
        collapse_threshold = math.log(1000)
    measured = sorted(L for L in S if L <= record_len)
    slope = tail_slope(S, measured[-fit_bins:])
    top = measured[-1]
    s_record = (S[record_len] if record_len in S
                else S[top] + slope * (record_len - top))
    return {
        "slope": slope,
        "S_at_best": S[our_best],
        "S_at_record": s_record,
        "classification": ("collapse" if s_record <= collapse_threshold
                           else "fat_tail"),
    }


def run_window_checkpointed(
    n: int,
    lmin: int,
    lmax: int,
    seed_snake: list[int],
    ckpt_path: str,
    p_pivot: float = 0.3,
    flat: float = 0.85,
    f_stop: float = 1e-7,
    move_cap: int = 4_000_000_000,
    chunk: int = 500_000,
    moves_per_ckpt: int = 5_000_000,
    seed: int = 7,
    on_progress=None,
) -> WLState:
    """Resumable single-window WL run (pivot-augmented, all-bins flatness).

    If ``ckpt_path`` exists, resume from it; otherwise start seeded at lmax.
    Checkpoints every ``moves_per_ckpt`` moves — kill-safe: after the machine
    kills the job, re-invoking with the same ``ckpt_path`` continues from the
    last dump (state, RNG, f, and p_pivot all round-trip). Runs until the
    modification factor reaches ``f_stop`` or ``move_cap`` moves. Returns the
    WLState.
    """
    if os.path.exists(ckpt_path):
        st = WLState.load(Path(ckpt_path))
        if (st.n, st.lmin, st.lmax) != (n, lmin, lmax):
            raise ValueError(
                f"checkpoint {ckpt_path} is for Q{st.n} [{st.lmin},{st.lmax}],"
                f" not Q{n} [{lmin},{lmax}] — refusing to resume a different"
                " window (move or delete the checkpoint)"
            )
        if st.p_pivot != p_pivot:
            print(f"warning: checkpoint p_pivot={st.p_pivot} wins over"
                  f" requested p_pivot={p_pivot} (resume keeps its pivots)",
                  flush=True)
    else:
        st = WLState(n, lmax=lmax, lmin=lmin, seed=seed, p_pivot=p_pivot)
        st.seed_from(seed_snake)
    since = 0
    while st.f > f_stop and st.moves < move_cap:
        st.run(chunk)
        since += chunk
        hs = [st.h[L] for L in range(lmin, lmax + 1)]
        if min(hs) > 0 and min(hs) >= flat * (sum(hs) / len(hs)):
            st.f *= 0.5
            st.h = [0] * len(st.h)
        if since >= moves_per_ckpt:
            st.dump(Path(ckpt_path))
            since = 0
            if on_progress is not None:
                on_progress(st)
    st.dump(Path(ckpt_path))
    return st


def run_windowed_campaign(
    n: int,
    windows: list[dict],
    reference: dict[int, float],
    ckpt_dir: str,
    our_best: int,
    record_len: int,
    p_pivot: float = 0.3,
    f_stop: float = 1e-7,
    move_cap: int = 4_000_000_000,
    seed: int = 7,
    on_progress=None,
) -> dict:
    """Run a ladder of pivot-augmented windows (each resumable/checkpointed),
    stitch them to ``reference`` (an anchored S(L) — e.g. the trusted global g),
    and return the assembled S(L), the anchor cross-check residual, and the tail
    verdict.

    ``windows`` is a list of ``{name, lmin, lmax, seed_snake}`` ordered by lmin;
    the lowest window must overlap ``reference``. Per the campaign design the
    lowest window is *re-measured* (pivot-augmented) rather than trusting the old
    global g at the seam, and ``overlap_residual`` reports whether the fresh
    measurement agrees with the reference in shape before the chain is trusted.
    """
    os.makedirs(ckpt_dir, exist_ok=True)
    ordered = sorted(windows, key=lambda w: w["lmin"])
    measured = []
    for i, w in enumerate(ordered):
        st = run_window_checkpointed(
            n, w["lmin"], w["lmax"], w["seed_snake"],
            os.path.join(ckpt_dir, w["name"] + ".json"),
            p_pivot=p_pivot, f_stop=f_stop, move_cap=move_cap, seed=seed + i,
            on_progress=on_progress,
        )
        measured.append(st.entropy())
    anchor_residual = overlap_residual(reference, measured[0])
    S = stitch(reference, measured)
    verdict = tail_verdict(S, our_best, record_len)
    return {
        "S": S,
        "anchor_residual": anchor_residual,
        "verdict": verdict,
        "windows": measured,
    }


def _default_exact_path(n: int) -> str:
    """Exact-counts JSON for Q_n if the repo has one, else "" (recompute).
    Keeps the CLI truth anchor tied to --n instead of hardwiring the Q6 file."""
    p = Path(f"data/snake_in_box/wl_q{n}_exact_counts.json")
    return str(p) if p.exists() else ""


def _confirm(n, lmin, lmax, ckpt, p_pivot, move_cap, f_stop, seed, exact_path):
    """Checkpointed single-window confirmation against exact truth (n<=6):
    run [lmin,lmax] pivot-augmented, anchor to exact S at the low edge, and
    report the tail error. Resumable — re-invoke to continue from ckpt."""
    import json

    if exact_path:
        exact = {int(k): v for k, v in
                 json.loads(Path(exact_path).read_text()).items()}
    else:
        from experiments.snake_in_box.wang_landau import exact_counts
        exact = exact_counts(n)
    s_exact = {L: math.log(v) for L, v in exact.items()}
    anchor = {L: s_exact[L] for L in range(lmin, lmin + 2)}

    def tail_err(st):
        merged = stitch(anchor, [st.entropy()])
        bins = [L for L in range(lmin + 2, lmax + 1) if L in merged]
        return max(abs(merged[L] - s_exact[L]) for L in bins) if bins else float("nan")

    def progress(st):
        print(f"  moves={st.moves / 1e6:.0f}M f={st.f:.0e} "
              f"tail_err={tail_err(st):.3f}", flush=True)

    st = run_window_checkpointed(
        n, lmin, lmax, find_snake(n, lmax), ckpt, p_pivot=p_pivot,
        f_stop=f_stop, move_cap=move_cap, seed=seed, on_progress=progress,
    )
    merged = stitch(anchor, [st.entropy()])
    print(f"\nQ{n} [{lmin},{lmax}] p_pivot={p_pivot} f={st.f:.0e} "
          f"moves={st.moves / 1e6:.0f}M")
    for L in range(lmin, lmax + 1):
        if L in merged:
            print(f"  L={L:3d}  exact={s_exact[L]:7.3f}  wl={merged[L]:7.3f}  "
                  f"err={merged[L] - s_exact[L]:+.3f}")
    me = tail_err(st)
    print(f"  max tail err = {me:.3f}  ({'PASS' if me < 0.3 else 'FAIL'} @ 0.3 nats)")


def main() -> None:
    """CLI: kill-safe windowed-WL confirmation. Defaults to the Q6 [21,26]
    scaling check (does the pivot fix hold at a ~10.5-nat range?). Launch in the
    background; re-invoke with the same --ckpt to resume after a kill.

        uv run python -m experiments.snake_in_box.windowed_wl --p-pivot 0.3
    """
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--lmin", type=int, default=21)
    ap.add_argument("--lmax", type=int, default=26)
    ap.add_argument("--ckpt", default="data/snake_in_box/wl_confirm_q6_21_26.json")
    ap.add_argument("--p-pivot", type=float, default=0.3)
    ap.add_argument("--move-cap", type=int, default=2_000_000_000)
    ap.add_argument("--f-stop", type=float, default=1e-6)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--exact", default=None,
                    help="exact-counts JSON (default: the repo's "
                         "wl_q<n>_exact_counts.json if present, else recompute)")
    a = ap.parse_args()
    exact = a.exact if a.exact is not None else _default_exact_path(a.n)
    _confirm(a.n, a.lmin, a.lmax, a.ckpt, a.p_pivot, a.move_cap, a.f_stop,
             a.seed, exact)


if __name__ == "__main__":
    main()
