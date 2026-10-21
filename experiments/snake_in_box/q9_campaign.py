"""Q9 windowed-WL DOS campaign — the open-record run.

Ladder (design: findings/windowed-wl-calibration-tension-20260706.md; gate
cleared by the Q6 [21,26] confirmation, 0.234 nats @ 0.3):

  seam_115_145  seeded from the 152 certificate (truncated to L=145)
  mid_120_160   seeded from the 181 certificate  (truncated to L=160)
  top_150_190   seeded from the 185 certificate  (starts at L=185)

The global Q9 WL state (wl_q9_checkpoint.json: 3.4B moves, best_seen=139) is
trusted only below the seam; its entropy restricted to L<=REF_MAX is the
stitching reference. The campaign re-measures the seam window and reports
overlap_residual against that reference instead of trusting the old global g
at the seam. Verdict: tail_verdict(S, our_best=185, record_len=190) —
fat_tail => keep hunting, collapse => census/upper bounds.

Run modes (all kill-safe: re-invoking resumes from the per-window
checkpoints under data/snake_in_box/q9_campaign/):

  full campaign, windows sequential::

      uv run python -m experiments.snake_in_box.q9_campaign

  one window per process — the sampler is a sequential pure-Python chain
  (a GPU does not help; three cores do)::

      uv run python -m experiments.snake_in_box.q9_campaign --window seam_115_145
      uv run python -m experiments.snake_in_box.q9_campaign --window mid_120_160
      uv run python -m experiments.snake_in_box.q9_campaign --window top_150_190

  stitch + provisional verdict from whatever the checkpoints hold
  (read-only, safe while windows are running)::

      uv run python -m experiments.snake_in_box.q9_campaign --report
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from experiments.snake_in_box.wang_landau import WLState
from experiments.snake_in_box.windowed_wl import (
    overlap_residual,
    run_window_checkpointed,
    run_windowed_campaign,
    stitch,
    tail_verdict,
)

OUR_BEST = 185
RECORD_LEN = 190
REF_MAX = 130
GLOBAL_CKPT = "data/snake_in_box/wl_q9_checkpoint.json"
CKPT_DIR = "data/snake_in_box/q9_campaign"


def load_certificate(fp: str | Path) -> list[int]:
    """Vertex path from a snake certificate JSON (``seq`` or ``path`` key)."""
    d = json.loads(Path(fp).read_text())
    seq = d["seq"] if "seq" in d else d["path"]
    if len(seq) != d["length"] + 1:
        raise ValueError(
            f"{fp}: {len(seq)} vertices for claimed length {d['length']}")
    return list(seq)


def build_windows() -> list[dict]:
    base = "data/snake_in_box"
    return [
        {"name": "seam_115_145", "lmin": 115, "lmax": 145,
         "seed_snake": load_certificate(
             f"{base}/certificates/q9_len152.json")},
        {"name": "mid_120_160", "lmin": 120, "lmax": 160,
         "seed_snake": load_certificate(f"{base}/seed6003_q9_181.json")},
        {"name": "top_150_190", "lmin": 150, "lmax": 190,
         "seed_snake": load_certificate(
             f"{base}/certificates/q9_len185_joint.json")},
    ]


def build_reference(ckpt_path: str = GLOBAL_CKPT,
                    ref_max: int = REF_MAX) -> dict[int, float]:
    """Global-run S(L) restricted to the trusted region below the seam."""
    st = WLState.load(Path(ckpt_path))
    return {L: v for L, v in st.entropy().items() if L <= ref_max}


def _progress(st: WLState) -> None:
    print(f"  [{st.lmin},{st.lmax}] moves={st.moves / 1e6:.0f}M "
          f"f={st.f:.1e} best_seen={st.best_seen}", flush=True)


def _report(windows: list[dict], reference: dict[int, float],
            ckpt_dir: str, f_stop: float) -> int:
    """Stitch + verdict from the checkpoints as they stand. Read-only."""
    measured = []
    for w in sorted(windows, key=lambda w: w["lmin"]):
        fp = Path(ckpt_dir) / (w["name"] + ".json")
        if not fp.exists():
            print(f"{w['name']}: no checkpoint yet — nothing to stitch",
                  flush=True)
            return 1
        st = WLState.load(fp)
        state = "converged" if st.f <= f_stop else "annealing"
        print(f"{w['name']}: moves={st.moves / 1e6:.0f}M f={st.f:.1e} "
              f"({state})", flush=True)
        measured.append(st.entropy())
    anchor = overlap_residual(reference, measured[0])
    S = stitch(reference, measured)
    verdict = tail_verdict(S, our_best=OUR_BEST, record_len=RECORD_LEN)
    print(f"anchor residual vs trusted global g: {anchor:.3f} nats (rms)")
    tail = {L: round(v, 2) for L, v in sorted(S.items()) if L >= 170}
    print(f"S(L) 170..: {tail}")
    print(f"verdict: {verdict}", flush=True)
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--window", default=None,
                    help="run a single window by name (one process per "
                         "window parallelises the campaign)")
    ap.add_argument("--report", action="store_true",
                    help="stitch + verdict from current checkpoints, "
                         "no sampling")
    ap.add_argument("--p-pivot", type=float, default=0.3)
    ap.add_argument("--f-stop", type=float, default=1e-6)
    ap.add_argument("--move-cap", type=int, default=10_000_000_000,
                    help="per-window total-move ceiling (raise and re-invoke "
                         "to continue past it)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--ckpt-dir", default=CKPT_DIR)
    ap.add_argument("--ref-max", type=int, default=REF_MAX,
                    help="trust the global g only up to this L")
    a = ap.parse_args()

    windows = build_windows()
    reference = build_reference(ref_max=a.ref_max)
    os.makedirs(a.ckpt_dir, exist_ok=True)

    if a.report:
        raise SystemExit(_report(windows, reference, a.ckpt_dir, a.f_stop))

    if a.window:
        ordered = sorted(windows, key=lambda w: w["lmin"])
        names = [w["name"] for w in ordered]
        if a.window not in names:
            raise SystemExit(
                f"unknown window {a.window!r}; choose from {names}")
        i = names.index(a.window)
        w = ordered[i]
        # seed + i matches run_windowed_campaign's per-window seeds, so a
        # later full-campaign invocation resumes these checkpoints as-is
        print(f"{w['name']} [{w['lmin']},{w['lmax']}] p_pivot={a.p_pivot} "
              f"seed={a.seed + i} move_cap={a.move_cap:,}", flush=True)
        st = run_window_checkpointed(
            9, w["lmin"], w["lmax"], w["seed_snake"],
            os.path.join(a.ckpt_dir, w["name"] + ".json"),
            p_pivot=a.p_pivot, f_stop=a.f_stop, move_cap=a.move_cap,
            seed=a.seed + i, on_progress=_progress)
        done = st.f <= a.f_stop
        print(f"{w['name']}: moves={st.moves / 1e6:.0f}M f={st.f:.1e} "
              + ("converged" if done
                 else "move cap reached — re-invoke to continue"), flush=True)
        return

    out = run_windowed_campaign(
        9, windows, reference, a.ckpt_dir,
        our_best=OUR_BEST, record_len=RECORD_LEN,
        p_pivot=a.p_pivot, f_stop=a.f_stop, move_cap=a.move_cap,
        seed=a.seed, on_progress=_progress)
    print(f"anchor residual vs trusted global g: "
          f"{out['anchor_residual']:.3f} nats (rms)")
    tail = {L: round(v, 2) for L, v in sorted(out["S"].items()) if L >= 170}
    print(f"S(L) 170..: {tail}")
    print(f"verdict: {out['verdict']}", flush=True)


if __name__ == "__main__":
    main()
