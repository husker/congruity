"""Route-then-deepen: take a router assembly and spend real depth on it.

The router (hierarchical_router.py) buys breadth — shaped, probe-ranked
assemblies — but saturates near 178 because no piece of the assembly
gets pipeline-grade depth. This second phase deepens one assembly:

  A. two-ended exact polish (polish_sequence — unwind each end k steps,
     recomplete exhaustively at deep caps), then
  B. a sliding window-repair sweep (window_repair — cut seq[i+1:j],
     exact-search a replacement of the same length + extra), looping
     back to A whenever anything improves, until a full pass is dry.

Every improvement is pack-validated, checkpointed to disk immediately
(chunk-safe: rerunning resumes from the checkpoint), and ledgered.

    python -m experiments.snake_in_box.route_then_deepen \
        --in data/snake_in_box/hrouter_best.json --widths 8,12 --extras 1,2
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from experiments.snake_in_box.baselines import validate_snake
from experiments.snake_in_box.complete_certificate import polish_sequence
from experiments.snake_in_box.window_repair import window_repair

LEDGER = Path("data/snake_in_box/deepen_ledger.jsonl")


def _log(entry: dict) -> None:
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    with LEDGER.open("a") as f:
        f.write(json.dumps(entry) + "\n")


def deepen(n: int, seq: list[int], widths: list[int], extras: list[int],
           unwind: int, polish_cap: int, window_cap: int, stride: int,
           checkpoint: Path, max_rounds: int = 8) -> list[int]:
    ok, length = validate_snake(n, seq)
    assert ok, "input path failed validation"
    best = list(seq)
    for rnd in range(1, max_rounds + 1):
        improved = False
        # A: two-ended exact polish
        t0 = time.time()
        p = polish_sequence(n, best, unwind, polish_cap)
        if len(p) > len(best):
            best = p
            improved = True
            print(f"round {rnd}: polish {length}e -> {len(best) - 1}e "
                  f"({time.time() - t0:.0f}s)", flush=True)
        length = len(best) - 1
        checkpoint.write_text(json.dumps({"n": n, "length": length,
                                          "path": best, "round": rnd}))
        # B: sliding window surgery, widest cuts first
        for w in sorted(widths, reverse=True):
            for extra in extras:
                i = 0
                while i + w + 1 < len(best):
                    rep, nodes, exh = window_repair(
                        n, best, i, i + w + 1, extra, window_cap)
                    if rep is not None:
                        okr, lr = validate_snake(n, rep)
                        if okr and int(lr) > length:
                            print(f"round {rnd}: window w={w} i={i} "
                                  f"+{extra} -> {int(lr)}e", flush=True)
                            best, length, improved = rep, int(lr), True
                            checkpoint.write_text(json.dumps(
                                {"n": n, "length": length, "path": best,
                                 "round": rnd}))
                            _log({"event": "window_gain", "w": w, "i": i,
                                  "extra": extra, "length": length})
                            i = 0        # restart the sweep on the new path
                            continue
                    i += stride
        if not improved:
            print(f"round {rnd}: dry pass — fixed point at {length}e",
                  flush=True)
            break
    return best


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, type=Path,
                    help="router --out JSON ({n, length, path, ...}) or a "
                         "deepen checkpoint")
    ap.add_argument("--widths", default="8,12")
    ap.add_argument("--extras", default="1,2")
    ap.add_argument("--unwind", type=int, default=40)
    ap.add_argument("--polish-cap", type=int, default=200_000_000)
    ap.add_argument("--window-cap", type=int, default=20_000_000)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--max-rounds", type=int, default=8)
    ap.add_argument("--checkpoint", type=Path,
                    default=Path("data/snake_in_box/deepen_checkpoint.json"))
    args = ap.parse_args(argv)

    src = json.loads(args.inp.read_text())
    n, seq = src["n"], src["path"]
    widths = [int(w) for w in args.widths.split(",")]
    extras = [int(e) for e in args.extras.split(",")]
    t0 = time.time()
    best = deepen(n, seq, widths, extras, args.unwind, args.polish_cap,
                  args.window_cap, args.stride, args.checkpoint,
                  args.max_rounds)
    ok, length = validate_snake(n, best)
    print(f"deepen: {src['length']}e -> "
          f"{int(length) if ok else 'INVALID'}e ({time.time() - t0:.0f}s)",
          flush=True)
    _log({"event": "done", "src": str(args.inp), "start": src["length"],
          "end": int(length), "widths": widths, "extras": extras,
          "unwind": args.unwind, "polish_cap": args.polish_cap,
          "window_cap": args.window_cap, "stride": args.stride})
    if ok and int(length) > 185:
        cert = Path(f"data/snake_in_box/certificates/"
                    f"q{n}_len{int(length)}_deepen.json")
        cert.write_text(json.dumps({"n": n, "length": int(length),
                                    "seq": best, "method": "route_then_deepen"}))
        print(f"*** NEW BEST {int(length)}e -> {cert}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(None))
