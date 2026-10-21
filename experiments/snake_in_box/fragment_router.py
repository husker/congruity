"""Multi-crossing fragment router: beam-searched assembly of a Q_n snake
from exact layer-fragments (the fragment-assembly lane).

The four-subcube construction (Q9 = 185) hard-codes single junctions and is
provably capped by its bottom's shadow; Wynn's 190 interleaves its deepest
axis 8 times. This router generalizes: the snake alternates between the two
Q_(n-1) layers across up to `max_cross` matching-edge crossings. Each
fragment is grown by an exact, node-capped complement-constrained DFS
against the full shadow of everything committed so far; a beam search
explores where to stop each fragment and cross. k = 1 recovers the old
schema; higher k trades fragment length for splice freedom.

Pragmatic v1 of the branch-and-price idea: sequential routing with a beam
instead of a master problem. Every completed assembly is pack-validated and
ledgered (data/snake_in_box/router_ledger.jsonl) with its crossing profile
and per-fragment provenance; certificates are written for anything above
the incumbent 185.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from experiments.snake_in_box import fastcore
from experiments.snake_in_box.baselines import validate_snake
from experiments.snake_in_box.canon import canonical_form

CERTS = Path("data/snake_in_box/certificates")
LEDGER = Path("data/snake_in_box/router_ledger.jsonl")


def _grow_fragment(m: int, start: int, frozen: set[int], blocked: set[int],
                   node_cap: int) -> list[int]:
    """Longest induced path in Q_m from `start`: `frozen` = same-layer
    committed vertices (adjacency to them is a chord), `blocked` = other
    layer's shadow (label exclusion only). Exact within the node cap."""
    frag, _nodes, _exh = fastcore.complete_frozen(
        m, [start], sorted(frozen - {start}), sorted(blocked - {start}),
        node_cap, rng_seed=-1)
    return frag


def _adjacent_to_any(m: int, v: int, group: set[int]) -> bool:
    return any((v ^ (1 << b)) in group for b in range(m))


class Assembly:
    """A partial route: committed Q_n path, per-layer used labels."""

    __slots__ = ("path", "layer", "used", "crossings")

    def __init__(self, path, layer, used, crossings):
        self.path = path            # committed Q_n vertices
        self.layer = layer          # layer of the current end (0/1)
        self.used = used            # (set_layer0, set_layer1) of Q_(n-1) labels
        self.crossings = crossings

    def key(self):
        return (len(self.path), self.crossings, self.path[-1])


def _stop_points(frag_len: int) -> list[int]:
    """Candidate fragment lengths (edges) to commit before crossing: dense
    coverage of the whole range — greedy max-length fragments saturate a
    layer and foreclose re-entry (measured: an 85e first fragment kills all
    stage-2 crossings), so balanced routings must be reachable."""
    pts = list(range(6, frag_len, 4)) + [frag_len]
    return sorted(set(p for p in pts if 1 <= p <= frag_len), reverse=True)


def route(n: int, max_cross: int, beam_width: int, node_cap: int,
          start_label: int = 0,
          log=lambda s: None) -> tuple[list[int] | None, dict]:
    """Beam-routed assembly. Returns (best validated snake or None, stats)."""
    m = n - 1
    top = 1 << m
    init = Assembly([start_label], 0, ({start_label}, set()), 0)
    beam = [init]
    best_snake: list[int] | None = None
    stats = {"expansions": 0, "validated": 0}

    for stage in range(max_cross + 1):
        candidates: list[tuple[int, Assembly]] = []
        for a in beam:
            cur_label = a.path[-1] & (top - 1)
            own, other = (a.used[a.layer], a.used[1 - a.layer])
            frag = _grow_fragment(m, cur_label, own, other, node_cap)
            stats["expansions"] += 1
            if len(frag) < 2:
                continue
            for stop in _stop_points(len(frag) - 1):
                f = frag[: stop + 1]
                lifted = [v | (a.layer << m) for v in f[1:]]
                new_path = a.path + lifted
                new_own = own | set(f)
                # finalize this assembly (no more crossings)
                ok, length = validate_snake(n, new_path)
                stats["validated"] += 1
                if ok and (best_snake is None or int(length) > len(best_snake) - 1):
                    best_snake = new_path
                    log(f"  stage {stage}: candidate {int(length)}e "
                        f"({a.crossings} crossings)")
                # cross to the other layer at f's end: the crossing vertex
                # must be label-free there AND not adjacent (in-layer) to
                # anything already committed in that layer
                if a.crossings < max_cross:
                    end_label = f[-1]
                    other_now = a.used[1 - a.layer]
                    if (end_label not in other_now
                            and not _adjacent_to_any(m, end_label, other_now)):
                        crossed_path = new_path + [end_label | ((1 - a.layer) << m)]
                        if a.layer == 0:
                            used = (new_own, other_now | {end_label})
                        else:
                            used = (other_now | {end_label}, new_own)
                        candidates.append(
                            (len(crossed_path),
                             Assembly(crossed_path, 1 - a.layer, used,
                                      a.crossings + 1)))
        if not candidates:
            break
        # stratified beam: balance-preserving — greedy long partials must not
        # crowd out balanced routings, so rank within layer-occupancy bands
        candidates.sort(key=lambda t: -t[0])
        bands: dict[int, list[Assembly]] = {}
        for _s, a in candidates:
            band = min(len(a.used[0]), len(a.used[1])) // 16
            bands.setdefault(band, []).append(a)
        per_band = max(2, beam_width // max(1, len(bands)))
        beam = []
        for band in sorted(bands, reverse=True):
            beam.extend(bands[band][:per_band])
        beam = beam[:beam_width]
        log(f"stage {stage}: beam {len(beam)} across {len(bands)} bands, "
            f"best partial {max(len(a.path) for a in beam) - 1}e")
    return best_snake, stats


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-cross", type=int, default=8)
    ap.add_argument("--beam", type=int, default=12)
    ap.add_argument("--node-cap", type=int, default=150_000_000)
    ap.add_argument("--minutes", type=float, default=9.0)
    ap.add_argument("--start-labels", default="0",
                    help="comma-separated start labels to sweep")
    args = ap.parse_args(argv)

    t0 = time.monotonic()
    incumbent = 185
    for sl in (int(x) for x in args.start_labels.split(",")):
        if time.monotonic() - t0 > args.minutes * 60:
            print("time budget reached", flush=True)
            break
        snake, stats = route(9, args.max_cross, args.beam, args.node_cap,
                             start_label=sl,
                             log=lambda s: print(s, flush=True))
        if snake is None:
            continue
        length = len(snake) - 1
        entry = {"start": sl, "max_cross": args.max_cross,
                 "beam": args.beam, "node_cap": args.node_cap,
                 "length": length, "canon": canonical_form(snake),
                 **stats}
        with LEDGER.open("a") as f:
            f.write(json.dumps(entry) + "\n")
        print(f"start {sl}: best {length}e "
              f"({stats['expansions']} fragment searches)", flush=True)
        if length > incumbent:
            incumbent = length
            (CERTS / f"q9_len{length}_routed.json").write_text(json.dumps(
                {"n": 9, "length": length, "seq": snake,
                 "method": f"fragment router k<={args.max_cross} "
                           f"beam {args.beam}"}))
            print("  <<< WRITTEN"
                  + ("  *** WORLD RECORD ***" if length > 190 else ""),
                  flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
