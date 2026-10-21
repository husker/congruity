"""Conflict-core prototype: minimize the 185-bottom's topA refutation.

The 185's topA search exhausted at 43 given blockedA (the bottom's
low-cell labels). Greedy deletion-based core: drop label batches, re-run
the exact search; a batch stays dropped iff the cap provably (exhausted)
stays <= baseline. Output: core size, probes, wall time.
"""
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from experiments.snake_in_box import fastcore  # noqa: E402 — bootstrap direct-script imports

BASE_BUDGET = 4_000_000_000
PROBE_BUDGET = 1_000_000_000
SLACK = 1          # mine the core of the weaker cap "<= baseline+SLACK"

cert = json.loads((REPO / "data/snake_in_box/certificates/q9_len185_joint.json").read_text())
seq = cert["seq"]
bottom = [v for v in seq if v < 256]
topA = [v - 256 for v in seq if 256 <= v < 384]
j8 = topA[0]
lb_low = {v for v in bottom if v < 128}
blockedA = sorted(lb_low - {j8})
print(f"bottom {len(bottom)}v, topA {len(topA)}v (={len(topA)-1}e), "
      f"blockedA {len(blockedA)} labels, j8={j8}")

t0 = time.time()
base, _n, exh = fastcore.complete_blocked(7, [j8], blockedA, BASE_BUDGET,
                                          rng_seed=-1)
baseline = len(base) - 1
print(f"baseline topA cap: {baseline}e exhausted={exh} "
      f"({time.time()-t0:.1f}s)", flush=True)
assert exh, "baseline not exhausted — raise BASE_BUDGET"
cap = baseline + SLACK          # core certifies: topA <= cap

# far-from-junction labels are the likeliest deletable (walls form
# near the path) — try them first
core = sorted(blockedA, key=lambda v: -bin(v ^ j8).count("1"))
probes = 0


def still_capped(labels):
    global probes
    probes += 1
    got, _n, e = fastcore.complete_blocked(7, [j8], sorted(labels),
                                           PROBE_BUDGET, rng_seed=-1)
    return e and (len(got) - 1) <= cap


# pass 1: batches of 6, pass 2: singles
for size in (6, 1):
    i = 0
    while i < len(core):
        batch = core[:i] + core[i + size:]
        if still_capped(batch):
            core = batch
        else:
            i += size
    print(f"after pass size={size}: {len(core)} labels, {probes} probes, "
          f"{time.time()-t0:.0f}s", flush=True)
print(f"CORE for topA<={cap}: {len(core)}/{len(blockedA)} labels, "
      f"{probes} probes, {time.time()-t0:.0f}s total")
print("core labels:", sorted(core))
