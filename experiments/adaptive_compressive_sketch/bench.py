"""C1 oracle verification — compose run 20260523-033802 (streaming sketches).

C1's synthesis claim:

    P159's single-pixel compressive measurement idea + P233's constrained
    submodular allocation + P194's continual adaptation suggests an
    adaptive compressive sketch: instead of fixed Count-Min/AMS rows,
    allocate projection rows to stream regions whose marginal error
    reduction is highest, and update that allocation when regimes shift.
    The measurable claim is that, on synthetic Zipf/mixture streams with
    drift, adaptive row allocation can reduce frequency or quantile
    error at the same byte budget versus fixed random projections or
    classical sketches.

Same falsification shape as C5/C6: build the ORACLE upper bound and ask
whether it beats fixed sketches by enough to justify the bookkeeping. If
even an oracle with perfect drift-point and per-region statistics can't
open a meaningful gap, no learned controller can — the composition is
foreclosed.

Decisive sub-test for the C1-specific "adapt when regimes shift" piece:
*dynamic-oracle vs static-oracle gap on drift streams*. The static-oracle
already gets perfect full-stream allocation; the dynamic-oracle additionally
reallocates at known drift points. If dynamic ≈ static, the drift-handling
component adds nothing. If static ≈ best fixed, the partitioning component
adds nothing.

Stop rule (regime-aware — lesson from C6 stop-rule wording bug):

  - Non-saturated budgets only (baseline frequency-L2 error must be a
    meaningful fraction of the trivial-estimator error).
  - At non-saturated budgets, dynamic-oracle must beat best fixed
    baseline by >= 15% relative L2 error reduction to count as
    structural support.
  - Saturated budgets (baseline already near zero error) are skipped:
    the regime where sketches don't matter.

Test:

  1. Synthetic Zipf(s=1.1) stream over V keys, N events, with mid-stream
     drift via key-rank permutation: heavy keys in phase 1 become rare
     in phase 2 and vice versa. Full-stream true frequency = phase-1
     count + phase-2 count.
  2. Fixed baselines at total byte budget B:
       - Conservative-update Count-Min (CU-CMS)
       - CountSketch (signed AMS-style — direct random compressive
         projection, the closest classical cousin to P159's mechanism)
  3. Static-oracle partition: knows full-stream key set. Splits keys
     into a "heavy" region (top-H by full-stream freq) and a "rare"
     region. Gives the heavy region exact counters (B/2) and the rare
     region a smaller CU-CMS (B/2). This is the partition C6 proved
     opens a gap under stationarity.
  4. Dynamic-oracle partition: same partition logic, BUT the partition
     is recomputed at the drift point using each phase's own heavy
     keys. Phase 1 uses phase-1 heavy keys as the exact-counter set,
     phase 2 uses phase-2 heavy keys. Per-phase sub-sketches are
     summed at query time to give a full-stream estimate.
  5. Metric: relative L2 frequency estimation error over all keys that
     appear in the stream:
         rel_L2 = sqrt(sum_k (est_k - true_k)^2) / sqrt(sum_k true_k^2)
     Equivalently the cosine-distance complement; a unitless gap that
     normalises away N and V.

If dynamic-oracle ≈ static-oracle on drift streams, the C1-specific
"adapt-when-regime-shifts" addition is structurally dead.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from collections import Counter
from dataclasses import dataclass

import numpy as np


# ---------- Count-Min Sketch (reused from C6) -----------------------------

class CountMinSketch:
    """Standard CMS using SHA-256-derived hash families; optional
    conservative update (Estan & Varghese, 2002)."""

    def __init__(self, width: int, depth: int, conservative: bool = True):
        self.w = max(1, width)
        self.d = max(1, depth)
        self.conservative = conservative
        self.table = np.zeros((self.d, self.w), dtype=np.int64)

    def _hashes(self, key: str) -> list[int]:
        h = hashlib.sha256(key.encode()).digest()
        out = []
        for i in range(self.d):
            start = (i * 7) % (len(h) - 8 + 1)
            chunk = h[start : start + 8]
            out.append(int.from_bytes(chunk, "big") % self.w)
        return out

    def update(self, key: str, count: int = 1) -> None:
        cells = self._hashes(key)
        if not self.conservative:
            for r, c in enumerate(cells):
                self.table[r, c] += count
            return
        current = [int(self.table[r, c]) for r, c in enumerate(cells)]
        new_min = min(current) + count
        for r, c in enumerate(cells):
            if current[r] < new_min:
                self.table[r, c] = new_min

    def estimate(self, key: str) -> int:
        cells = self._hashes(key)
        return int(min(self.table[r, c] for r, c in enumerate(cells)))

    def bytes(self) -> int:
        return self.table.nbytes


# ---------- CountSketch (signed AMS-style compressive projection) --------

class CountSketch:
    """Charikar/Chen/Farach-Colton CountSketch: each of d rows holds a
    signed sum over a hashed column. Estimation is the median of the
    signed cell values. This is a literal Bernoulli-±1 compressive
    measurement model — the P159 analogy in the composition.
    """

    def __init__(self, width: int, depth: int):
        self.w = max(1, width)
        self.d = max(1, depth)
        self.table = np.zeros((self.d, self.w), dtype=np.int64)

    def _hashes_and_signs(self, key: str) -> tuple[list[int], list[int]]:
        h = hashlib.sha256(key.encode()).digest()
        cols, signs = [], []
        for i in range(self.d):
            start = (i * 5) % (len(h) - 9 + 1)
            chunk = h[start : start + 8]
            cols.append(int.from_bytes(chunk, "big") % self.w)
            sign_byte = h[(start + 8) % len(h)]
            signs.append(1 if (sign_byte & 1) else -1)
        return cols, signs

    def update(self, key: str, count: int = 1) -> None:
        cols, signs = self._hashes_and_signs(key)
        for r, (c, s) in enumerate(zip(cols, signs)):
            self.table[r, c] += s * count

    def estimate(self, key: str) -> int:
        cols, signs = self._hashes_and_signs(key)
        vals = np.array(
            [signs[r] * int(self.table[r, cols[r]]) for r in range(self.d)],
            dtype=np.int64,
        )
        return int(np.median(vals))

    def bytes(self) -> int:
        return self.table.nbytes


# ---------- Oracle partition allocators ----------------------------------

class StaticOraclePartition:
    """Static-oracle: knows the full-stream top-H heavy keys. Allocates
    half the budget to exact counters for those H keys, half to a
    CU-CMS for everyone else. Single sketch over the whole stream.
    """

    def __init__(self, heavy_keys: set[str], exact_budget_bytes: int,
                 cms_width: int, cms_depth: int):
        self.heavy = heavy_keys
        self.exact: dict[str, int] = {k: 0 for k in heavy_keys}
        self.cms = CountMinSketch(cms_width, cms_depth, conservative=True)
        self.exact_budget = exact_budget_bytes

    def update(self, key: str, count: int = 1) -> None:
        if key in self.heavy:
            self.exact[key] += count
        else:
            self.cms.update(key, count)

    def estimate(self, key: str) -> int:
        if key in self.heavy:
            return self.exact[key]
        return self.cms.estimate(key)

    def bytes(self) -> int:
        # Charge exact-counter slots (int64 each), not key strings — both
        # sketches pay only for the counter table, key storage is external.
        return self.exact_budget + self.cms.bytes()


class DynamicOraclePartition:
    """Dynamic-oracle: single continuous CMS for the whole stream plus a
    dynamic exact-counter set that reallocates at the drift point. This
    is the fairest formulation of "adapt allocation when regimes shift":
    the CMS keeps coherent random-projection semantics over the full
    stream, and the only thing that adapts is *which* keys bypass the
    sketch to exact counters.

    Memory accounting (charged in bytes() against the same total budget
    as the fixed baselines):
      - one continuous CU-CMS shared across phases
      - exact-active dict: H slots for whichever phase's heavy keys are
        currently being counted exactly
      - exact-retired dict: H slots holding the frozen exact-count
        values of the previous phase's heavies (must be retained so
        queries return correct totals)

    Update protocol:
      phase 1: P1-heavy keys → exact-active; all others → CMS
      drift: exact-active is moved into exact-retired (values frozen);
             exact-active is re-keyed to P2-heavy with zero counts
      phase 2: P2-heavy keys → exact-active; all others (including
             P1-heavy, which are now retired) → CMS

    Query protocol:
      est(k) = exact-retired[k] (if k was previously exact-active)
             + exact-active[k]  (if k is currently exact-active)
             + CMS estimate for all updates that went to the CMS
      For keys never exact (heavy in neither phase), est is just the
      CMS estimate. For keys exact in both phases (overlap), they're
      either in retired-from-P1 (their P1 counts) plus exact-active-
      in-P2 (their P2 counts), summing to the full true count.
    """

    def __init__(
        self,
        phase1_heavy: set[str],
        phase2_heavy: set[str],
        exact_slots_per_phase: int,
        cms_width: int,
        cms_depth: int,
    ):
        self.phase1_heavy = phase1_heavy
        self.phase2_heavy = phase2_heavy
        # Active = current phase's heavies (gets updates direct to exact)
        # Retired = previous phase's heavies (frozen value at drift)
        self.exact_active: dict[str, int] = {k: 0 for k in phase1_heavy}
        self.exact_retired: dict[str, int] = {}
        self.cms = CountMinSketch(cms_width, cms_depth, conservative=True)
        self.exact_slots_per_phase = exact_slots_per_phase
        self.current_phase = 1

    def set_phase(self, phase: int) -> None:
        assert phase in (1, 2)
        if phase == 2 and self.current_phase == 1:
            # Drift event: freeze active counts into retired, swap active
            # to phase-2 heavy keys with zero counts.
            self.exact_retired = dict(self.exact_active)
            self.exact_active = {k: 0 for k in self.phase2_heavy}
        self.current_phase = phase

    def update(self, key: str, count: int = 1) -> None:
        if key in self.exact_active:
            self.exact_active[key] += count
        else:
            # Everyone else — including retired heavies that aren't
            # currently exact — accumulates in the CMS.
            self.cms.update(key, count)

    def estimate(self, key: str) -> int:
        est = 0
        if key in self.exact_retired:
            est += self.exact_retired[key]
        if key in self.exact_active:
            est += self.exact_active[key]
        # CMS captures everything that wasn't exact-active at the time
        # of its update; only add CMS estimate when the key spent some
        # time in the CMS lane (i.e. wasn't exact in *both* phases).
        was_active_in_p1 = key in self.phase1_heavy
        was_active_in_p2 = key in self.phase2_heavy
        if not (was_active_in_p1 and was_active_in_p2):
            est += self.cms.estimate(key)
        return int(est)

    def bytes(self) -> int:
        # 2 * H int64 slots (active + retired) + CMS table.
        return 2 * self.exact_slots_per_phase * 8 + self.cms.bytes()


# ---------- Stream construction ------------------------------------------

@dataclass
class DriftStream:
    events: list[str]
    drift_at: int
    phase1_truth: dict[str, int]
    phase2_truth: dict[str, int]
    full_truth: dict[str, int]


def make_drift_stream(
    n_events: int,
    n_keys: int,
    zipf_s: float,
    rng: np.random.Generator,
) -> DriftStream:
    """Two-phase Zipf stream. Phase 2 permutes ranks: heavy-in-P1 keys
    become rare in P2 and vice versa.
    """
    ranks = np.arange(1, n_keys + 1)
    weights = 1.0 / (ranks ** zipf_s)
    weights /= weights.sum()
    half = n_events // 2

    # Phase 1: ordered keys k_0..k_{V-1}, key_i has Zipf rank-i weight.
    perm1 = np.arange(n_keys)
    rng.shuffle(perm1)
    p1_ids = rng.choice(perm1, size=half, p=weights)
    events_p1 = [f"k{int(i)}" for i in p1_ids]

    # Phase 2: REVERSE the ranking so heavy→rare and rare→heavy.
    perm2 = perm1[::-1].copy()
    p2_ids = rng.choice(perm2, size=n_events - half, p=weights)
    events_p2 = [f"k{int(i)}" for i in p2_ids]

    events = events_p1 + events_p2
    phase1_truth = dict(Counter(events_p1))
    phase2_truth = dict(Counter(events_p2))
    full = Counter()
    full.update(events_p1)
    full.update(events_p2)
    return DriftStream(
        events=events,
        drift_at=half,
        phase1_truth=phase1_truth,
        phase2_truth=phase2_truth,
        full_truth=dict(full),
    )


def top_heavy_keys(truth: dict[str, int], k: int) -> set[str]:
    """Return the top-k keys by frequency."""
    if k <= 0:
        return set()
    items = sorted(truth.items(), key=lambda kv: -kv[1])
    return {kv[0] for kv in items[:k]}


# ---------- Metric --------------------------------------------------------

def relative_l2(true: dict[str, int], estimator) -> float:
    """sqrt(sum (est - true)^2) / sqrt(sum true^2), summed over keys
    seen at least once. Unitless; saturation ⇒ 0, naive zero ⇒ 1.
    """
    keys = list(true.keys())
    diffs = np.array(
        [estimator.estimate(k) - true[k] for k in keys], dtype=np.float64
    )
    truths = np.array([true[k] for k in keys], dtype=np.float64)
    num = float(np.sqrt((diffs ** 2).sum()))
    den = float(np.sqrt((truths ** 2).sum()))
    return num / den if den > 0 else 0.0


# ---------- Bench harness -------------------------------------------------

@dataclass
class BudgetSpec:
    total_bytes: int
    cms_depth: int
    heavy_count: int


def run_one(
    stream: DriftStream,
    spec: BudgetSpec,
    seed: int,
) -> dict:
    bpc = 8  # int64 cells

    # ----- Fixed baselines at total budget B -----
    cells_full = spec.total_bytes // bpc
    width_full = max(1, cells_full // spec.cms_depth)

    cms_cu = CountMinSketch(width_full, spec.cms_depth, conservative=True)
    countsketch = CountSketch(width_full, spec.cms_depth)
    for e in stream.events:
        cms_cu.update(e)
        countsketch.update(e)

    # ----- Static-oracle: full-stream top-H heavies + smaller CU-CMS -----
    full_heavy = top_heavy_keys(stream.full_truth, spec.heavy_count)
    exact_bytes_static = spec.heavy_count * bpc
    exact_bytes_static = min(exact_bytes_static, spec.total_bytes // 2)
    cms_bytes_static = spec.total_bytes - exact_bytes_static
    cms_cells_static = max(spec.cms_depth, cms_bytes_static // bpc)
    width_static = max(1, cms_cells_static // spec.cms_depth)
    static = StaticOraclePartition(
        full_heavy, exact_bytes_static, width_static, spec.cms_depth
    )
    for e in stream.events:
        static.update(e)

    # ----- Dynamic-oracle: continuous CMS + reallocating exact set -----
    # Storage: 2H exact slots (active + retired) + remaining CMS budget.
    p1_heavy = top_heavy_keys(stream.phase1_truth, spec.heavy_count)
    p2_heavy = top_heavy_keys(stream.phase2_truth, spec.heavy_count)
    exact_slots_per_phase = spec.heavy_count
    exact_bytes_dynamic = 2 * exact_slots_per_phase * bpc
    exact_bytes_dynamic = min(exact_bytes_dynamic, spec.total_bytes // 2)
    cms_bytes_dynamic = spec.total_bytes - exact_bytes_dynamic
    cms_cells_dynamic = max(spec.cms_depth, cms_bytes_dynamic // bpc)
    width_dynamic = max(1, cms_cells_dynamic // spec.cms_depth)
    dynamic = DynamicOraclePartition(
        p1_heavy, p2_heavy, exact_slots_per_phase, width_dynamic, spec.cms_depth
    )
    dynamic.set_phase(1)
    for i, e in enumerate(stream.events):
        if i == stream.drift_at:
            dynamic.set_phase(2)
        dynamic.update(e)

    rl2_cms = relative_l2(stream.full_truth, cms_cu)
    rl2_cs = relative_l2(stream.full_truth, countsketch)
    rl2_static = relative_l2(stream.full_truth, static)
    rl2_dynamic = relative_l2(stream.full_truth, dynamic)
    best_fixed = min(rl2_cms, rl2_cs)
    static_gap = (best_fixed - rl2_static) / best_fixed if best_fixed > 0 else 0.0
    dynamic_gap = (best_fixed - rl2_dynamic) / best_fixed if best_fixed > 0 else 0.0
    dyn_minus_static = (rl2_static - rl2_dynamic) / rl2_static if rl2_static > 0 else 0.0

    return {
        "seed": seed,
        "total_bytes": spec.total_bytes,
        "depth": spec.cms_depth,
        "heavy_count": spec.heavy_count,
        "bytes_cms": cms_cu.bytes(),
        "bytes_countsketch": countsketch.bytes(),
        "bytes_static": static.bytes(),
        "bytes_dynamic": dynamic.bytes(),
        "rl2_cms": rl2_cms,
        "rl2_countsketch": rl2_cs,
        "rl2_static_oracle": rl2_static,
        "rl2_dynamic_oracle": rl2_dynamic,
        "rel_gap_static_vs_best_fixed": static_gap,
        "rel_gap_dynamic_vs_best_fixed": dynamic_gap,
        "rel_gap_dynamic_vs_static": dyn_minus_static,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-events", type=int, default=200_000)
    ap.add_argument("--n-keys", type=int, default=10_000)
    ap.add_argument("--zipf-s", type=float, default=1.1)
    ap.add_argument(
        "--budgets",
        type=int,
        nargs="+",
        default=[16_000, 64_000, 256_000, 1_024_000],
    )
    ap.add_argument("--depths", type=int, nargs="+", default=[4])
    ap.add_argument("--heavy-count", type=int, default=100)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument(
        "--saturation-rl2",
        type=float,
        default=0.05,
        help="If best-fixed relative L2 error <= this, the budget is "
        "considered saturated and the verdict ignores it.",
    )
    ap.add_argument(
        "--required-gap",
        type=float,
        default=0.15,
        help="Required relative L2 error reduction (dynamic-oracle vs "
        "best fixed) at non-saturated budgets to count as structural "
        "support for C1.",
    )
    args = ap.parse_args()

    print("\n=== C1 oracle verification — adaptive compressive sketch ===")
    print(
        f"stream: {args.n_events} events, {args.n_keys} keys, Zipf(s={args.zipf_s}), "
        f"drift via key-rank reversal at midpoint"
    )
    print(
        "baselines: CU-CMS, CountSketch; "
        "oracles: static-partition, dynamic-partition (per-phase)"
    )
    print(f"budgets={args.budgets}  depths={args.depths}  seeds={args.seeds}")
    print(
        f"stop rule: at non-saturated budgets (best-fixed rl2 > "
        f"{args.saturation_rl2:g}), dynamic-oracle must reduce "
        f"rl2 error vs best fixed by >= {args.required_gap:.0%}"
    )
    print(
        "C1-specific test: dynamic-oracle vs static-oracle gap (does "
        "'adapt when regime shifts' add anything over static partitioning?)\n"
    )

    rows = []
    for depth in args.depths:
        print(f"--- CMS depth = {depth} ---")
        for budget in args.budgets:
            spec = BudgetSpec(
                total_bytes=budget,
                cms_depth=depth,
                heavy_count=args.heavy_count,
            )
            for seed in range(args.seeds):
                rng = np.random.default_rng(2000 + seed)
                stream = make_drift_stream(
                    args.n_events, args.n_keys, args.zipf_s, rng
                )
                r = run_one(stream, spec, seed)
                rows.append(r)
                print(
                    f"  B={budget:>7d}B  d={depth}  s={seed}: "
                    f"cms={r['rl2_cms']:.3f}  cs={r['rl2_countsketch']:.3f}  "
                    f"static={r['rl2_static_oracle']:.3f}  "
                    f"dynamic={r['rl2_dynamic_oracle']:.3f}  "
                    f"Δdyn-vs-best={r['rel_gap_dynamic_vs_best_fixed']:+.2%}  "
                    f"Δdyn-vs-static={r['rel_gap_dynamic_vs_static']:+.2%}"
                )

    print("\n=== aggregate (mean over seeds) ===")
    print(
        "  budget  | depth | rl2_cms | rl2_cs | rl2_static | rl2_dyn | "
        "Δdyn-vs-best | Δdyn-vs-static | regime"
    )
    print(
        " ---------+-------+---------+--------+------------+---------+"
        "--------------+----------------+--------"
    )
    by_bd = {}
    for r in rows:
        by_bd.setdefault((r["total_bytes"], r["depth"]), []).append(r)
    win_budgets: list[int] = []
    fail_budgets: list[int] = []
    saturated_budgets: list[int] = []
    static_addon_budgets: list[int] = []
    for (budget, depth), rs in sorted(by_bd.items()):
        m = {
            k: float(np.mean([r[k] for r in rs]))
            for k in (
                "rl2_cms", "rl2_countsketch",
                "rl2_static_oracle", "rl2_dynamic_oracle",
                "rel_gap_dynamic_vs_best_fixed",
                "rel_gap_dynamic_vs_static",
            )
        }
        best_fixed = min(m["rl2_cms"], m["rl2_countsketch"])
        if best_fixed <= args.saturation_rl2:
            regime = "saturated"
            saturated_budgets.append(budget)
        elif m["rel_gap_dynamic_vs_best_fixed"] >= args.required_gap:
            regime = "win"
            win_budgets.append(budget)
            # also flag whether the static partition alone explains the gap
            if m["rel_gap_dynamic_vs_static"] < 0.02:
                static_addon_budgets.append(budget)
        else:
            regime = "fail"
            fail_budgets.append(budget)
        print(
            f"  {budget:>7d} | {depth:>5d} | "
            f"{m['rl2_cms']:.4f}  | {m['rl2_countsketch']:.4f} | "
            f"{m['rl2_static_oracle']:.4f}     | "
            f"{m['rl2_dynamic_oracle']:.4f}  | "
            f"{m['rel_gap_dynamic_vs_best_fixed']:+.2%}       | "
            f"{m['rel_gap_dynamic_vs_static']:+.2%}         | {regime}"
        )

    print("\n=== verdict ===")
    print(f"non-saturated wins  (dyn-vs-best-fixed >= {args.required_gap:.0%}): {win_budgets}")
    print(f"non-saturated fails (dyn-vs-best-fixed <  {args.required_gap:.0%}): {fail_budgets}")
    print(f"saturated budgets (best-fixed rl2 <= {args.saturation_rl2}):       {saturated_budgets}")
    print(f"of the wins, dynamic ≈ static (drift handling adds nothing):     {static_addon_budgets}")
    print()

    if win_budgets and not static_addon_budgets and not fail_budgets:
        print(
            "ORACLE-POSITIVE — dynamic-oracle opens a >=15% rel-L2 error\n"
            "  reduction vs best fixed AT non-saturated budgets, AND the\n"
            "  reduction is NOT explained by static partitioning alone.\n"
            "  The 'adapt when regime shifts' piece carries structural\n"
            "  weight beyond what static heavy/rare partitioning gives.\n"
            "  Open follow-up: can a non-oracle drift detector recoup\n"
            "  most of the dyn-vs-static gap cheaply?"
        )
    elif win_budgets and static_addon_budgets:
        print(
            "PARTIAL-POSITIVE FORECLOSED ON C1's NOVELTY — oracle\n"
            "  partitioning beats fixed sketches at non-saturated\n"
            "  budgets, BUT the dynamic reallocation adds essentially\n"
            "  nothing over static (full-stream) partitioning. The\n"
            "  gap reduces to C6 (importance-aware allocation) and the\n"
            "  C1-specific 'adapt-on-drift' contribution is structurally\n"
            "  dead — the static oracle already gets the full gain."
        )
    elif fail_budgets and not win_budgets:
        print(
            "FORECLOSED — even with oracle knowledge of drift points AND\n"
            "  per-phase heavy keys, the dynamic-oracle adaptive sketch\n"
            "  cannot reduce rel-L2 error by 15% vs best fixed baseline\n"
            "  at any non-saturated budget. The composition's premise is\n"
            "  structurally dead at the upper bound; no learned variant\n"
            "  can recover what the oracle already cannot deliver."
        )
    else:
        print(
            "MIXED — verdict depends on budget regime; inspect the table\n"
            "  above. Likely needs additional drift severity or a wider\n"
            "  budget sweep before committing to a clean verdict."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
