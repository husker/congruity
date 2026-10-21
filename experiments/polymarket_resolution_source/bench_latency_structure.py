"""Follow-up to C8 — Polymarket resolution-latency structural analysis.

C8 found that resolution-source category does NOT predict delay, but
flagged a stable platform-level ~30-40h median delay across ALL
categories. This bench asks the natural follow-up: what *causes* that
floor?

Method (no new API calls — pure analysis on the cached 2000-market
fetch from C8):

  1. Build a delay histogram with fine bins. A sharp peak at a
     specific value would identify a fixed mechanism (UMA liveness
     period, bot polling cadence, MIC SLA). A smooth distribution
     would indicate multi-process latency.

  2. Stratify by closure year. Polymarket's mechanism has evolved
     — modal-shift over years would point to specific contract or
     process changes.

  3. Hour-of-day clustering of closedTime. Concentration at specific
     UTC hours implies a scheduled job; uniform distribution implies
     event-driven resolution.

  4. Per-category breakdown. The aggregate median might mask faster
     resolution for one category (e.g., sports) and slower for
     another (e.g., political).

  5. Mode-identification. Kernel density estimate + peak finding.

Run:
    uv run --with numpy python \
        -m experiments.polymarket_resolution_source.bench_latency_structure
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

CACHE = Path(__file__).parent / "_cache"


def _parse_dt(s: str | None) -> datetime | None:
    if not s:
        return None
    s = s.strip()
    for fmt in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S.%fZ",
                "%Y-%m-%d %H:%M:%S+00", "%Y-%m-%d %H:%M:%S.%f+00"):
        try:
            dt = datetime.strptime(s, fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        return None


def ascii_histogram(values: np.ndarray, lo: float, hi: float, n_bins: int,
                    bar_width: int = 50, label_fmt: str = "{:>7.1f}") -> str:
    """Render a horizontal text histogram."""
    bins = np.linspace(lo, hi, n_bins + 1)
    hist, _ = np.histogram(values, bins=bins)
    max_count = max(hist.max(), 1)
    lines = []
    for i in range(n_bins):
        center = (bins[i] + bins[i + 1]) / 2
        bar_len = int(hist[i] * bar_width / max_count)
        pct = 100 * hist[i] / len(values)
        lines.append(
            f"  {label_fmt.format(center)}h | "
            f"{'█' * bar_len:<{bar_width}}  {hist[i]:>4d} ({pct:>4.1f}%)"
        )
    return "\n".join(lines)


def kde_peaks(values: np.ndarray, lo: float, hi: float,
              n_points: int = 400, bandwidth: float = 2.0) -> list[float]:
    """Find local maxima in a Gaussian KDE over [lo, hi]. Returns peak
    locations sorted by density (descending)."""
    xs = np.linspace(lo, hi, n_points)
    # Gaussian kernel
    diffs = (xs[:, None] - values[None, :]) / bandwidth
    kde = np.exp(-0.5 * diffs * diffs).sum(axis=1)
    kde /= kde.sum()
    # Find local maxima
    peaks = []
    for i in range(2, len(kde) - 2):
        if kde[i] > kde[i - 1] and kde[i] > kde[i + 1] and \
           kde[i] > kde[i - 2] and kde[i] > kde[i + 2]:
            peaks.append((xs[i], float(kde[i])))
    peaks.sort(key=lambda p: -p[1])
    return peaks


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-file",
                    default=str(CACHE / "polymarket_resolved_2000.json"))
    args = ap.parse_args()

    if not Path(args.cache_file).exists():
        print(f"Cache not found: {args.cache_file}")
        print("Run experiments.polymarket_resolution_source.bench first.")
        return 1
    markets = json.loads(Path(args.cache_file).read_text())
    print(f"loaded {len(markets)} markets from cache\n")

    # Compute delays + decorations
    rows = []
    for m in markets:
        end = _parse_dt(m.get("end_date"))
        closed = _parse_dt(m.get("closed_time"))
        if end is None or closed is None:
            continue
        delay = (closed - end).total_seconds() / 3600.0
        rows.append({
            "delay_h": delay,
            "closed_year": closed.year,
            "closed_hour_utc": closed.hour,
            "closed_dow": closed.weekday(),  # 0=Mon
            "category": m.get("category") or "uncategorized",
            "volume": m.get("volume", 0),
        })
    print(f"usable rows: {len(rows)}\n")

    delays = np.array([r["delay_h"] for r in rows])

    # ---- percentile table ----
    print("=== delay percentiles (hours) ===")
    pct = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    for p in pct:
        v = np.percentile(delays, p)
        print(f"  P{p:>2d}: {v:>+10.2f}")
    print(f"  mean: {delays.mean():>+10.2f}    std: {delays.std():>10.2f}")
    print(f"  range: [{delays.min():+.1f}, {delays.max():+.1f}]")

    # ---- KDE peaks ----
    print("\n=== KDE-identified peaks in the 0-200h range ===")
    peaks = kde_peaks(delays, lo=0, hi=200, bandwidth=2.0)
    for x, dens in peaks[:8]:
        # Density value relative to mode
        rel = dens / peaks[0][1] if peaks else 0
        print(f"  peak at {x:>+7.2f}h   relative density = {rel:.3f}")

    # ---- fine-grained ASCII histogram in 0-150h range ----
    print("\n=== fine histogram, 0-150h, 30 bins (5h each) ===")
    in_range = delays[(delays >= 0) & (delays <= 150)]
    print(f"  ({len(in_range)} / {len(delays)} markets in range)")
    print(ascii_histogram(in_range, lo=0, hi=150, n_bins=30))

    # ---- narrow histogram in 0-72h range (sub-day resolution) ----
    print("\n=== sub-3-day histogram, 0-72h, 36 bins (2h each) ===")
    in_range_72 = delays[(delays >= 0) & (delays <= 72)]
    print(f"  ({len(in_range_72)} / {len(delays)} markets in range)")
    print(ascii_histogram(in_range_72, lo=0, hi=72, n_bins=36))

    # ---- year stratification ----
    print("\n=== year-over-year stratification ===")
    by_year = defaultdict(list)
    for r in rows:
        by_year[r["closed_year"]].append(r["delay_h"])
    print("  year |  n   | median  |   P25   |   P75   |  mean    | KDE_peak(0-200h)")
    print("  -----+------+---------+---------+---------+----------+------------------")
    for year in sorted(by_year):
        d = np.array(by_year[year])
        if len(d) < 20:
            continue
        peaks_y = kde_peaks(d, lo=0, hi=200, bandwidth=2.0)
        peak_str = f"{peaks_y[0][0]:.1f}h" if peaks_y else "n/a"
        print(f"  {year} | {len(d):>4d} | "
              f"{np.median(d):>+7.2f} | "
              f"{np.percentile(d, 25):>+7.2f} | "
              f"{np.percentile(d, 75):>+7.2f} | "
              f"{d.mean():>+8.2f} | "
              f"{peak_str}")

    # ---- hour-of-day clustering ----
    print("\n=== closedTime hour-of-UTC histogram ===")
    hours = [r["closed_hour_utc"] for r in rows]
    hcount = Counter(hours)
    expected = len(rows) / 24
    print(f"  expected uniform: {expected:.1f} per hour")
    max_bar = 40
    max_count = max(hcount.values())
    for h in range(24):
        c = hcount.get(h, 0)
        bar = "█" * int(c * max_bar / max_count)
        ratio = c / expected
        marker = "  ★" if ratio > 1.5 else ("  ·" if ratio < 0.5 else "")
        print(f"  {h:>2d}:00 UTC | {bar:<40}  {c:>3d}  ({ratio:.2f}x uniform){marker}")

    # ---- day-of-week clustering ----
    print("\n=== closedTime day-of-week histogram (0=Mon … 6=Sun) ===")
    dows = [r["closed_dow"] for r in rows]
    dcount = Counter(dows)
    expected_d = len(rows) / 7
    max_dcount = max(dcount.values())
    names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    for d in range(7):
        c = dcount.get(d, 0)
        bar = "█" * int(c * max_bar / max_dcount)
        ratio = c / expected_d
        marker = "  ★" if ratio > 1.2 else ("  ·" if ratio < 0.7 else "")
        print(f"  {names[d]} | {bar:<40}  {c:>3d}  ({ratio:.2f}x uniform){marker}")

    # ---- per-category median ----
    print("\n=== per-category delay summary (top 10 by count) ===")
    by_cat = defaultdict(list)
    for r in rows:
        by_cat[r["category"]].append(r["delay_h"])
    cat_n = sorted(by_cat.items(), key=lambda kv: -len(kv[1]))[:10]
    print("  category                          n     median   P25-P75 IQR")
    print("  ----------------------------------+----+---------+----------")
    for cat, ds in cat_n:
        d = np.array(ds)
        print(f"  {cat[:32]:<32}  {len(d):>4d}  "
              f"{np.median(d):>+7.2f}  "
              f"{np.percentile(d, 75) - np.percentile(d, 25):>+8.2f}")

    # ---- shape diagnostic ----
    print("\n=== distribution shape diagnostic ===")
    # Identify the sharpest peak: look at the highest-density 2h bin
    bins = np.arange(0, 100, 2.0)
    hist, _ = np.histogram(delays[delays >= 0], bins=bins)
    if hist.sum() == 0:
        print("  (no positive-delay markets)")
        return 0
    modal_bin = int(np.argmax(hist))
    modal_center = bins[modal_bin] + 1.0
    modal_count = int(hist[modal_bin])
    total_in_window = int(hist.sum())
    pct_at_mode = 100.0 * modal_count / total_in_window
    pct_within_4h = 100.0 * hist[max(0, modal_bin - 2): modal_bin + 3].sum() / total_in_window
    print(f"  modal 2h bin centered at {modal_center:+.1f}h")
    print(f"  fraction of markets in this 2h bin:        {pct_at_mode:.1f}%")
    print(f"  fraction within ±4h of mode (10h window): {pct_within_4h:.1f}%")
    print(f"  expected uniform over 0-100h with 2h bins: {100/50:.1f}%")
    print(f"  expected uniform over 0-100h with 10h bins: {100/10:.1f}%")
    print()
    if pct_at_mode > 10.0:
        print("  → SHARP MODE — concentration at this delay strongly suggests")
        print("    a single dominant mechanism (UMA liveness, bot cadence,")
        print("    or scheduled review queue) operates around this value.")
    elif pct_within_4h > 25.0:
        print("  → MODERATE PEAK — most markets cluster within ±4h of the")
        print("    mode, but the distribution is broader than a single fixed")
        print("    deadline would produce. Likely a few dominant mechanisms.")
    else:
        print("  → SMOOTH DISTRIBUTION — no single dominant mechanism. The")
        print("    platform-level ~30-40h median emerges from many small")
        print("    contributing factors with no single fixed-deadline floor.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
