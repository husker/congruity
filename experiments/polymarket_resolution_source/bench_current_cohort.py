"""Polymarket latency-structure replication on current-cohort markets.

Follow-up to `bench_latency_structure.py` (finding:
`findings/polymarket-latency-structure-20260524.md`).

The original analysis used a 2000-market cache that Gamma pagination
delivered oldest-first, yielding a 2020-2022 cohort. The finding:
sharp ~41h modal delay, US-business-hours concentration, weekend dip.

This bench tests **whether that mechanism persists on the current
(2024+) high-volume political/sports cohort.** Two failure modes to
distinguish:

  1. **Stable mechanism**: same modal value, same hour-of-day
     pattern → the resolution machinery is unchanged; the original
     finding generalizes forward in time.
  2. **Process drift**: different modal value or hour distribution →
     Polymarket's resolution mechanism evolved (UMA contract update,
     bot cadence change, MIC SLA shift). The original finding is a
     historical snapshot, not a current-state claim.

Method: refetch via Gamma API with `end_date_min=2024-01-01T00:00:00Z`,
sorted ascending by endDate. Run the same latency-structure analysis
and compare key statistics directly against the 2020-2022 cohort
numbers in the existing finding.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import requests

CACHE = Path(__file__).parent / "_cache"
CACHE.mkdir(exist_ok=True)


def _parse_dt(s):
    if not s:
        return None
    s = s.strip()
    for fmt in (
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S.%fZ",
        "%Y-%m-%d %H:%M:%S+00",
        "%Y-%m-%d %H:%M:%S.%f+00",
    ):
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


def fetch_current_cohort(
    n_target: int, end_date_min: str, force: bool = False
) -> list[dict]:
    cache = CACHE / f"polymarket_current_{n_target}_{end_date_min[:10]}.json"
    if cache.exists() and not force:
        print(f"  cached: {cache.name}", flush=True)
        return json.loads(cache.read_text())
    print(
        f"  fetching ~{n_target} closed markets with endDate >= {end_date_min}",
        flush=True,
    )
    out: list[dict] = []
    offset = 0
    page_size = 100
    while len(out) < n_target:
        url = (
            f"https://gamma-api.polymarket.com/markets"
            f"?limit={page_size}&offset={offset}&closed=true"
            f"&end_date_min={end_date_min}"
        )
        try:
            r = requests.get(url, timeout=30)
            r.raise_for_status()
        except Exception as e:
            print(f"    fetch error at offset {offset}: {e}", flush=True)
            break
        page = r.json()
        if not page:
            break
        for m in page:
            uma_raw = m.get("umaResolutionStatuses") or "[]"
            try:
                uma = json.loads(uma_raw) if isinstance(uma_raw, str) else uma_raw
            except Exception:
                uma = []
            out.append(
                {
                    "id": m.get("id"),
                    "question": m.get("question") or "",
                    "description": m.get("description") or "",
                    "category": m.get("category") or "uncategorized",
                    "end_date": m.get("endDate"),
                    "closed_time": m.get("closedTime"),
                    "volume": float(m.get("volumeNum") or 0),
                    "uma_resolution_statuses": uma,
                }
            )
        offset += page_size
        if len(page) < page_size:
            break
        if offset % 500 == 0:
            print(f"    fetched {len(out)}...", flush=True)
        time.sleep(0.2)
    cache.write_text(json.dumps(out))
    print(f"  cached {len(out)} markets to {cache.name}", flush=True)
    return out


def delay_hours(m):
    end = _parse_dt(m.get("end_date"))
    closed = _parse_dt(m.get("closed_time"))
    if end is None or closed is None:
        return None
    return (closed - end).total_seconds() / 3600.0


def kde_peaks(values, lo, hi, n_points=400, bandwidth=2.0):
    xs = np.linspace(lo, hi, n_points)
    diffs = (xs[:, None] - values[None, :]) / bandwidth
    kde = np.exp(-0.5 * diffs * diffs).sum(axis=1)
    kde /= kde.sum()
    peaks = []
    for i in range(2, len(kde) - 2):
        if (
            kde[i] > kde[i - 1]
            and kde[i] > kde[i + 1]
            and kde[i] > kde[i - 2]
            and kde[i] > kde[i + 2]
        ):
            peaks.append((float(xs[i]), float(kde[i])))
    peaks.sort(key=lambda p: -p[1])
    return peaks[:5]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-target", type=int, default=2500)
    ap.add_argument("--end-date-min", default="2024-01-01T00:00:00Z")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    print(
        "\n=== Current-cohort Polymarket latency replication ==="
        f"\ntarget: {args.n_target} markets with endDate >= {args.end_date_min}\n"
    )

    markets = fetch_current_cohort(
        args.n_target, args.end_date_min, force=args.force
    )
    print(f"\nfetched: {len(markets)} markets")

    # Restrict to those with usable delay
    rows = []
    for m in markets:
        d = delay_hours(m)
        if d is None:
            continue
        rows.append((m, d))
    print(f"with usable delay: {len(rows)}")

    delays = np.array([r[1] for r in rows])
    closure_years = []
    for m, _ in rows:
        ct = _parse_dt(m.get("closed_time"))
        if ct:
            closure_years.append(ct.year)
        else:
            closure_years.append(None)

    print(
        f"\nclosure-year distribution: {dict(sorted(Counter([y for y in closure_years if y]).items()))}"
    )
    print(f"end-date year range: "
          f"{min((_parse_dt(m['end_date']).year for m, _ in rows), default='?')} → "
          f"{max((_parse_dt(m['end_date']).year for m, _ in rows), default='?')}")

    print("\ndelay statistics (hours):")
    print(f"  count        = {len(delays)}")
    print(f"  median       = {float(np.median(delays)):.2f}h")
    print(f"  mean         = {float(np.mean(delays)):.2f}h")
    print(f"  10th pctl    = {float(np.percentile(delays, 10)):.2f}h")
    print(f"  90th pctl    = {float(np.percentile(delays, 90)):.2f}h")

    # Mode finding via KDE: focus on 0-200h window
    in_window = delays[(delays >= 0) & (delays <= 200)]
    print(f"\n0-200h window: {len(in_window)} markets")
    peaks = kde_peaks(in_window, 0, 200, bandwidth=2.0)
    print("KDE peaks (top-5, hours):")
    for p, d in peaks:
        within_4h = float(np.mean((in_window >= p - 4) & (in_window <= p + 4)))
        print(f"    {p:>6.1f}h  density={d:.4f}  within ±4h: {within_4h:>5.1%}")

    # Hour-of-day of closure
    hours_utc = []
    for m, _ in rows:
        ct = _parse_dt(m.get("closed_time"))
        if ct:
            hours_utc.append(ct.hour)
    hour_counts = Counter(hours_utc)
    total_h = sum(hour_counts.values())
    print("\nclosure hour-of-day (UTC):")
    uniform = total_h / 24 if total_h else 1
    for h in range(0, 24, 2):
        c = hour_counts.get(h, 0) + hour_counts.get(h + 1, 0)
        ratio = c / (2 * uniform) if uniform else 0
        bar = "█" * int(min(40, ratio * 8))
        print(f"  {h:02d}-{h+2:02d} UTC | {bar:<40}  {c:>4d}  ratio={ratio:.2f}x")

    # Day-of-week
    dow = []
    for m, _ in rows:
        ct = _parse_dt(m.get("closed_time"))
        if ct:
            dow.append(ct.weekday())  # 0=Mon, 6=Sun
    dow_counts = Counter(dow)
    print("\nclosure day-of-week:")
    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    uniform_dow = sum(dow_counts.values()) / 7
    for i in range(7):
        c = dow_counts.get(i, 0)
        ratio = c / uniform_dow if uniform_dow else 0
        bar = "█" * int(min(40, ratio * 16))
        print(f"  {days[i]:>3s} | {bar:<40}  {c:>4d}  ratio={ratio:.2f}x")

    # Volume vs delay
    vols = np.array([float(m.get("volume") or 0) for m, _ in rows])
    print("\nvolume distribution (USD):")
    print(f"  median = ${float(np.median(vols)):,.0f}")
    print(f"  90th pctl = ${float(np.percentile(vols, 90)):,.0f}")
    print(f"  max = ${float(np.max(vols)):,.0f}")

    # Bucket by volume tertile, compare median delay
    pos_vols = vols[vols > 0]
    if len(pos_vols) > 30:
        t1 = float(np.percentile(pos_vols, 33))
        t2 = float(np.percentile(pos_vols, 67))
        low = delays[vols < t1]
        mid = delays[(vols >= t1) & (vols < t2)]
        high = delays[vols >= t2]
        print("\ndelay by volume tertile (only markets with vol > 0):")
        print(f"  low  vol (<${t1:,.0f}):    n={len(low):>4d}  "
              f"median delay = {float(np.median(low)):.2f}h")
        print(f"  mid  vol (${t1:,.0f}-${t2:,.0f}): n={len(mid):>4d}  "
              f"median delay = {float(np.median(mid)):.2f}h")
        print(f"  high vol (>=${t2:,.0f}):   n={len(high):>4d}  "
              f"median delay = {float(np.median(high)):.2f}h")

    # Category breakdown
    cat_delays = {}
    for m, d in rows:
        c = m.get("category") or "uncategorized"
        cat_delays.setdefault(c, []).append(d)
    print("\nper-category delay (categories with >=30 markets):")
    print("  category                       n     median (h)   IQR (h)")
    for c, ds in sorted(cat_delays.items(), key=lambda kv: -len(kv[1])):
        if len(ds) < 30:
            continue
        arr = np.array(ds)
        med = float(np.median(arr))
        iqr = float(np.percentile(arr, 75) - np.percentile(arr, 25))
        print(f"  {c:<30s}  {len(arr):>4d}  {med:>9.2f}  {iqr:>7.2f}")

    # Comparison with 2020-2022 cohort numbers (from finding doc)
    print("\n=== comparison with 2020-2022 cohort (from polymarket-latency-structure-20260524.md) ===")
    print("            metric            |  2020-22 cohort  |  current cohort  ")
    print(" ---------------------------- + ---------------- + -----------------")
    print(f"  modal delay (KDE peak)     |        41.0h     |  {peaks[0][0]:>7.1f}h        " if peaks else "(no peaks)")
    print(f"  median delay overall       |        ~32h      |  {float(np.median(delays)):>7.2f}h")
    print("  18:00 UTC concentration    |        3.0x      |  ?  (see hour table)")
    print("  weekend dip (Sat ratio)    |        0.66x     |  ?  (see dow table)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
