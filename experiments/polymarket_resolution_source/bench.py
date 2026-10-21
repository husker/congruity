"""C8 verification — compose run 20260524-052010.

C8's recipe:
    P201's IPFS availability measurement + P193's search-LLM
    reliability + P178's safe testing gives a resolution-source
    stability monitor for modern prediction markets. The falsifiable
    claim: markets whose resolution criteria depend on unstable/
    decentralized/searchable sources experience more
    delay/dispute/revision than markets that resolve from official
    APIs.

We test on Polymarket (the largest crypto-native event market) where
both resolution-delay and UMA-dispute history are public:

  - delay = closedTime − endDate
       Negative = resolved early; positive = late;
       zero/near-zero = on-schedule.
  - dispute = umaResolutionStatuses array is non-empty
       (UMA is Polymarket's optimistic-oracle dispute mechanism).

We classify resolution sources from the description text into seven
buckets via keyword heuristics — no LLM call needed, fully
deterministic, easy to audit. Test whether source category predicts
delay AND dispute rate.

Stop rule (paraphrased from recipe): fewer than 30 "unstable" and
30 "official-API" examples after labeling → abandon.

Run:
    uv run --with numpy --with requests \
        python -m experiments.polymarket_resolution_source.bench
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import requests

CACHE = Path(__file__).parent / "_cache"
CACHE.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Source classifier
# ---------------------------------------------------------------------------


SOURCE_KEYWORDS = {
    "official_api": [
        # Government / regulator
        "official government", "bureau of labor", "bls.gov", "noaa",
        "national weather service", "fema", "sec filing", "ftc filing",
        "ecb", "imf", "federal reserve", "cdc", "nih.gov",
        "ipcc", "treasury", "bls cpi", "bls.gov",
        # Financial market data
        "bloomberg terminal", "refinitiv", "reuters financial", "yahoo finance",
        "marketwatch", "tradingview official",
        # Sports official
        "official scorebook", "official result", "espn", "nba.com", "nfl.com",
        "fifa.com", "uefa.com", "olympic.org",
        # Other regulated sources
        "wikipedia article", "according to wikipedia",
    ],
    "mainstream_news": [
        "new york times", "nytimes.com", "reuters", "associated press",
        "ap news", "bbc", "the guardian", "washington post", "wall street journal",
        "wsj", "cnn", "nbc news", "abc news", "cbs news", "the economist",
        "financial times", "ft.com", "axios",
    ],
    "social_media": [
        "twitter", "x.com", "@realdonaldtrump", "tweet from", "instagram",
        "facebook post", "tiktok", "youtube video", "reddit thread",
        "discord announcement", "telegram channel",
    ],
    "on_chain": [
        "block explorer", "etherscan", "polygonscan", "block height",
        "transaction hash", "wallet balance", "smart contract",
        "uniswap pool", "chainlink oracle", "on-chain", "onchain",
        "mainnet", "blockchain price",
    ],
    "decentralized_source": [
        "ipfs", "arweave", "decentralized storage", "filecoin",
        "decentralized oracle", "umadefi", "polymarket community",
        "discretion of polymarket", "discretion of the polymarket",
    ],
    "discretionary": [
        "sole discretion", "at the discretion of", "markets integrity committee",
        "the mic ", "resolution committee", "manual review",
        "subjective", "ambiguous", "good faith",
    ],
    # Crypto/DeFi-specific price sources
    "crypto_oracle": [
        "coinmarketcap", "coinbase", "binance", "kraken", "ftx",
        "coingecko", "coindesk",
    ],
}


def classify_source(text: str) -> set[str]:
    """Return the set of source categories matched by keywords in the
    market's description. Multi-label so we can study mutually exclusive
    subsets later."""
    low = (text or "").lower()
    matches = set()
    for category, keywords in SOURCE_KEYWORDS.items():
        for kw in keywords:
            if kw in low:
                matches.add(category)
                break
    return matches


# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------


def fetch_polymarket_resolved(n_target: int, force: bool = False) -> list[dict]:
    cache = CACHE / f"polymarket_resolved_{n_target}.json"
    if cache.exists() and not force:
        return json.loads(cache.read_text())
    print(f"  fetching ~{n_target} resolved Polymarket markets...", flush=True)
    out: list[dict] = []
    offset = 0
    page_size = 100
    while len(out) < n_target:
        url = (
            f"https://gamma-api.polymarket.com/markets"
            f"?limit={page_size}&offset={offset}&closed=true"
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
            out.append({
                "id": m.get("id"),
                "question": m.get("question") or "",
                "description": m.get("description") or "",
                "category": m.get("category") or "uncategorized",
                "end_date": m.get("endDate"),
                "closed_time": m.get("closedTime"),
                "volume": float(m.get("volumeNum") or 0),
                "uma_resolution_statuses": uma,
                "outcomes": m.get("outcomes"),
            })
        offset += page_size
        if len(page) < page_size:
            break
        if offset % 500 == 0:
            print(f"    fetched {len(out)}...", flush=True)
        time.sleep(0.2)
    cache.write_text(json.dumps(out))
    return out


# ---------------------------------------------------------------------------
# Delay calculation
# ---------------------------------------------------------------------------


def _parse_dt(s: str | None) -> datetime | None:
    if not s:
        return None
    s = s.strip()
    # Polymarket uses two formats: '2020-11-04T00:00:00Z' and '2020-11-02 16:31:01+00'
    for fmt in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S.%fZ",
                "%Y-%m-%d %H:%M:%S+00", "%Y-%m-%d %H:%M:%S.%f+00"):
        try:
            dt = datetime.strptime(s, fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except ValueError:
            continue
    # ISO 8601 fallback
    try:
        s2 = s.rstrip("Z").replace(" ", "T")
        return datetime.fromisoformat(s2).replace(tzinfo=timezone.utc) \
            if "+" not in s and "-" not in s.split("T")[-1] \
            else datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        return None


def delay_hours(market: dict) -> float | None:
    end = _parse_dt(market.get("end_date"))
    closed = _parse_dt(market.get("closed_time"))
    if end is None or closed is None:
        return None
    return (closed - end).total_seconds() / 3600.0


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------


def kruskal_wallis(groups: list[np.ndarray]) -> tuple[float, float]:
    """Kruskal-Wallis H test on a list of 1D arrays. Returns (H, p) using
    chi-squared approximation with k-1 degrees of freedom."""
    from math import erf, sqrt
    # Rank pooled data
    all_vals = np.concatenate(groups)
    # Use scipy-style ranks (average for ties)
    order = np.argsort(all_vals, kind="mergesort")
    ranks = np.empty(len(all_vals))
    ranks[order] = np.arange(1, len(all_vals) + 1, dtype=float)
    # Average ranks within ties
    sorted_vals = all_vals[order]
    i = 0
    while i < len(sorted_vals):
        j = i + 1
        while j < len(sorted_vals) and sorted_vals[j] == sorted_vals[i]:
            j += 1
        if j - i > 1:
            mean_rank = ranks[order[i:j]].mean()
            ranks[order[i:j]] = mean_rank
        i = j

    n = len(all_vals)
    idx = 0
    rank_groups = []
    for g in groups:
        rank_groups.append(ranks[idx:idx + len(g)])
        idx += len(g)
    H = (12.0 / (n * (n + 1))) * sum(
        len(rg) * (rg.mean() - (n + 1) / 2) ** 2 for rg in rank_groups
    )
    # Chi-squared survival function approximation (Wilson-Hilferty)
    k = len(groups)
    df = k - 1
    # df=k-1, chi-sq survival via Wilson-Hilferty
    if df <= 0:
        return H, 1.0
    z = ((H / df) ** (1.0 / 3.0) - (1 - 2 / (9 * df))) / math.sqrt(2 / (9 * df))
    # Standard normal survival
    p_one_sided = 0.5 * (1 - erf(z / sqrt(2)))
    return H, max(min(p_one_sided, 1.0), 0.0)


def bootstrap_median_diff(
    a: np.ndarray, b: np.ndarray, n_boot: int = 2000, seed: int = 42,
) -> dict:
    rng = np.random.default_rng(seed)
    obs = float(np.median(a) - np.median(b))
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        a_s = rng.choice(a, len(a), replace=True)
        b_s = rng.choice(b, len(b), replace=True)
        diffs[i] = np.median(a_s) - np.median(b_s)
    return {
        "obs_median_diff": obs,
        "ci_95": (float(np.quantile(diffs, 0.025)),
                  float(np.quantile(diffs, 0.975))),
        "se": float(np.std(diffs)),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-target", type=int, default=2000,
                    help="Target number of resolved markets to fetch")
    ap.add_argument("--force-refetch", action="store_true")
    args = ap.parse_args()

    print("\n=== C8 verification: Polymarket resolution-source category"
          " → delay & dispute ===")
    print(f"target: {args.n_target} resolved markets")
    print("stop rule: <30 'official-API' or <30 'unstable' (decentralized,"
          " social, discretionary) markets → abandon")

    print("\n--- fetching ---")
    markets = fetch_polymarket_resolved(args.n_target, force=args.force_refetch)
    print(f"  total resolved Polymarket markets: {len(markets)}")

    # Filter to those with usable delay + non-empty description
    usable = []
    for m in markets:
        d = delay_hours(m)
        if d is None:
            continue
        if not (m.get("description") or "").strip():
            continue
        m["_delay_hours"] = d
        m["_sources"] = classify_source(m.get("description") or "")
        m["_has_dispute"] = bool(m.get("uma_resolution_statuses"))
        usable.append(m)
    print(f"  usable markets (have endDate, closedTime, description): {len(usable)}")

    # Tally source category presence
    print("\n--- source-category coverage (each market may match >=1) ---")
    cat_counts = defaultdict(int)
    for m in usable:
        if not m["_sources"]:
            cat_counts["(none matched)"] += 1
        for s in m["_sources"]:
            cat_counts[s] += 1
    for cat in sorted(cat_counts, key=lambda c: -cat_counts[c]):
        print(f"  {cat:<22} {cat_counts[cat]:>5d}")

    # Build mutually-exclusive primary category by priority for the main
    # comparison: discretionary > decentralized > social_media > on_chain
    # > crypto_oracle > official_api > mainstream_news > (none).
    priority = [
        "discretionary", "decentralized_source", "social_media", "on_chain",
        "crypto_oracle", "official_api", "mainstream_news",
    ]
    primary_buckets = defaultdict(list)
    for m in usable:
        chosen = "uncategorized"
        for p in priority:
            if p in m["_sources"]:
                chosen = p
                break
        m["_primary"] = chosen
        primary_buckets[chosen].append(m)

    print("\n--- primary (priority-ordered) buckets ---")
    for p in priority + ["uncategorized"]:
        n = len(primary_buckets.get(p, []))
        if n == 0:
            continue
        delays = np.array([m["_delay_hours"] for m in primary_buckets[p]])
        disp = np.mean([m["_has_dispute"] for m in primary_buckets[p]])
        print(f"  {p:<22} n={n:>5d}  median_delay_h={np.median(delays):>10.2f}  "
              f"mean_delay_h={np.mean(delays):>10.2f}  dispute_rate={disp:.3f}")

    # Stop-rule check: need at least 30 "official_api" and 30 in
    # any "unstable" bucket (social, decentralized, discretionary)
    n_official = len(primary_buckets.get("official_api", []))
    unstable_sets = ["social_media", "decentralized_source", "discretionary"]
    n_unstable = sum(len(primary_buckets.get(u, [])) for u in unstable_sets)
    print(f"\nstop-rule check: official_api={n_official}  unstable={n_unstable}")
    if n_official < 30 or n_unstable < 30:
        print("\n=== verdict: STOP-RULE ABORT — insufficient examples (need ≥30 each)")
        return 0

    # Kruskal-Wallis: do delays differ across categories?
    comparison_buckets = [
        b for b in primary_buckets
        if len(primary_buckets[b]) >= 30 and b != "uncategorized"
    ]
    groups = [
        np.array([m["_delay_hours"] for m in primary_buckets[b]])
        for b in comparison_buckets
    ]
    H, p_kw = kruskal_wallis(groups)
    print(f"\n--- Kruskal-Wallis test for delay across {len(comparison_buckets)}"
          f" categories ---")
    for b, g in zip(comparison_buckets, groups):
        print(f"  {b:<22} n={len(g):>5d}  median_h={np.median(g):>10.2f}")
    print(f"  H = {H:.2f},  approx p = {p_kw:.4f}")

    # Dispute rate by category — Fisher-exact-style 2x2 for "unstable vs
    # official-api", computed as observed rate diff with bootstrap CI.
    print("\n--- dispute rate analysis ---")
    official = primary_buckets.get("official_api", [])
    unstable = []
    for u in unstable_sets:
        unstable.extend(primary_buckets.get(u, []))
    off_disp = np.array([1 if m["_has_dispute"] else 0 for m in official])
    uns_disp = np.array([1 if m["_has_dispute"] else 0 for m in unstable])
    print(f"  official_api    n={len(off_disp):>5d}  dispute_rate={off_disp.mean():.4f}")
    print(f"  any unstable    n={len(uns_disp):>5d}  dispute_rate={uns_disp.mean():.4f}")
    diff = bootstrap_median_diff(uns_disp.astype(float), off_disp.astype(float))
    print(f"  diff (unstable - official) = {diff['obs_median_diff']:+.4f}  "
          f"95% CI {diff['ci_95']}  SE {diff['se']:.4f}")

    # Pairwise median delay: unstable vs official
    off_delays = np.array([m["_delay_hours"] for m in official])
    uns_delays = np.array([m["_delay_hours"] for m in unstable])
    delay_diff = bootstrap_median_diff(uns_delays, off_delays)
    print("\n--- delay (hours): unstable vs official ---")
    print(f"  official_api    n={len(off_delays):>5d}  median_h={np.median(off_delays):>+10.2f}  "
          f"P75-P25={np.quantile(off_delays, 0.75) - np.quantile(off_delays, 0.25):>10.2f}")
    print(f"  any unstable    n={len(uns_delays):>5d}  median_h={np.median(uns_delays):>+10.2f}  "
          f"P75-P25={np.quantile(uns_delays, 0.75) - np.quantile(uns_delays, 0.25):>10.2f}")
    print(f"  median_diff (unstable - official) = {delay_diff['obs_median_diff']:+.2f}h  "
          f"95% CI ({delay_diff['ci_95'][0]:+.2f}, {delay_diff['ci_95'][1]:+.2f})")

    # Verdict
    print("\n=== verdict ===")
    ratio = (uns_disp.mean() / off_disp.mean()) if off_disp.mean() > 0 else float("inf")
    median_uns = float(np.median(uns_delays))
    median_off = float(np.median(off_delays))
    abs_gap_pp = (uns_disp.mean() - off_disp.mean()) * 100  # percentage points
    ci_excludes_zero = delay_diff["ci_95"][0] > 0 or delay_diff["ci_95"][1] < 0
    dispute_ci_excludes_zero = diff["ci_95"][0] > 0 or diff["ci_95"][1] < 0

    print(f"  unstable/official dispute rate ratio: {ratio:.2f}x")
    print(f"  unstable - official dispute absolute gap: {abs_gap_pp:+.2f} pp")
    print(f"  delay median diff (unstable - official): {median_uns - median_off:+.2f}h")
    print(f"  delay CI excludes 0?    {ci_excludes_zero}")
    print(f"  dispute CI excludes 0?  {dispute_ci_excludes_zero}")
    print(f"  Kruskal-Wallis p:       {p_kw:.4f}")

    # Stop rule from recipe: risk ratio >= 2.0 and absolute gap >= +5 pp
    # for dispute; OR a meaningful delay gap with CI excluding zero.
    decisive_positive_dispute = (ratio >= 2.0 and abs_gap_pp >= 5.0
                                 and dispute_ci_excludes_zero)
    decisive_positive_delay = (abs(median_uns - median_off) >= 24
                               and ci_excludes_zero)
    if decisive_positive_dispute or decisive_positive_delay:
        print(
            "\nCLEAN-POSITIVE — resolution-source category predicts adverse"
            "\n  resolution behavior on Polymarket. Specifically:"
        )
        if decisive_positive_dispute:
            print(
                f"    Unstable-source markets show {ratio:.2f}x the dispute"
                f"\n    rate of official-API-source markets ({abs_gap_pp:+.1f}pp gap),"
                "\n    bootstrap CI excludes zero."
            )
        if decisive_positive_delay:
            print(
                f"    Median resolution delay differs by {median_uns - median_off:+.1f}h,"
                "\n    bootstrap CI excludes zero."
            )
        print("  Resolution-source choice is a measurable, actionable lever for"
              "\n  market-design reliability.")
    elif p_kw < 0.05 and not (dispute_ci_excludes_zero or ci_excludes_zero):
        print(
            "\nMIXED — Kruskal-Wallis sees some across-category structure"
            f" (p={p_kw:.4f})"
            "\n  but the headline unstable-vs-official contrast does not"
            " clear the recipe's"
            "\n  effect-size floor (2x dispute, +5pp, or 24h delay gap)."
            " Worth deeper investigation."
        )
    else:
        print(
            "\nCLEAN-NEGATIVE — resolution-source category does NOT predict"
            "\n  adverse resolution behavior at the recipe's effect-size"
            "\n  thresholds. Polymarket's resolution mechanism (UMA + community"
            "\n  review) appears to neutralize source-stability differences,"
            "\n  OR the test lacks power to detect the postulated effect."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
