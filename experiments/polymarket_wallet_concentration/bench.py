"""C4 verification — compose run 20260524-052010 (prediction markets).

C4's synthesis claim:

    P158's decentralization-concentration measurements + P134's
    higher-order interaction model + P145's decentralized-exchange
    architecture review yields an on-chain concentration audit for
    Polymarket trading behavior. The falsifiable claim is: Polygon-
    address participation in Polymarket markets is more concentrated
    and coalition-like than pairwise wallet-volume graphs suggest,
    with hypergraph clusters of addresses repeatedly co-participating
    across related markets explaining liquidity and price-move timing
    better than simple holder-count or Gini measures.

First on-chain verification in the track record. Uses Polymarket's
public data-api (no API key required, no Graph subgraph needed):
  - gamma-api.polymarket.com/markets — market metadata + conditionId
  - data-api.polymarket.com/holders?market=<cid>&limit=N — top-N
    holder distribution per outcome token
  - data-api.polymarket.com/trades?market=<cid>&limit=N — recent
    trade history per market

Falsification target. The composition makes TWO claims:
  (A) Polymarket trading is concentrated (top wallets dominate).
  (B) HYPERGRAPH co-participation explains liquidity/price-move
      timing BETTER than simple Gini/holder-count.

Building the full hypergraph predictor is multi-week per L4. The
cheapest decisive test for (B) is to ask whether SIMPLE concentration
metrics already predict market outcomes; if they do, the composition's
"hypergraph adds explanatory power" claim has a high bar to clear.

For (A) and the co-participation premise we measure:
  1. Concentration metrics per market: Gini, top-10 share, HHI,
     holder count, holder count above threshold.
  2. Top-K wallet overlap between politically-related vs unrelated
     market pairs — tests whether co-participation is structured
     beyond random.
  3. Correlation: do concentration metrics predict (i) total
     volume, (ii) closure-delay, (iii) trade-count, (iv) price
     volatility from trade history?

Stop rule:
  - If holder distributions are uniformly diffuse (Gini < 0.5 across
    all markets), C4's "concentration" frame is dead.
  - If top-K overlap between related-market pairs is at-or-below
    expected random overlap, the "coalition" premise is dead.
  - If simple concentration metrics predict outcomes with non-trivial
    R² (>= 0.10 held-out), the hypergraph claim's structural bar is
    raised: the composition must explain *additional* variance not
    captured by simple metrics.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import requests
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

CACHE = Path(__file__).parent / "_cache"
CACHE.mkdir(exist_ok=True)


# --------------------------------------------------------------------------
# Data fetching (cached)
# --------------------------------------------------------------------------

def _cached_get(url: str, key: str, ttl_days: int = 30) -> dict | list | None:
    cache = CACHE / f"{key}.json"
    if cache.exists():
        return json.loads(cache.read_text())
    try:
        r = requests.get(url, timeout=20)
        if r.status_code != 200:
            return None
        data = r.json()
    except Exception as e:
        print(f"  fetch error {url}: {e}", flush=True)
        return None
    cache.write_text(json.dumps(data))
    return data


def fetch_market_detail(market_id: int) -> dict | None:
    return _cached_get(
        f"https://gamma-api.polymarket.com/markets/{market_id}",
        f"market_detail_{market_id}",
    )


def fetch_holders(condition_id: str, limit: int = 500) -> list | None:
    data = _cached_get(
        f"https://data-api.polymarket.com/holders?market={condition_id}&limit={limit}",
        f"holders_{condition_id[:10]}_{limit}",
    )
    if isinstance(data, list):
        return data
    return None


def fetch_trades(condition_id: str, limit: int = 500) -> list | None:
    data = _cached_get(
        f"https://data-api.polymarket.com/trades?market={condition_id}&limit={limit}",
        f"trades_{condition_id[:10]}_{limit}",
    )
    if isinstance(data, list):
        return data
    return None


# --------------------------------------------------------------------------
# Concentration metrics
# --------------------------------------------------------------------------

def gini(values: list[float]) -> float:
    """Standard Gini coefficient on non-negative values."""
    if not values:
        return 0.0
    arr = np.array(sorted(values), dtype=float)
    n = len(arr)
    if n == 0 or arr.sum() == 0:
        return 0.0
    ranks = np.arange(1, n + 1)
    return float((2.0 * (ranks * arr).sum() - (n + 1) * arr.sum()) / (n * arr.sum()))


def hhi(values: list[float]) -> float:
    """Herfindahl-Hirschman Index — sum of squared shares (0 to 1)."""
    if not values:
        return 0.0
    total = sum(values)
    if total == 0:
        return 0.0
    return float(sum((v / total) ** 2 for v in values))


def top_k_share(values: list[float], k: int) -> float:
    if not values:
        return 0.0
    arr = sorted(values, reverse=True)
    total = sum(arr)
    if total == 0:
        return 0.0
    return float(sum(arr[:k]) / total)


def aggregate_token_distributions(holders_payload: list) -> dict:
    """Polymarket binary markets have 2 outcome tokens (YES + NO).
    For concentration we use the COMBINED distribution: each wallet's
    total holding across both tokens (so a wallet that holds equal
    YES + NO counts once with the sum)."""
    by_wallet: dict[str, float] = {}
    n_tokens = len(holders_payload)
    for token in holders_payload:
        for h in token.get("holders", []):
            wallet = h.get("proxyWallet") or h.get("user") or h.get("address")
            if not wallet:
                continue
            try:
                amt = float(h.get("amount", 0))
            except (ValueError, TypeError):
                amt = 0.0
            by_wallet[wallet] = by_wallet.get(wallet, 0.0) + amt
    return {
        "n_tokens": n_tokens,
        "by_wallet": by_wallet,
        "n_holders_combined": len(by_wallet),
    }


# --------------------------------------------------------------------------
# Market outcome features
# --------------------------------------------------------------------------

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


def market_outcomes(market: dict, trades: list | None) -> dict:
    """Extract outcome variables from a market + recent trades."""
    out = {
        "volume": float(market.get("volumeNum") or market.get("volume") or 0),
        "delay_h": None,
        "trade_count": len(trades) if trades else 0,
        "trade_price_std": None,
        "trade_size_mean": None,
    }
    end = _parse_dt(market.get("endDate") or market.get("end_date"))
    closed = _parse_dt(market.get("closedTime") or market.get("closed_time"))
    if end is not None and closed is not None:
        out["delay_h"] = (closed - end).total_seconds() / 3600.0
    if trades:
        prices = []
        sizes = []
        for t in trades:
            try:
                prices.append(float(t.get("price", 0)))
                sizes.append(float(t.get("size", 0)))
            except (ValueError, TypeError):
                continue
        if len(prices) >= 2:
            out["trade_price_std"] = float(np.std(prices))
        if sizes:
            out["trade_size_mean"] = float(np.mean(sizes))
    return out


# --------------------------------------------------------------------------
# Bench
# --------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--current-cache",
        default=str(Path(__file__).resolve().parents[1]
                    / "polymarket_resolution_source/_cache/polymarket_current_2500_2024-01-01.json"),
    )
    ap.add_argument("--n-markets", type=int, default=50)
    ap.add_argument("--min-volume", type=float, default=1_000_000.0)
    ap.add_argument("--holder-limit", type=int, default=500)
    ap.add_argument("--trade-limit", type=int, default=500)
    args = ap.parse_args()

    print("\n=== C4 verification — Polymarket wallet concentration ===")
    print(f"sampling top {args.n_markets} by volume (min ${args.min_volume:,.0f})")

    base = json.load(open(args.current_cache))
    candidates = [m for m in base if m.get("volume", 0) >= args.min_volume]
    candidates.sort(key=lambda m: -m.get("volume", 0))
    candidates = candidates[: args.n_markets]
    print(f"candidates: {len(candidates)} markets above volume floor\n")

    rows = []
    politics_markets = []

    for i, m in enumerate(candidates):
        detail = fetch_market_detail(m["id"])
        cid = (detail or {}).get("conditionId")
        if not cid:
            continue
        holders_payload = fetch_holders(cid, limit=args.holder_limit)
        if holders_payload is None:
            continue
        agg = aggregate_token_distributions(holders_payload)
        amounts = list(agg["by_wallet"].values())
        if not amounts:
            continue
        g = gini(amounts)
        h_val = hhi(amounts)
        top10 = top_k_share(amounts, 10)
        top50 = top_k_share(amounts, 50)
        trades = fetch_trades(cid, limit=args.trade_limit)
        outs = market_outcomes(detail or m, trades)

        # Pick top-K wallets for overlap test
        top_wallets = set(
            w
            for w, _ in sorted(
                agg["by_wallet"].items(), key=lambda kv: -kv[1]
            )[:50]
        )

        is_politics = any(
            kw in (m.get("question") or "").lower()
            for kw in [
                "trump", "harris", "biden", "republican", "democrat",
                "election", "president", "senate", "popular vote",
            ]
        )
        if is_politics:
            politics_markets.append(
                {"id": m["id"], "question": m["question"], "top_wallets": top_wallets}
            )

        rows.append(
            {
                "id": m["id"],
                "question": (m.get("question") or "")[:60],
                "volume": float(m.get("volume", 0)),
                "n_holders": agg["n_holders_combined"],
                "gini": g,
                "hhi": h_val,
                "top10_share": top10,
                "top50_share": top50,
                "delay_h": outs["delay_h"],
                "trade_count": outs["trade_count"],
                "trade_price_std": outs["trade_price_std"],
                "is_politics": is_politics,
            }
        )
        if (i + 1) % 10 == 0:
            print(
                f"  processed {i+1}/{len(candidates)}: latest gini={g:.3f} top10={top10:.2%}",
                flush=True,
            )
        time.sleep(0.1)

    print(f"\nfinal sample: {len(rows)} markets")

    # ----- Concentration distributions -----
    ginis = [r["gini"] for r in rows]
    top10s = [r["top10_share"] for r in rows]
    hhis = [r["hhi"] for r in rows]
    holder_counts = [r["n_holders"] for r in rows]

    print("\n=== concentration statistics ===")
    print(f"  Gini:          median={float(np.median(ginis)):.3f}  "
          f"p10={float(np.percentile(ginis, 10)):.3f}  "
          f"p90={float(np.percentile(ginis, 90)):.3f}")
    print(f"  top-10 share:  median={float(np.median(top10s)):.2%}  "
          f"p10={float(np.percentile(top10s, 10)):.2%}  "
          f"p90={float(np.percentile(top10s, 90)):.2%}")
    print(f"  HHI:           median={float(np.median(hhis)):.4f}  "
          f"p10={float(np.percentile(hhis, 10)):.4f}  "
          f"p90={float(np.percentile(hhis, 90)):.4f}")
    print(f"  n_holders:     median={int(np.median(holder_counts))}  "
          f"p10={int(np.percentile(holder_counts, 10))}  "
          f"p90={int(np.percentile(holder_counts, 90))}")

    # ----- Pairwise top-50 wallet overlap (related-market premise) -----
    n_pol = len(politics_markets)
    if n_pol >= 2:
        print(f"\n=== top-50 wallet overlap (n={n_pol} politics markets) ===")
        overlaps_pol = []
        for i in range(n_pol):
            for j in range(i + 1, n_pol):
                a = politics_markets[i]["top_wallets"]
                b = politics_markets[j]["top_wallets"]
                if not a or not b:
                    continue
                ov = len(a & b)
                overlaps_pol.append(ov)
        all_top = set()
        for p in politics_markets:
            all_top |= p["top_wallets"]
        avg_universe = len(all_top)
        # Expected overlap under random pairing: |A|*|B|/universe
        avg_a = float(np.mean([len(p["top_wallets"]) for p in politics_markets]))
        expected = (avg_a * avg_a) / max(1, avg_universe)
        print(f"  observed mean pairwise overlap: {float(np.mean(overlaps_pol)):.1f}")
        print(f"  median overlap:                 {float(np.median(overlaps_pol)):.1f}")
        print(f"  unique-top-wallet universe:     {avg_universe}")
        print(f"  expected overlap if random:     {expected:.1f}")
        print(f"  ratio observed/expected:        {float(np.mean(overlaps_pol)) / expected:.2f}x")

    # ----- Concentration vs market outcomes -----
    valid = [r for r in rows if r["delay_h"] is not None]
    print(f"\n=== concentration → outcomes (n={len(valid)} markets with delay) ===")
    if len(valid) >= 20:
        X = np.array(
            [
                [r["gini"], r["top10_share"], r["hhi"], math.log10(r["n_holders"] + 1)]
                for r in valid
            ]
        )
        targets = {
            "log10(volume)": np.array([math.log10(r["volume"] + 1) for r in valid]),
            "log10(delay_h)": np.array(
                [math.log10(max(0.01, r["delay_h"])) for r in valid]
            ),
            "log10(trade_count)": np.array(
                [math.log10(r["trade_count"] + 1) for r in valid]
            ),
        }

        def cv_r2(X, y):
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            scores = []
            for tr, te in kf.split(X):
                sc = StandardScaler()
                X_tr = sc.fit_transform(X[tr])
                X_te = sc.transform(X[te])
                m = Ridge(alpha=1.0)
                m.fit(X_tr, y[tr])
                scores.append(r2_score(y[te], m.predict(X_te)))
            return float(np.mean(scores)), float(np.std(scores))

        for tgt_name, y in targets.items():
            r2m, r2s = cv_r2(X, y)
            # also test pearson of just top-10 share with target
            rho = float(
                np.corrcoef(
                    [r["top10_share"] for r in valid], y
                )[0, 1]
            )
            print(
                f"  predicting {tgt_name:<22s} R² (5-fold) = {r2m:+.3f} ± {r2s:.3f}   "
                f"ρ(top10, target) = {rho:+.3f}"
            )

    # ----- Verdict -----
    median_gini = float(np.median(ginis))
    median_top10 = float(np.median(top10s))
    print("\n=== verdict ===")
    if median_gini < 0.5:
        print(
            "DIFFUSE-MARKETS FORECLOSURE — median Gini is below 0.5; "
            "markets are not concentrated enough for any coalition narrative."
        )
    else:
        print(
            f"CONCENTRATION CONFIRMED — median Gini = {median_gini:.3f}, "
            f"median top-10 share = {median_top10:.2%}. Polymarket markets are "
            f"highly concentrated."
        )
        # Co-participation
        if n_pol >= 2 and overlaps_pol:
            ratio = float(np.mean(overlaps_pol)) / expected
            if ratio >= 2.0:
                print(
                    f"CO-PARTICIPATION STRUCTURED — top-50 overlap across "
                    f"politics-related markets is {ratio:.2f}x random; "
                    f"hypergraph premise has empirical support."
                )
            else:
                print(
                    f"CO-PARTICIPATION NOT STRUCTURED — overlap ratio {ratio:.2f}x "
                    f"random; hypergraph 'coalition' premise weak."
                )

    return 0


if __name__ == "__main__":
    sys.exit(main())
