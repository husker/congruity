"""Analyze the blind hit-rate ratings against the pre-registered plan.

Inputs:
  data/hitrate/key.json           : anon_id -> {polarity POS/NEG, confidence, ...}
  data/hitrate/ratings/r*.json    : each a list of {id, promise, positive_or_better, reason}

Metrics (as pre-registered in findings/expert-hit-rate-study-20260707.md):
  PRIMARY   1. inter-rater agreement: mean pairwise Spearman on promise;
               Fleiss' kappa on the binary calls.
            2. proxy-consensus (mean promise) vs pipeline confidence: Spearman.
  SUPPORTING (point estimate + bootstrap 95% CI, never the verdict):
            3. AUC(mean promise -> POS)         + CI
            4. AUC(confidence   -> POS)         + CI
            5. binary accuracy of proxy majority vs the base rate.
"""

from __future__ import annotations

import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parent.parent
H = REPO / "data" / "hitrate"


def load_ratings() -> dict[str, list[dict]]:
    out = {}
    for p in sorted((H / "ratings").glob("r*.json")):
        out[p.stem] = json.loads(p.read_text())
    return out


def fleiss_kappa(binary_by_rater: dict[str, dict[str, bool]], ids: list[str]) -> float:
    """Fleiss' kappa for N items x 2 categories (True/False) over R raters."""
    raters = list(binary_by_rater)
    R = len(raters)
    counts = np.zeros((len(ids), 2))  # col0 = #True, col1 = #False
    for i, cid in enumerate(ids):
        for rk in raters:
            v = binary_by_rater[rk].get(cid)
            if v is True:
                counts[i, 0] += 1
            elif v is False:
                counts[i, 1] += 1
    n = counts.sum(axis=1)
    valid = n == R
    counts = counts[valid]
    if len(counts) == 0:
        return float("nan")
    P_i = (counts**2).sum(axis=1) - R
    P_i = P_i / (R * (R - 1))
    P_bar = P_i.mean()
    p_j = counts.sum(axis=0) / (len(counts) * R)
    P_e = (p_j**2).sum()
    return float((P_bar - P_e) / (1 - P_e)) if (1 - P_e) > 1e-12 else float("nan")


def auc_ci(y: np.ndarray, score: np.ndarray, n_boot: int = 5000, seed: int = 7):
    auc = roc_auc_score(y, score)
    rng = np.random.default_rng(seed)
    boots = []
    idx = np.arange(len(y))
    for _ in range(n_boot):
        b = rng.choice(idx, size=len(idx), replace=True)
        if len(np.unique(y[b])) < 2:
            continue
        boots.append(roc_auc_score(y[b], score[b]))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(auc), float(lo), float(hi)


def main() -> int:
    key = json.loads((H / "key.json").read_text())
    ids = sorted(key)  # COMP-01..24
    y = np.array([1 if key[c]["polarity"] == "POS" else 0 for c in ids])
    conf = np.array([key[c]["confidence"] or 0 for c in ids], dtype=float)

    ratings = load_ratings()
    if len(ratings) < 2:
        print(f"only {len(ratings)} rater files in {H/'ratings'}; need >=2")
        return 1

    # per-rater promise vectors + binary maps, aligned to ids
    promise = {}
    binary = {}
    for rk, rows in ratings.items():
        pm = {r["id"]: float(r["promise"]) for r in rows if "promise" in r}
        bm = {r["id"]: bool(r["positive_or_better"]) for r in rows
              if "positive_or_better" in r}
        promise[rk] = np.array([pm.get(c, np.nan) for c in ids])
        binary[rk] = bm

    P = np.vstack([promise[rk] for rk in sorted(promise)])  # [R, 24]
    print(f"=== Expert hit-rate study — analysis (n={len(ids)}, "
          f"{int(y.sum())} POS / {len(y)-int(y.sum())} NEG, "
          f"{len(ratings)} raters) ===\n")

    # PRIMARY 1a: inter-rater agreement (mean pairwise Spearman on promise)
    pair_rhos = []
    for a, b in combinations(range(P.shape[0]), 2):
        rho = spearmanr(P[a], P[b], nan_policy="omit").correlation
        pair_rhos.append(rho)
    print("-- PRIMARY: inter-rater agreement --")
    print(f"  mean pairwise Spearman on promise: {np.nanmean(pair_rhos):+.3f} "
          f"(range {np.nanmin(pair_rhos):+.3f}..{np.nanmax(pair_rhos):+.3f})")

    # PRIMARY 1b: Fleiss kappa on binary
    kappa = fleiss_kappa(binary, ids)
    frac_pos = np.mean([np.mean([binary[rk].get(c, False) for rk in binary])
                        for c in ids])
    print(f"  Fleiss' kappa on binary positive_or_better calls: {kappa:+.3f}")
    print(f"  mean fraction of raters calling POS: {frac_pos:.2f} "
          f"(actual base rate {y.mean():.2f})")

    # consensus promise (mean across raters, ignoring nans)
    consensus = np.nanmean(P, axis=0)

    # PRIMARY 2: proxy consensus vs pipeline confidence
    rho_cf = spearmanr(consensus, conf).correlation
    print("\n-- PRIMARY: proxy consensus vs pipeline confidence --")
    print(f"  Spearman(mean promise, confidence): {rho_cf:+.3f}")

    # SUPPORTING: outcome AUCs with bootstrap CI
    print("\n-- SUPPORTING (noisy at this n; point est + 95% bootstrap CI) --")
    a_p, lo_p, hi_p = auc_ci(y, consensus)
    print(f"  AUC(proxy consensus -> POS): {a_p:.3f}  [{lo_p:.3f}, {hi_p:.3f}]")
    a_c, lo_c, hi_c = auc_ci(y, conf)
    print(f"  AUC(pipeline confidence -> POS): {a_c:.3f}  [{lo_c:.3f}, {hi_c:.3f}]")
    rho_out = spearmanr(consensus, y).correlation
    print(f"  Spearman(mean promise, outcome): {rho_out:+.3f}")

    # binary accuracy of proxy majority vs base rate
    maj = np.array([1 if np.mean([binary[rk].get(c, False) for rk in binary]) >= 0.5
                    else 0 for c in ids])
    acc = float((maj == y).mean())
    base = max(y.mean(), 1 - y.mean())
    print(f"  proxy-majority binary accuracy: {acc:.3f} "
          f"(majority-class base rate {base:.3f})")

    summary = {
        "n": len(ids), "n_pos": int(y.sum()), "n_raters": len(ratings),
        "inter_rater_mean_spearman": float(np.nanmean(pair_rhos)),
        "fleiss_kappa_binary": kappa,
        "proxy_vs_confidence_spearman": float(rho_cf),
        "auc_proxy": [a_p, lo_p, hi_p],
        "auc_confidence": [a_c, lo_c, hi_c],
        "spearman_proxy_outcome": float(rho_out),
        "proxy_majority_accuracy": acc, "base_rate": float(base),
        "mean_frac_pos_calls": float(frac_pos),
    }
    (H / "analysis.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {H/'analysis.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
