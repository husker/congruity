# C4 Verification — Polymarket Wallet Concentration & Co-participation

**Run:** `20260524-052010` (prediction markets, frontier framing)
**Composition:** C4 (3 papers, conf 8/10, L1/L2/L3-weak/L4-PURELY_TESTABLE)
**Verdict:** **partial-positive — concentration confirmed, co-participation structured, simple-metric predictive lift modest**
**Bench:** `experiments/polymarket_wallet_concentration/bench.py`
**Track record note:** First on-chain verification.

## Claim under test

> P158's decentralization-concentration measurements + P134's higher-
> order interaction model + P145's decentralized-exchange architecture
> review yields an on-chain concentration audit for Polymarket trading
> behavior. The falsifiable claim is: Polygon-address participation
> in Polymarket markets is more concentrated and coalition-like than
> pairwise wallet-volume graphs suggest, with hypergraph clusters of
> addresses repeatedly co-participating across related markets
> explaining liquidity and price-move timing better than simple
> holder-count or Gini measures.

## Falsification design

The composition makes two claims:

- **(A)** Polymarket trading is highly concentrated (top wallets dominate).
- **(B)** **Hypergraph co-participation** explains liquidity/price-move
  timing **better than simple Gini/holder-count**.

Building the full hypergraph predictor is multi-week per L4. The
cheapest decisive test for (B) is to ask whether **simple concentration
metrics already predict market outcomes** — if they do, the
composition's "hypergraph adds explanatory power" claim faces a high
bar. If they do *not*, the entire C4 framing has weak signal at the
upstream level regardless of hypergraph structure.

Plus the co-participation premise (top wallet overlap across related
markets vs random expectation) is testable cheaply and is upstream of
both A and B.

## Data source

First fully on-chain Congruity verification. Uses Polymarket's public
data APIs (no Graph API key, no UMA subgraph):

- `gamma-api.polymarket.com/markets/{id}` — market metadata + `conditionId`
- `data-api.polymarket.com/holders?market={conditionId}&limit=500` —
  top-500 wallet positions per outcome token (YES + NO)
- `data-api.polymarket.com/trades?market={conditionId}&limit=500` —
  recent trade history per market

Sample: top 50 markets by volume from the 2024+ cohort (min volume
$1M; 37/50 are politics-related — election, popular vote, candidate,
etc.).

## Results

### Concentration (Claim A)

Aggregated by wallet (each wallet's YES + NO holdings combined):

| metric | median | p10 | p90 |
|---|---|---|---|
| **Gini coefficient** | **0.981** | 0.912 | 0.994 |
| **top-10 share** | **94.79%** | 72.35% | 99.35% |
| HHI | 0.5495 | 0.2465 | 0.7501 |
| n_holders (unique wallets) | 387 | 224 | 620 |

Concentration is **extreme**. For context:

- A Gini of 0.5 already signals strong inequality; 0.981 means the
  top sliver of wallets effectively own the market.
- A top-10 share of 94.79% means **94.79% of all token exposure** on
  the median high-volume market is held by **just 10 wallets**.
- HHI of 0.55 is far above the 0.25 antitrust threshold for highly
  concentrated industries.

The L3 challenge that "addresses can be contracts, market-maker
infrastructure, sybil fragments, custodial/proxy accounts" is real —
some of these top-10 wallets are almost certainly market-maker proxies
or AMM contracts, not human traders. But the *raw* concentration
signal is unambiguous: a small number of address-level entities hold
the bulk of market exposure on every high-volume market sampled.

### Co-participation (premise B upstream)

Top-50 wallet overlap across n=37 politics-related markets:

|                                | value |
|---|---|
| observed mean pairwise overlap | 4.8   |
| median overlap                 | 3.0   |
| unique top-wallet universe     | 1,188 |
| expected overlap if random     | 2.1   |
| **observed / expected ratio**  | **2.26x** |

Observed overlap is **2.26× the random-baseline expectation**.
Co-participation is structured: top wallets in one politics market
have a meaningfully-elevated probability of also being top wallets in
another politics market. This is consistent with — but does not prove
— professional traders / market-maker concentration / coordinated
behavior. The composition's hypergraph premise has empirical support
at this upstream test.

### Concentration → market outcomes

Simple concentration features (Gini, top-10 share, HHI, log(n_holders))
predicting market outcomes via 5-fold CV Ridge regression:

| target | R² (5-fold) | ρ(top10, target) |
|---|---|---|
| log10(volume) | −0.188 ± 0.703 | +0.052 |
| **log10(delay_h)** | **+0.193 ± 0.215** | **+0.244** |
| log10(trade_count) | (artifact — see note) | — |

**Note on trade_count**: trade_count is capped at limit=500 in our
fetch, so most markets in the sample are at the boundary. The R² of
+1.000 in the raw output is a saturation artifact; it should not be
interpreted. Excluded from the verdict.

Real signal:

- **Volume is not predicted by concentration** (R² = −0.19, near-zero
  correlation). High-volume markets are not systematically more or
  less concentrated within the high-volume cohort.
- **Closure-delay is weakly predicted** (R² = +0.19, ρ ≈ +0.24).
  More-concentrated markets resolve slightly slower. Modest signal,
  consistent with the prior C8 finding that delay has multiple
  drivers and source category alone wasn't predictive.

## Verdict

**Partial-positive — concentration confirmed, co-participation
structured, simple-metric predictive lift modest.**

Same general shape as the Hamiltonian C1 finding (oracle-supported,
cheap-classifier-falsified):

| C4 claim | status |
|---|---|
| (A) Markets are concentrated | **Confirmed** at extreme levels (Gini 0.98) |
| Upstream: wallets co-participate across related markets | **Confirmed** at 2.26× random |
| (B) Simple Gini / holder-count predict market behavior | **Weak signal** (R² ≈ +0.19 for delay, near-zero for volume) |
| (B') Hypergraph features beat simple metrics | **Untested** (multi-week per L4); bar is set by the simple-metric R² ceiling |

Implications:

1. The composition's empirical premise (Polymarket is concentrated,
   wallets cluster) is **upheld at the on-chain data level**. This is
   the first compose-engine output where on-chain data validates a
   structural claim.
2. The "simple metrics already work" path that foreclosed C5/C11 etc.
   does NOT close C4 — simple Gini/top-10 alone have a low predictive
   ceiling (R² < 0.20). So the composition's hypergraph-vs-pairwise
   contrast has structural room to demonstrate value if implemented.
3. **But** the L3 challenges around entity resolution remain
   load-bearing — without distinguishing market-maker proxies from
   human traders, the 0.98 Gini may collapse partially. The
   composition needs an entity-resolution step before the hypergraph
   claim is operationally credible.

## Composition pieces in retrospect

- **P158 (governance concentration metrics)**: transfers to trading
  context cleanly at the descriptive level. Gini, HHI, and top-K
  share all reveal extreme concentration on Polymarket high-volume
  markets. The L3 "may not transfer from governance voting" challenge
  is *partially* addressed — descriptive metrics do transfer, but
  whether they carry the same *interpretive* weight (in voting,
  concentration → control; in trading, may → market-making) requires
  the entity-resolution step.
- **P134 (higher-order networks)**: the upstream co-participation
  test (2.26× random overlap) is consistent with the higher-order
  premise. Not enough to validate the *predictive* claim, but enough
  to validate the *existence* of co-participation structure that a
  hypergraph framework could plausibly capture.
- **P145 (DEX architecture)**: load-bearing at the data-access level.
  Polymarket's public on-chain trace plus public data-api endpoints
  are exactly the kind of public-data substrate P145 describes. The
  verification was operationally feasible because of this.

## Open follow-ups

1. **Entity resolution**: cluster known market-maker proxies (high-
   activity-burst patterns, contract addresses, repeated cross-market
   top-10 appearance) and recompute concentration on de-duplicated
   entities. If Gini drops below ~0.85, the "concentration" framing
   needs nuance.
2. **Larger sample**: 50 markets at the top of the volume distribution
   may overfit to the politics regime. Stratify by category (sports,
   crypto, tech, lifestyle) and check whether the pattern holds.
3. **Out-of-sample prediction**: temporal split (train on early-
   period markets, test on later) to handle market-maker churn /
   regime shifts. Current 5-fold CV doesn't model temporal leakage.
4. **The actual hypergraph predictor**: per L4, this is the 2-week
   experiment that would test the C4-distinctive claim. With the
   simple-metric R² ceiling now established at ~0.20, the bar for
   a meaningful hypergraph contribution is "explain > +0.10 additional
   R² over Gini + top-K + holder count".

## Reproducibility

```
uv run --with requests --with scikit-learn python \
    experiments/polymarket_wallet_concentration/bench.py \
    --n-markets 50 --min-volume 1000000
```

Cached fetches to `experiments/polymarket_wallet_concentration/_cache/`
(gitignored). Total API calls per market: 1 (gamma) + 1 (holders) +
1 (trades) = 3. Runs in ~3 minutes including the initial fetches.
