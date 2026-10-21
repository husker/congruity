# C8 Dispute Hypothesis — Follow-up Investigation

**Run:** follow-up to `findings/c8-polymarket-resolution-source-20260524.md`
**Verdict:** **still-inconclusive — data path enumerated, on-chain event scan required**
**Status:** documents the blocker; not a verification result

## Background

The original C8 verification tested whether Polymarket resolution-source
category predicts (i) **delay** and (ii) **disputes**. The delay test
returned clean-negative. The dispute test was inconclusive because the
`umaResolutionStatuses` (plural) array was empty for all 1,969 markets
fetched via Gamma API.

The handoff flagged the open question: query the UMA Optimistic
Oracle V2 subgraph on Polygon, or scrape per-market dispute view from
polymarket.com.

## What this follow-up adds

This is not a verification of the dispute hypothesis. It documents:

1. **What data IS reachable** without a Graph API key or HTML scraping
2. **Why those reachable surfaces don't carry a dispute signal**
3. **The minimum-viable path forward** for whoever picks this up next

## Findings

### Polymarket's `umaResolutionStatus` (singular) field exists but is uninformative

Investigating individual market detail (vs the list-mode endpoint
used in C8), each closed market exposes a **`umaResolutionStatus`**
field (singular). Tally across the current 2024+ cohort (n=100
sampled):

```
umaResolutionStatus distribution:
  'resolved': 100
```

Every single closed market is marked `'resolved'`. This is a *terminal*
state field — once a market completes the UMA process (whether through
clean propose-and-finalize or through propose-dispute-vote-finalize),
it reaches `'resolved'`. The field does not distinguish *how* the
market got there. The `umaResolutionStatuses` (plural) array that the
original C8 bench checked was empty for the same reason: Polymarket's
public API only exposes the *final* status, not the dispute history.

So the dispute hypothesis cannot be tested with the Polymarket Gamma
or data APIs alone. The hypothesis requires either:

- UMA Optimistic Oracle V2 contract event logs (`RequestPrice`,
  `ProposePrice`, **`DisputePrice`**, `Settle`) filtered by
  Polymarket's requester address(es) on Polygon
- A maintained subgraph indexing those events
- polymarket.com per-market dispute view (HTML, may rate-limit)

### Sub-question: does a "long-delay → dispute" proxy work?

A reasonable proxy: markets with delay >> the modal value (31h
from `polymarket-latency-current-cohort-20260524.md`) are likely
disputed, since UMA disputes add a 48-72h liveness window.

The original C8 finding tested this exact slicing and returned
clean-negative on delay overall — source category did not predict
delay even at the long-tail end. So the "long-delay → dispute"
proxy doesn't rescue the dispute test indirectly either.

### What's reachable

| surface | accessible? | dispute signal? |
|---|---|---|
| Polymarket Gamma `markets/{id}` (singular `umaResolutionStatus`) | Yes | No (always 'resolved') |
| Polymarket Gamma `markets/{id}` (plural `umaResolutionStatuses`) | Yes | No (always empty) |
| Polymarket data-api `/holders` | Yes | No |
| Polymarket data-api `/trades` | Yes | No |
| Polymarket data-api `/positions?user=X` | Yes (with user) | No |
| UMA OO V2 subgraph (hosted Graph) | **Deprecated** | Would have been yes |
| UMA OO V2 subgraph (decentralized Graph) | **Auth required** (API key) | Would be yes |
| Polygon RPC + raw event log scan | Yes (with rate limits) | **Yes** (canonical) |
| polymarket.com per-market HTML | Yes (with rate limits) | **Yes** (UI surface) |

The decentralized Graph requires a paid API key. The hosted service
endpoints UMA used to expose have returned 301 redirects since the
hosted-service deprecation.

### Minimum-viable path forward

1. Set up a Polygon archive-node RPC (Alchemy / Infura / etc. free
   tier should be enough for historical event scans on a single
   contract).
2. Filter `DisputePrice` events from UMA OO V2 (Polygon contract
   `0xeE3Afe347D5C74317041E2618C49534dAf887c24`), batched by block
   range, for Polymarket's requester address(es).
3. Match disputes to Polymarket markets via the `ancillaryData` →
   `questionID` mapping. Both fields are stored on-chain and visible
   in Polymarket Gamma market detail (as `questionID`).
4. Per-market dispute count = number of DisputePrice events keyed to
   that questionID.
5. Run the original C8 source-category Kruskal-Wallis test on the
   dispute count, exactly as the delay test was structured.

Effort estimate: ~1-2 days including rate-limit handling.

## Track record implication

The original C8 finding stays in INDEX as **clean-negative** because
its delay test returned a clean negative. The dispute sub-question is
**not** reclassified — it was inconclusive in the original finding
and remains inconclusive. This document just enumerates the blocker
and the unblock path so the next session can decide whether to invest
the 1-2 days.

## Reproducibility

No new bench code. The relevant exploration commands:

```python
import requests
r = requests.get("https://gamma-api.polymarket.com/markets?limit=100&closed=true&end_date_min=2024-01-01T00:00:00Z")
from collections import Counter
Counter(m.get("umaResolutionStatus") for m in r.json())
# → Counter({'resolved': 100})
```

```bash
# Both deprecated / require auth:
curl https://api.thegraph.com/subgraphs/name/umaprotocol/polygon-optimistic-oracle-v2
# → 301 Moved Permanently
curl https://gateway.thegraph.com/api/subgraphs/id/QmZ
# → auth error: missing authorization header
```
