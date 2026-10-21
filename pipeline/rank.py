"""Layer 4a — score discovered bridges to surface top-N for the digest.

Score = weighted combination of:
  - Codex confidence (1-10 self-rating)        weight 0.30
  - Citation grounding (1 if all grounded)     weight 0.25
  - Cross-cite density (inverse: lower better) weight 0.20
  - Mechanism specificity (keyword count)      weight 0.15
  - Corpus completeness (both sides full)      weight 0.10

Weights are starting guesses; tune after the first batch shows what
correlates with "actually interesting."
"""

from __future__ import annotations

import re

# Heuristic for "mechanism specificity" — count of named mechanisms /
# techniques / specific terms in the response. The more concrete, the higher.
SPECIFICITY_TERMS = [
    r"\balgorithm\b", r"\bpolicy\b", r"\bprotocol\b", r"\bmechanism\b",
    r"\binvariant\b", r"\bbound\b", r"\bcomplexity\b", r"\bdistribution\b",
    r"\bfunction\b", r"\bclass(?:ifier|es)?\b", r"\bmodel\b", r"\bsystem\b",
    r"\bschedul", r"\bcache\b", r"\beviction\b", r"\bbarrier\b",
    r"\bcommit\b", r"\bverif", r"\bquery\b", r"\bindex\b",
]
_SPEC_PATS = [re.compile(p, re.IGNORECASE) for p in SPECIFICITY_TERMS]


def specificity_score(response: str) -> float:
    """0–1 normalized count of specificity terms (cap at 15 distinct hits)."""
    hits = sum(1 for p in _SPEC_PATS if p.search(response))
    return min(hits, 15) / 15.0


def score_bridge(discovery_result: dict, density_pair: dict) -> dict:
    """Compute weighted score. Returns the discovery_result enriched with `score`."""
    if discovery_result.get("status") != "complete":
        return {**discovery_result, "score": 0.0, "score_breakdown": None}

    confidence = discovery_result.get("confidence") or 5
    confidence_n = max(0.0, min(1.0, confidence / 10.0))

    check = discovery_result.get("citation_check") or {}
    grounding = 1.0 if check.get("passes") else 0.0

    max_ratio = density_pair.get("max_ratio", 0.5)
    # Inverse density: 0% → 1.0, 10% → 0.5, 20% → 0.0
    density_n = max(0.0, 1.0 - (max_ratio / 0.20))

    # Specificity: need to read the response
    response_path = discovery_result.get("response_path")
    if response_path:
        try:
            from pathlib import Path
            response = Path(response_path).read_text()
            specificity_n = specificity_score(response)
        except OSError:
            specificity_n = 0.0
    else:
        specificity_n = 0.0

    # Completeness: full corpus on both sides
    a = discovery_result.get("side_a_count", 0)
    b = discovery_result.get("side_b_count", 0)
    completeness_n = (min(a, 8) + min(b, 8)) / 16.0

    breakdown = {
        "confidence": (0.30, confidence_n),
        "grounding": (0.25, grounding),
        "density": (0.20, density_n),
        "specificity": (0.15, specificity_n),
        "completeness": (0.10, completeness_n),
    }
    total = sum(w * v for (w, v) in breakdown.values())

    return {
        **discovery_result,
        "score": round(total, 4),
        "score_breakdown": {k: {"weight": w, "value": round(v, 3)}
                            for k, (w, v) in breakdown.items()},
    }
