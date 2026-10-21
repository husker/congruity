"""Build the blinded rating dataset + scoring key for the expert hit-rate study.

Reuses outcome_calibration's INDEX join. Emits two files under
data/hitrate/:
  - blinded.json : anonymized composition content a rater sees (NO outcome,
                   confidence, verdict, run_id, or cid).
  - key.json     : anon_id -> {run_id, cid, polarity, outcome_label, confidence}
                   (used only at scoring time, never shown to raters).

Anonymization order is canonical (sorted by run_id, cid) so COMP-NN ids are
stable across rebuilds; per-rater shuffling happens at rating time.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.outcome_calibration import load_composition, parse_index  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "data" / "hitrate"


def paper_brief(p) -> dict:
    """A rater-visible paper summary (title + role/field), no ids that leak."""
    if isinstance(p, str):
        return {"title": p}
    if isinstance(p, dict):
        return {
            "title": p.get("title") or p.get("name") or "",
            "role": p.get("role") or p.get("contribution") or "",
            "field": p.get("field") or p.get("discipline") or "",
        }
    return {"title": str(p)}


def main() -> int:
    rows = parse_index()
    recs = []
    for r in rows:
        if r["polarity"] not in ("POS", "NEG"):
            continue
        comp = load_composition(r["run_id"], r["cid"])
        if comp is None:
            continue
        recs.append((r, comp))

    # canonical order for stable anon ids
    recs.sort(key=lambda rc: (rc[0]["run_id"], rc[0]["cid"] or ""))

    blinded = []
    key = {}
    for i, (r, c) in enumerate(recs, 1):
        anon = f"COMP-{i:02d}"
        papers = c.get("papers") or []
        blinded.append({
            "id": anon,
            "synthesis_claim": c.get("synthesis_claim") or "",
            "compatibility_argument": c.get("compatibility_argument") or "",
            "papers": [paper_brief(p) for p in papers],
            "covered_dimensions": c.get("covered_dimensions") or [],
            "open_questions": c.get("open_questions") or [],
        })
        key[anon] = {
            "run_id": r["run_id"],
            "cid": r["cid"],
            "polarity": r["polarity"],
            "outcome_label": r["outcome_label"],
            "confidence": c.get("confidence"),
        }

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "blinded.json").write_text(json.dumps(blinded, indent=2))
    (OUT / "key.json").write_text(json.dumps(key, indent=2))

    # per-rater shuffled variants (deterministic) so order effects don't
    # inflate inter-rater agreement; scoring joins by anon id regardless.
    import random
    for rk in range(1, 6):
        shuffled = blinded[:]
        random.Random(1000 + rk).shuffle(shuffled)
        (OUT / f"blinded_r{rk}.json").write_text(json.dumps(shuffled, indent=2))

    npos = sum(1 for a in key.values() if a["polarity"] == "POS")
    nneg = sum(1 for a in key.values() if a["polarity"] == "NEG")
    print(f"wrote {len(blinded)} blinded compositions ({npos} POS / {nneg} NEG)")
    print(f"  -> {OUT / 'blinded.json'}")
    print(f"  -> {OUT / 'key.json'}")
    # sanity: show one blinded record's field lengths + a claim sample
    b0 = blinded[0]
    print(f"\nsample {b0['id']}: claim_chars={len(b0['synthesis_claim'])} "
          f"compat_chars={len(b0['compatibility_argument'])} "
          f"papers={len(b0['papers'])} dims={len(b0['covered_dimensions'])}")
    print(f"  claim: {b0['synthesis_claim'][:200]!r}")
    print(f"  paper[0]: {b0['papers'][0] if b0['papers'] else None}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
