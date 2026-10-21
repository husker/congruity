"""Transfer-distance instrument for the verification track record.

The goal congruity is actually for: surface a technique from a DISTANT field that
beats the destination field's own best-known method. This script measures the
*transfer distance* of each verified composition — how far apart the source
fields of its constituent papers are — and joins it to the empirical outcome.

Transfer distance = how cross-disciplinary the composition is:
  - n_source_topics : distinct OpenAlex topics the papers were drawn from
  - n_source_fields : distinct OpenAlex *fields* (broader than topics)
  - field_entropy   : Shannon entropy (bits) of the paper-field distribution
                      (0 = all one field; higher = drawn from many fields)

A high-transfer composition that beats baselines is the valuable case
(no single human spans those fields). A low-transfer one is within-field
engineering. The "beats the home-field SOTA" axis is judged separately in the
catalog (it is not in structured data); this script supplies the distance half.

Run:
    uv run python scripts/transfer_distance.py
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
COMPOSITIONS = REPO / "data" / "pipeline" / "compositions"
TOPIC_LISTS = REPO / "data" / "pipeline" / "topic_lists"
INDEX = REPO / "findings" / "INDEX.md"
RUN_ID_RE = re.compile(r"^\d{8}-\d{6}$")
CID_RE = re.compile(r"(C\d+[a-z]?)")


def build_topic_field_map() -> dict[str, str]:
    """topic display_name + topic id -> field display_name, from all topic lists."""
    m: dict[str, str] = {}
    for f in TOPIC_LISTS.glob("*.json"):
        try:
            topics = json.loads(f.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if not isinstance(topics, list):
            continue
        for t in topics:
            field = t.get("field")
            fname = field.get("display_name") if isinstance(field, dict) else str(field)
            if t.get("display_name"):
                m[t["display_name"]] = fname
            if t.get("id"):
                m[t["id"]] = fname
    return m


def load_corpus_fields(run_dir: Path, topic_field: dict[str, str]) -> dict[str, str]:
    """paper_id -> field, for one run (via source_topic)."""
    for name in ("corpus_validated.json", "corpus.json", "corpus_raw.json"):
        fp = run_dir / name
        if fp.exists():
            papers = json.loads(fp.read_text())
            out = {}
            for p in papers:
                fld = (topic_field.get(p.get("source_topic_id", ""))
                       or topic_field.get(p.get("source_topic", ""))
                       or p.get("source_topic", "?"))
                out[p.get("paper_id")] = fld
            return out
    return {}


def transfer_metrics(fields: list[str]) -> dict:
    fields = [f for f in fields if f and f != "?"]
    if not fields:
        return {"n_fields": 0, "field_entropy": 0.0, "fields": []}
    counts = Counter(fields)
    total = len(fields)
    ent = -sum((c / total) * math.log2(c / total) for c in counts.values())
    return {"n_fields": len(counts), "field_entropy": round(ent, 2),
            "fields": sorted(counts)}


def parse_index() -> list[dict]:
    rows = []
    for ln in INDEX.read_text().splitlines():
        if not ln.startswith("| ") or "---" in ln:
            continue
        c = [x.strip() for x in ln.strip().strip("|").split("|")]
        if len(c) < 4 or c[0] in ("run_id", ""):
            continue
        m = CID_RE.match(c[2])
        rows.append({"run_id": c[0], "experiment": c[1], "cid": m.group(1) if m else None,
                     "outcome": c[3]})
    return rows


def main() -> int:
    topic_field = build_topic_field_map()
    rows = parse_index()
    print(f"topic->field map: {len(topic_field)} entries\n")
    print(f"{'run_id':17} {'cid':4} {'#topics':>7} {'#fields':>7} {'entropy':>7}  outcome / fields")
    print("-" * 100)
    results = []
    for r in rows:
        run = r["run_id"]
        if not RUN_ID_RE.match(run):
            continue
        d = COMPOSITIONS / run
        sp = d / "state.json"
        if not sp.exists():
            continue
        comp = next((c for c in json.loads(sp.read_text()).get("compositions", [])
                     if c.get("id") == r["cid"]), None)
        if comp is None:
            continue
        pf = load_corpus_fields(d, topic_field)
        pids = [p.get("paper_id") for p in comp.get("papers", [])]
        topics_seen = {pf.get(pid) for pid in pids if pf.get(pid)}
        tm = transfer_metrics([pf.get(pid) for pid in pids])
        short = re.sub(r"\*\*", "", r["outcome"])[:32]
        results.append({**r, **tm, "n_topics": len(topics_seen)})
        print(f"{run:17} {r['cid'] or '-':4} {len(topics_seen):>7} {tm['n_fields']:>7} "
              f"{tm['field_entropy']:>7}  {short}  | {', '.join(tm['fields'])[:60]}")
    # Highlight: high-transfer wins
    print("\nHigh-transfer (>=2 fields) AND positive outcome — the valuable quadrant:")
    for r in results:
        if r["n_fields"] >= 2 and ("positive" in r["outcome"].lower()):
            print(f"  {r['run_id']} {r['cid']}: {r['n_fields']} fields, entropy {r['field_entropy']} "
                  f"— {r['experiment'][:40]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
