"""Citation check on the step3 KV-cache experiment.

Adapted from step2_citation_check.py — same logic, different corpus path."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parent.parent / "data" / "step3_experiment"


def normalize_title(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def title_match(claim_title: str, corpus_papers: list[dict]) -> dict | None:
    cn = normalize_title(claim_title)
    if len(cn) < 10:
        return None
    cw = set(cn.split())
    best = None
    best_jacc = 0.0
    for p in corpus_papers:
        pn = normalize_title(p["title"])
        if cn in pn or pn in cn:
            return p
        pw = set(pn.split())
        if cw and pw:
            jacc = len(cw & pw) / len(cw | pw)
            if jacc > best_jacc:
                best_jacc = jacc
                best = p
    return best if best_jacc >= 0.45 else None


# Codex citation patterns
PAT_ITALIC = re.compile(r'\*(?P<title>[^*\n]{15,200})\*\s*\((?P<year>(19|20)\d{2})\)')
PAT_BOLD = re.compile(r'\*\*(?P<title>[^*\n]{15,200})\*\*\s*\((?P<year>(19|20)\d{2})\)')
PAT_QUOTED = re.compile(r'["""\'](?P<title>[^"""\']{15,200})["""\']\s*\((?P<year>(19|20)\d{2})\)')


def extract_citations(text: str) -> list[dict]:
    seen = set()
    out = []
    for pat in (PAT_BOLD, PAT_ITALIC, PAT_QUOTED):
        for m in pat.finditer(text):
            title = m.group("title").strip().strip("*\"'")
            year = int(m.group("year"))
            key = (normalize_title(title), year)
            if key in seen:
                continue
            seen.add(key)
            out.append({"title": title, "year": year})
    return out


def main() -> int:
    with (EXP_DIR / "corpus.json").open() as f:
        corp = json.load(f)
    papers = corp["side_a"] + corp["side_b"]
    years = {p["year"] for p in papers if p.get("year")}
    print(f"Corpus: {len(papers)} papers, years: {sorted(years)}\n")

    text = (EXP_DIR / "codex_response.txt").read_text()
    cites = extract_citations(text)

    print(f"Citations extracted: {len(cites)}\n")
    confab = []
    for c in cites:
        match = title_match(c["title"], papers)
        if match is None:
            print(f"  ✗ CONFABULATED ({c['year']}) {c['title'][:80]}")
            confab.append(c)
        elif match.get("year") != c["year"]:
            print(f"  ? YEAR MISMATCH ({c['year']}) {c['title'][:80]}")
            print(f"      corpus has ({match['year']}) {match['title'][:70]}")
        else:
            print(f"  ✓ IN_CORPUS ({c['year']}) {c['title'][:80]}")

    all_years_in_codex = sorted({int(m.group("year")) for m in re.finditer(r'\((?P<year>(19|20)\d{2})\)', text)})
    bad_years = [y for y in all_years_in_codex if y not in years]
    print(f"\nAll year mentions in Codex output: {all_years_in_codex}")
    print(f"Years not in corpus: {bad_years}")

    verdict = "PASS — all citations grounded" if not confab and not bad_years else "FAIL — confabulation detected"
    print(f"\nVERDICT: {verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
