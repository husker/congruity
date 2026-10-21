"""Mechanical citation verification on Codex's responses.

Codex confabulated citations in both the clear and redacted runs. The doc's
verifier (V) in § 10.4 step 4 is supposed to catch exactly this. This script
is the first cut of that verifier, applied retrospectively to the Step 2
experiment outputs.

Method
------
1. Read the corpus (data/step2_experiment/corpus.json) — these are the only
   papers Codex was given.
2. Read each Codex response file.
3. Extract paper-title-like patterns: **"..." (YEAR)** or **"..."**, **...** (YEAR),
   plus any standalone (YEAR) mentions.
4. For each extracted citation, check:
   - Does the year appear among corpus years?
   - Does the title fuzzy-match a corpus title (case-insensitive substring)?
5. Flag each citation as IN_CORPUS / CONFABULATED / AMBIGUOUS.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

EXPERIMENT_DIR = Path(__file__).resolve().parent.parent / "data" / "step2_experiment"


def load_corpus() -> tuple[set[int], list[dict]]:
    with (EXPERIMENT_DIR / "corpus.json").open() as f:
        data = json.load(f)
    papers = data["pl"] + data["os"]
    years = {p["year"] for p in papers if p.get("year")}
    return years, papers


def normalize_title(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def title_match(claim_title: str, corpus_papers: list[dict]) -> dict | None:
    """Fuzzy match: claim is a substring of corpus title (or vice versa),
    after normalization."""
    claim_norm = normalize_title(claim_title)
    if len(claim_norm) < 10:
        return None
    for p in corpus_papers:
        corpus_norm = normalize_title(p["title"])
        if claim_norm in corpus_norm or corpus_norm in claim_norm:
            return p
        # Also accept significant overlap (Jaccard on words)
        cw = set(claim_norm.split())
        pw = set(corpus_norm.split())
        if cw and pw:
            jacc = len(cw & pw) / len(cw | pw)
            if jacc >= 0.6:
                return p
    return None


# Pattern 1: **"Title" (YEAR)** — bolded title in quotes with year
P1 = re.compile(r'\*\*[\*"\'""]*?(?P<title>[^"\']{15,200}?)[\*"\'""]*?\s*\(?(?P<year>(19|20)\d{2})\)?\*\*')
# Pattern 2: **Title** (YEAR) — bolded title not in quotes, year follows in parens
P2 = re.compile(r'\*\*(?P<title>[^*\n]{15,200})\*\*\s*\((?P<year>(19|20)\d{2})\)')
# Pattern 3: "Title" (YEAR) — quoted title with year
P3 = re.compile(r'[""\']\s*(?P<title>[^""\']{15,200})\s*[""\']\s*\((?P<year>(19|20)\d{2})\)')
# Pattern 4 (catchall, years only): \(YEAR\)
P_YEAR = re.compile(r'\((?P<year>(19|20)\d{2})\)')


def extract_citations(text: str) -> list[dict]:
    seen = set()
    cites: list[dict] = []
    for pat in (P1, P2, P3):
        for m in pat.finditer(text):
            title = re.sub(r'^[\*"\'""]+|[\*"\'""]+$', '', m.group("title")).strip()
            year = int(m.group("year"))
            key = (normalize_title(title), year)
            if key in seen:
                continue
            seen.add(key)
            cites.append({"title": title, "year": year})
    return cites


def check_file(label: str, path: Path, corpus_years: set[int],
               corpus_papers: list[dict]) -> dict:
    text = path.read_text()
    cites = extract_citations(text)

    # Year-only mentions (for flag of "model dropped a year that's not in corpus")
    year_mentions = sorted({int(m.group("year")) for m in P_YEAR.finditer(text)})

    results = []
    for c in cites:
        match = title_match(c["title"], corpus_papers)
        if match is not None:
            year_match = match.get("year") == c["year"]
            status = "IN_CORPUS" if year_match else "YEAR_MISMATCH"
            results.append({
                **c,
                "status": status,
                "matched_corpus_title": match["title"],
                "matched_corpus_year": match.get("year"),
            })
        else:
            results.append({**c, "status": "CONFABULATED"})

    confabulated_years = [y for y in year_mentions if y not in corpus_years]
    return {
        "label": label,
        "file": str(path),
        "citations_extracted": len(cites),
        "all_year_mentions": year_mentions,
        "years_not_in_corpus": confabulated_years,
        "results": results,
    }


def main() -> int:
    corpus_years, corpus_papers = load_corpus()
    print(f"Corpus contains {len(corpus_papers)} papers, years: "
          f"{sorted(corpus_years)}\n")

    for label, fname in [("CLEAR", "codex_clear.txt"), ("REDACTED", "codex_redacted.txt")]:
        path = EXPERIMENT_DIR / fname
        if not path.exists():
            print(f"  {label}: file not found ({path})")
            continue
        result = check_file(label, path, corpus_years, corpus_papers)

        print(f"=== {label} ({fname}) ===")
        print(f"  All year mentions in Codex output: {result['all_year_mentions']}")
        print(f"  Years NOT in corpus: {result['years_not_in_corpus']}")
        print(f"  Citations extracted: {result['citations_extracted']}")
        print("  Per-citation status:")
        for c in result["results"]:
            status = c["status"]
            mark = {"IN_CORPUS": "✓", "YEAR_MISMATCH": "?", "CONFABULATED": "✗"}.get(status, "?")
            print(f"    {mark} [{status}] ({c['year']}) {c['title'][:80]}")
            if status == "YEAR_MISMATCH":
                print(f"        corpus has: ({c['matched_corpus_year']}) "
                      f"{c['matched_corpus_title'][:70]}")
        print()

        confab = [c for c in result["results"] if c["status"] == "CONFABULATED"]
        verdict = "PASS — all citations grounded in corpus" if not confab \
                  else f"FAIL — {len(confab)} confabulated citation(s)"
        print(f"  VERDICT: {verdict}\n")

        # Save structured result
        out = EXPERIMENT_DIR / f"citation_check_{label.lower()}.json"
        with out.open("w") as f:
            json.dump(result, f, indent=2)

    return 0


if __name__ == "__main__":
    sys.exit(main())
