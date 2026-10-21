"""Step 2 experiment — Codex as the hand-run adjudicator.

Tests whether an independent AI (Codex, OpenAI model family) can perform the
memorization-vs-reasoning ablation that the doc specifies as a human hand-run.

Method
------
1. Fetch abstracts from OpenAlex for ~4 pre-bridge papers from each side of the
   session_types_singularity_channels candidate (PL session-types literature
   1993–2004; OS IPC/microkernel literature 1986–2003). Singularity itself
   (2007) is excluded — the system must derive the bridge from the surrounding
   literature.
2. Build two prompts:
   - "clear" — abstracts as-is.
   - "redacted" — bridge-implicating vocabulary paraphrased away (session types →
     communication-protocol types; π-calculus → process algebra; IPC → message-
     passing primitive; microkernel → minimal-kernel system).
3. Write both prompts to files under data/step2_experiment/. The Codex runs
   happen via separate Bash invocations after this script writes the prompts —
   that keeps the experiment hand-inspectable and reproducible.

This is a one-shot experiment to test the *option 3* path: can an independent
AI substitute for the human in Step 2 adjudication? Result either way is
useful information about the verification approach.
"""

from __future__ import annotations

import os
import json
import re
import sys
import time
from pathlib import Path

import requests

BASE = "https://api.openalex.org/works"
MAILTO = os.environ.get("OPENALEX_MAILTO", "")
SLEEP_S = 0.3
REQUEST_TIMEOUT = 60

# Curated pre-bridge papers. Search by title; take the first result that has the
# expected year and a recoverable abstract.
PL_PAPERS = [
    ("Types for dyadic interaction", 1993),
    ("Language primitives and type discipline for structured communication-based programming", 1998),
    ("Types and subtypes for client-server interactions", 1999),
    ("Session types for functional multithreading", 2004),
]

OS_PAPERS = [
    ("Mach: a new kernel foundation for UNIX development", 1986),
    ("On micro-kernel construction", 1995),
    ("Extensibility safety and performance in the SPIN operating system", 1995),
    ("Exokernel: an operating system architecture for application-level resource management", 1995),
]


def abstract_from_inverted(idx: dict | None) -> str:
    """OpenAlex returns abstract_inverted_index = {word: [positions]}.
    Reconstruct linear text."""
    if not idx:
        return ""
    positions: dict[int, str] = {}
    for word, poses in idx.items():
        for p in poses:
            positions[p] = word
    if not positions:
        return ""
    return " ".join(positions[i] for i in sorted(positions.keys()))


def _title_similarity(a: str, b: str) -> float:
    """Word-Jaccard on lowercase titles. 1.0 = identical word sets."""
    aw = set(re.sub(r"[^a-z0-9 ]", " ", a.lower()).split())
    bw = set(re.sub(r"[^a-z0-9 ]", " ", b.lower()).split())
    if not aw or not bw:
        return 0.0
    return len(aw & bw) / len(aw | bw)


def fetch_paper(title: str, expected_year: int) -> dict | None:
    """Search OpenAlex for a paper by title, restricted to expected_year ±1
    via publication_year filter. Require Jaccard title similarity ≥ 0.4 against
    the requested title. Return None on no match — never silently substitute
    an off-topic paper.
    """
    params = {
        "filter": f"publication_year:{expected_year - 1}-{expected_year + 1}",
        "search": title,
        "per-page": 10,
        "select": "id,title,publication_year,authorships,abstract_inverted_index",
        "mailto": MAILTO,
    }
    try:
        resp = requests.get(BASE, params=params, timeout=REQUEST_TIMEOUT)
        if resp.status_code != 200:
            print(f"  WARN: HTTP {resp.status_code} for '{title}'", file=sys.stderr)
            return None
        body = resp.json()
        candidates = []
        for work in body.get("results", []):
            year = work.get("publication_year")
            abstract = abstract_from_inverted(work.get("abstract_inverted_index"))
            if not abstract:
                continue
            sim = _title_similarity(title, work.get("title", ""))
            candidates.append((sim, work, year, abstract))
        candidates.sort(key=lambda x: -x[0])
        for sim, work, year, abstract in candidates:
            if sim >= 0.4:
                authors = [a["author"]["display_name"]
                           for a in work.get("authorships", [])
                           if a.get("author")]
                return {
                    "openalex_id": work.get("id"),
                    "title": work.get("title"),
                    "year": year,
                    "authors": authors[:5],
                    "abstract": abstract,
                    "title_similarity": round(sim, 2),
                }
        # No year-bounded match with adequate title similarity — try without
        # year filter as a last attempt, but require very high similarity.
        params2 = dict(params)
        params2.pop("filter")
        resp2 = requests.get(BASE, params=params2, timeout=REQUEST_TIMEOUT)
        if resp2.status_code == 200:
            for work in resp2.json().get("results", []):
                abstract = abstract_from_inverted(work.get("abstract_inverted_index"))
                if not abstract:
                    continue
                sim = _title_similarity(title, work.get("title", ""))
                if sim >= 0.7:
                    year = work.get("publication_year")
                    authors = [a["author"]["display_name"]
                               for a in work.get("authorships", [])
                               if a.get("author")]
                    return {
                        "openalex_id": work.get("id"),
                        "title": work.get("title"),
                        "year": year,
                        "authors": authors[:5],
                        "abstract": abstract,
                        "title_similarity": round(sim, 2),
                        "note": f"YEAR MISMATCH (expected {expected_year}, got {year})",
                    }
        return None
    except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
        print(f"  ERR ({type(e).__name__}) for '{title}'", file=sys.stderr)
        return None


# Vocabulary that points at the bridge. The redaction is deliberately aggressive
# — if the model still surfaces the bridge after this, it's stronger evidence
# of reasoning over memorization.
REDACTION_MAP: list[tuple[str, str]] = [
    # PL side (session types literature)
    (r"\bsession type[s]?\b", "communication-protocol type"),
    (r"\bsession[s]?\b", "structured communication unit"),
    (r"\bduality\b", "complementarity"),
    (r"\bdual\b", "complementary"),
    (r"\bπ-calculus\b", "process algebra"),
    (r"\bpi-calculus\b", "process algebra"),
    (r"\bpi calculus\b", "process algebra"),
    (r"\bprocess calculus\b", "process algebra"),
    (r"\bchannel[s]?\b", "communication endpoint"),
    # OS side (IPC / microkernel)
    (r"\bIPC\b", "process-to-process messaging"),
    (r"\binter-?process communication\b", "process-to-process messaging"),
    (r"\bmicro-?kernel[s]?\b", "minimal-kernel system"),
    (r"\bmessage passing\b", "data exchange"),
    (r"\bmessage[s]?\b", "data unit"),
    (r"\bSPIN\b", "[the typed-extension OS]"),  # the system name itself
    (r"\bExokernel\b", "[the minimal-resource-management OS]"),
    (r"\bMach\b", "[the foundational kernel project]"),
    # Bridge term itself — not in pre-2007 abstracts but defensive
    (r"\bSingularity\b", "[the typed-IPC OS]"),
]


def redact(text: str) -> str:
    out = text
    for pat, rep in REDACTION_MAP:
        out = re.sub(pat, rep, out, flags=re.IGNORECASE)
    return out


def build_prompt(pl_corpus: list[dict], os_corpus: list[dict], redacted: bool) -> str:
    transform = redact if redacted else (lambda x: x)
    lines = [
        "You are reading abstracts from two corpora of computer science papers, "
        "all published before 2005. Each corpus is from a distinct research "
        "community with different conferences, vocabulary, and citation networks.",
        "",
        "Your task: identify any *operational equivalence* between concepts "
        "described in Corpus A and concepts described in Corpus B. By "
        "\"operational equivalence\" I mean a structural correspondence such that "
        "a mechanism developed for one corpus could be applied to solve problems "
        "in the other corpus — even if the authors in neither corpus state this.",
        "",
        "If you find one, output:",
        "  1. The equivalence claim, in one sentence.",
        "  2. Which paper in each corpus supports it (cite by title/year).",
        "  3. The mechanism — what would actually be transferred.",
        "  4. Why you believe the original authors did not state this equivalence.",
        "  5. A confidence rating 1–10 and a one-sentence reason for the rating.",
        "",
        "If you cannot find a substantive equivalence, say so explicitly. Do not "
        "fabricate one. If you find something trivial (e.g. both corpora discuss "
        "\"computation\"), say so and stop.",
        "",
        f"=========== Corpus A ({len(pl_corpus)} papers) ===========",
        "",
    ]
    for i, p in enumerate(pl_corpus, 1):
        lines.append(f"--- Paper A{i} ({p['year']}): {transform(p['title'])} ---")
        if p.get("authors"):
            lines.append(f"Authors: {', '.join(p['authors'])}")
        lines.append("")
        lines.append(transform(p["abstract"]))
        lines.append("")
    lines.append(f"=========== Corpus B ({len(os_corpus)} papers) ===========")
    lines.append("")
    for i, p in enumerate(os_corpus, 1):
        lines.append(f"--- Paper B{i} ({p['year']}): {transform(p['title'])} ---")
        if p.get("authors"):
            lines.append(f"Authors: {', '.join(p['authors'])}")
        lines.append("")
        lines.append(transform(p["abstract"]))
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    out_dir = Path(__file__).resolve().parent.parent / "data" / "step2_experiment"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Fetching PL papers (session types literature)...")
    pl_corpus: list[dict] = []
    for title, year in PL_PAPERS:
        print(f"  {year} — {title[:60]}...", end=" ", flush=True)
        paper = fetch_paper(title, year)
        if paper:
            print(f"[{len(paper['abstract'].split())} words]")
            pl_corpus.append(paper)
        else:
            print("MISSING")
        time.sleep(SLEEP_S)

    print("\nFetching OS papers (IPC / microkernel literature)...")
    os_corpus: list[dict] = []
    for title, year in OS_PAPERS:
        print(f"  {year} — {title[:60]}...", end=" ", flush=True)
        paper = fetch_paper(title, year)
        if paper:
            print(f"[{len(paper['abstract'].split())} words]")
            os_corpus.append(paper)
        else:
            print("MISSING")
        time.sleep(SLEEP_S)

    print(f"\nPL corpus: {len(pl_corpus)} papers, "
          f"OS corpus: {len(os_corpus)} papers.")
    if len(pl_corpus) < 2 or len(os_corpus) < 2:
        print("Insufficient corpus to run experiment.", file=sys.stderr)
        return 1

    # Save corpora for inspection
    with (out_dir / "corpus.json").open("w") as f:
        json.dump({"pl": pl_corpus, "os": os_corpus}, f, indent=2)

    # Build prompts
    prompt_clear = build_prompt(pl_corpus, os_corpus, redacted=False)
    prompt_redacted = build_prompt(pl_corpus, os_corpus, redacted=True)

    (out_dir / "prompt_clear.txt").write_text(prompt_clear)
    (out_dir / "prompt_redacted.txt").write_text(prompt_redacted)

    print("\nPrompts written:")
    print(f"  {out_dir / 'prompt_clear.txt'}  ({len(prompt_clear)} chars)")
    print(f"  {out_dir / 'prompt_redacted.txt'}  ({len(prompt_redacted)} chars)")
    print("\nNext: run codex on each prompt and capture output.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
