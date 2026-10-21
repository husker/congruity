"""Layer 3 — Codex bridge discovery + citation check.

For each pair that passed Layer 2 (cross-citation density), build a corpus
of post-cutoff papers from each side, run Codex to find an equivalence,
then verify all citations land in the supplied corpus.

Refactored from scripts/step3_kvcache_experiment.py + scripts/step3_citation_check.py.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

from pipeline._openalex import extract_id
from pipeline.corpus import sample_topic

RUNS_DIR = Path(__file__).resolve().parent.parent / "data" / "pipeline" / "runs"
POST_CUTOFF_YEAR_MIN = 2025  # papers Codex couldn't have seen
PER_SIDE_CAP = 8


def build_prompt(side_a: list[dict], side_b: list[dict],
                 a_name: str, b_name: str) -> str:
    lines = [
        "You are reading abstracts from two corpora of computer science papers, "
        "all published in 2025 or later. Each corpus is from a distinct research "
        "community with different conferences, vocabulary, and citation networks.",
        "",
        f"Corpus A is the topic: {a_name}",
        f"Corpus B is the topic: {b_name}",
        "",
        "Your task: identify any *operational equivalence* between concepts in "
        "Corpus A and concepts in Corpus B. By \"operational equivalence\" I mean "
        "a structural correspondence such that a mechanism developed for one "
        "corpus could be applied to solve problems in the other corpus — even if "
        "the authors in neither corpus state this.",
        "",
        "Output:",
        "  1. Equivalence claim (one sentence).",
        "  2. Supporting papers (cite by title and year from the corpus only).",
        "  3. Mechanism transferred — what would actually be reused.",
        "  4. Why the original authors likely did not state this equivalence.",
        "  5. Confidence rating 1–10 with a one-sentence reason.",
        "",
        "Do not fabricate citations. Cite ONLY papers present in the corpus below.",
        "If no substantive equivalence exists, say so explicitly — do not invent one.",
        "",
        f"=========== Corpus A — {a_name} — {len(side_a)} papers ===========",
        "",
    ]
    for i, p in enumerate(side_a, 1):
        lines.append(f"--- Paper A{i} ({p.get('year')}): {p.get('title')} ---")
        if p.get("authors"):
            lines.append(f"Authors: {', '.join(p['authors'])}")
        lines.append("")
        lines.append(p.get("abstract", ""))
        lines.append("")
    lines.append(f"=========== Corpus B — {b_name} — {len(side_b)} papers ===========")
    lines.append("")
    for i, p in enumerate(side_b, 1):
        lines.append(f"--- Paper B{i} ({p.get('year')}): {p.get('title')} ---")
        if p.get("authors"):
            lines.append(f"Authors: {', '.join(p['authors'])}")
        lines.append("")
        lines.append(p.get("abstract", ""))
        lines.append("")
    return "\n".join(lines)


def run_codex(prompt_path: Path, repo_root: Path) -> tuple[bool, str, str]:
    """Run codex exec on a prompt file. Returns (success, stdout, stderr)."""
    try:
        with prompt_path.open() as fin:
            result = subprocess.run(
                ["codex", "exec", "--skip-git-repo-check", "-",
                 "-C", str(repo_root), "-s", "read-only",
                 "-c", 'model_reasoning_effort="high"'],
                stdin=fin,
                capture_output=True,
                text=True,
                timeout=600,
            )
        return (result.returncode == 0, result.stdout, result.stderr)
    except subprocess.TimeoutExpired:
        return (False, "", "codex timeout (10 min)")
    except FileNotFoundError:
        return (False, "", "codex binary not found")


# Citation extraction patterns (from step3/step4 verifiers)
PAT_BOLD = re.compile(r'\*\*(?P<title>[^*\n]{15,200})\*\*\s*\((?P<year>(19|20)\d{2})\)')
PAT_ITALIC = re.compile(r'\*(?P<title>[^*\n]{15,200})\*\s*\((?P<year>(19|20)\d{2})\)')
PAT_QUOTED = re.compile(r'["""\'](?P<title>[^""\'\\n]{15,200})["""\']\s*\((?P<year>(19|20)\d{2})\)')
PAT_YEAR = re.compile(r'\((?P<year>(19|20)\d{2})\)')


def _normalize(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def citation_check(response: str, corpus: list[dict]) -> dict:
    """Verify cited titles + years against corpus. Returns check summary."""
    titles_norm = [(p.get("title", ""), _normalize(p.get("title", "")), p.get("year"))
                   for p in corpus]
    corpus_years = {p.get("year") for p in corpus if p.get("year")}

    seen = set()
    cites: list[dict] = []
    for pat in (PAT_BOLD, PAT_ITALIC, PAT_QUOTED):
        for m in pat.finditer(response):
            title = m.group("title").strip().strip("*\"'")
            year = int(m.group("year"))
            key = (_normalize(title), year)
            if key in seen:
                continue
            seen.add(key)
            cites.append({"title": title, "year": year})

    confabulated = 0
    grounded = 0
    for c in cites:
        cn = _normalize(c["title"])
        cw = set(cn.split())
        found = False
        for orig, pn, py in titles_norm:
            if not pn:
                continue
            if cn in pn or pn in cn:
                found = True
                break
            pw = set(pn.split())
            if cw and pw and len(cw & pw) / len(cw | pw) >= 0.5:
                found = True
                break
        if found:
            grounded += 1
        else:
            confabulated += 1
            c["status"] = "CONFABULATED"

    all_years_mentioned = sorted({int(m.group("year")) for m in PAT_YEAR.finditer(response)})
    years_not_in_corpus = [y for y in all_years_mentioned if y not in corpus_years]

    return {
        "citations_extracted": len(cites),
        "grounded": grounded,
        "confabulated": confabulated,
        "year_mentions": all_years_mentioned,
        "years_not_in_corpus": years_not_in_corpus,
        "passes": confabulated == 0 and not years_not_in_corpus,
        "details": cites,
    }


CONFIDENCE_PAT = re.compile(r"confidence[:\s]*(\d+)(?:\s*/\s*\d+)?", re.IGNORECASE)


def parse_confidence(response: str) -> int | None:
    """Extract Codex's self-rated confidence (1-10)."""
    m = CONFIDENCE_PAT.search(response)
    if not m:
        return None
    try:
        c = int(m.group(1))
        return c if 1 <= c <= 10 else None
    except ValueError:
        return None


def pair_id(a_short: str, b_short: str) -> str:
    return f"{a_short}__vs__{b_short}"


def discover_one(
    a_topic: dict,
    b_topic: dict,
    year_min: int = POST_CUTOFF_YEAR_MIN,
    per_side_cap: int = PER_SIDE_CAP,
    dry_run: bool = False,
) -> dict:
    """Run discovery on one pair. Returns a result dict; persists artifacts to disk."""
    a_short = extract_id(a_topic["id"])
    b_short = extract_id(b_topic["id"])
    pid = pair_id(a_short, b_short)
    out_dir = RUNS_DIR / pid
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"  fetching corpus for {a_topic['display_name']}...", flush=True)
    side_a = sample_topic(a_topic["id"], year_min=year_min, cap=per_side_cap,
                          include_refs=False, include_abstract=True)
    print(f"    side A: {len(side_a)} papers")
    print(f"  fetching corpus for {b_topic['display_name']}...", flush=True)
    side_b = sample_topic(b_topic["id"], year_min=year_min, cap=per_side_cap,
                          include_refs=False, include_abstract=True)
    print(f"    side B: {len(side_b)} papers")

    if len(side_a) < 3 or len(side_b) < 3:
        return {
            "pair_id": pid,
            "a_name": a_topic["display_name"],
            "b_name": b_topic["display_name"],
            "status": "skipped_insufficient_corpus",
            "side_a_count": len(side_a),
            "side_b_count": len(side_b),
        }

    prompt = build_prompt(side_a, side_b,
                          a_topic["display_name"], b_topic["display_name"])
    prompt_path = out_dir / "prompt.txt"
    prompt_path.write_text(prompt)
    (out_dir / "corpus.json").write_text(json.dumps(
        {"side_a": side_a, "side_b": side_b,
         "a_topic": a_topic["display_name"],
         "b_topic": b_topic["display_name"]}, indent=2))

    if dry_run:
        return {
            "pair_id": pid,
            "a_name": a_topic["display_name"],
            "b_name": b_topic["display_name"],
            "status": "dry_run",
            "side_a_count": len(side_a),
            "side_b_count": len(side_b),
            "prompt_chars": len(prompt),
            "prompt_path": str(prompt_path),
        }

    print("  running codex...", flush=True)
    repo_root = Path(__file__).resolve().parent.parent
    ok, stdout, stderr = run_codex(prompt_path, repo_root)
    (out_dir / "codex_response.txt").write_text(stdout)
    (out_dir / "codex_stderr.txt").write_text(stderr)

    if not ok:
        return {
            "pair_id": pid,
            "a_name": a_topic["display_name"],
            "b_name": b_topic["display_name"],
            "status": "codex_failed",
            "error": stderr[-500:] if stderr else "(no stderr)",
        }

    corpus_combined = side_a + side_b
    check = citation_check(stdout, corpus_combined)
    (out_dir / "citation_check.json").write_text(json.dumps(check, indent=2))

    confidence = parse_confidence(stdout)

    return {
        "pair_id": pid,
        "a_name": a_topic["display_name"],
        "b_name": b_topic["display_name"],
        "status": "complete",
        "side_a_count": len(side_a),
        "side_b_count": len(side_b),
        "response_chars": len(stdout),
        "confidence": confidence,
        "citation_check": check,
        "response_path": str(out_dir / "codex_response.txt"),
    }


def main() -> int:
    p = argparse.ArgumentParser(description="Run Codex discovery on one pair")
    p.add_argument("--a", required=True, help="OpenAlex topic ID for side A")
    p.add_argument("--b", required=True, help="OpenAlex topic ID for side B")
    p.add_argument("--a-name", default="(side A)")
    p.add_argument("--b-name", default="(side B)")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    result = discover_one(
        {"id": args.a, "display_name": args.a_name},
        {"id": args.b, "display_name": args.b_name},
        dry_run=args.dry_run,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
