"""Corpus quality gate (Step 1.5).

Validates a paper corpus before downstream LLM-heavy stages, so bad data
fails fast with a clear report instead of polluting composition output.

Three checks:
  1. Topic relevance — Codex/Anthropic LLM-judge scores or OpenAI embedding
     cosine similarity between the problem statement and each paper.
  2. Abstract encoding integrity — mojibake, HTML tags / entities,
     control chars, U+FFFD replacement chars.
  3. Citation/structure validity — openalex_id matches the W-id regex,
     title/year/abstract present at minimum lengths.

Soft-fail by default: drops papers that fail any check and returns a report with
drop_ids. Aborts only when relevance cannot run in strict mode, when actual
data-quality failures exceed max_fail_rate, or when too few papers survive for
downstream composition.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import requests

OPENALEX_ID_RE = re.compile(r"^(?:https?://openalex\.org/)?W\d+$")
MIN_ABSTRACT_WORDS = 30
MIN_TITLE_LEN = 8
MIN_YEAR = 1900
MAX_YEAR = 2030

MOJIBAKE_TOKENS = (
    "â€™", "â€œ", "â€\x9d", "â€\x9c", "â€“", "â€”", "â€¦",
    "Â£", "Â©", "Â®", "Â°", "Â±", "Â·",
    "Ã©", "Ã¨", "Ã ", "Ã¢", "Ã¯", "Ã®", "Ã´", "Ã»", "Ã§", "Ã±",
    "Ã¤", "Ã¶", "Ã¼", "ÃŸ",
)
REPLACEMENT_CHAR_RE = re.compile(r"�")
HTML_TAG_RE = re.compile(
    r"</?(?:p|br|i|b|em|strong|sub|sup|div|span|jats:[a-z]+)(?:\s[^>]*)?/?>",
    re.IGNORECASE,
)
HTML_ENTITY_RE = re.compile(
    r"&(?:amp|lt|gt|quot|apos|nbsp|#\d+|#x[0-9a-f]+|[a-z]{2,10});",
    re.IGNORECASE,
)
CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")

EMBED_MODEL = "text-embedding-3-small"
EMBED_BATCH = 100
EMBED_TIMEOUT = 60
EMBED_MAX_RETRIES = 2
EMBED_TEXT_CHARCAP = 8000


@dataclass
class PaperIssue:
    paper_id: str
    openalex_id: str | None
    check: str
    detail: str


@dataclass
class CorpusGateReport:
    total: int
    passed: int
    failed: int
    fail_rate: float
    by_check: dict[str, int]
    drop_ids: set[str]
    quality_failed: int = 0
    quality_fail_rate: float = 0.0
    min_passed: int = 0
    issues: list[PaperIssue] = field(default_factory=list)
    skipped_checks: list[str] = field(default_factory=list)
    aborted: bool = False
    abort_reason: str = ""

    def as_dict(self) -> dict:
        return {
            "total": self.total,
            "passed": self.passed,
            "failed": self.failed,
            "fail_rate": round(self.fail_rate, 4),
            "by_check": self.by_check,
            "drop_ids": sorted(self.drop_ids),
            "quality_failed": self.quality_failed,
            "quality_fail_rate": round(self.quality_fail_rate, 4),
            "min_passed": self.min_passed,
            "skipped_checks": self.skipped_checks,
            "aborted": self.aborted,
            "abort_reason": self.abort_reason,
            "issues": [asdict(i) for i in self.issues],
        }


# ---------------------------------------------------------------------------
# Check 1: topic relevance via embedding similarity
# ---------------------------------------------------------------------------


EMBED_CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "pipeline" / "cache"


def _embed_key(text: str) -> str:
    """Cache key for one embedding input: model + content hash."""
    return hashlib.sha256(f"{EMBED_MODEL}\x00{text}".encode()).hexdigest()


def _embed_cache_path() -> Path:
    return EMBED_CACHE_DIR / f"embeddings-{EMBED_MODEL}.json"


def _load_embed_cache() -> dict[str, list[float]]:
    p = _embed_cache_path()
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def _save_embed_cache(cache: dict[str, list[float]]) -> None:
    EMBED_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = _embed_cache_path().with_suffix(".tmp")
    tmp.write_text(json.dumps(cache))
    tmp.replace(_embed_cache_path())


def _openai_embed(inputs: list[str], api_key: str) -> list[list[float]]:
    """Call OpenAI embeddings API with retry. Raises RuntimeError on persistent failure."""
    backoff = 2.0
    last_err: str | None = None
    for _ in range(EMBED_MAX_RETRIES + 1):
        try:
            resp = requests.post(
                "https://api.openai.com/v1/embeddings",
                headers={"Authorization": f"Bearer {api_key}",
                         "Content-Type": "application/json"},
                json={"model": EMBED_MODEL, "input": inputs},
                timeout=EMBED_TIMEOUT,
            )
            if resp.status_code == 200:
                data = resp.json().get("data", [])
                return [d["embedding"] for d in data]
            last_err = f"HTTP {resp.status_code}: {resp.text[:200]}"
        except requests.RequestException as e:
            last_err = f"{type(e).__name__}: {e}"
        time.sleep(backoff)
        backoff *= 2
    raise RuntimeError(f"OpenAI embeddings failed: {last_err}")


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def check_relevance(
    papers: list[dict],
    problem: str,
    threshold: float,
    api_key: str | None,
) -> tuple[list[PaperIssue], bool]:
    """Returns (issues, skipped). skipped=True if no API key, no problem
    statement, or the API call failed — caller treats that as a non-fatal warning."""
    if not api_key:
        return [], True
    if not papers or not problem.strip():
        return [], True

    texts = [problem.strip()] + [
        (f"{p.get('title', '')}. {p.get('abstract', '')}")[:EMBED_TEXT_CHARCAP]
        for p in papers
    ]

    # Content-hash cache: the same problem + corpus re-embeds on every
    # compose rerun without this. Only cache misses hit the API.
    cache = _load_embed_cache()
    keys = [_embed_key(t) for t in texts]
    misses = [t for t, k in zip(texts, keys) if k not in cache]
    try:
        if misses:
            fresh: list[list[float]] = []
            for i in range(0, len(misses), EMBED_BATCH):
                batch = misses[i:i + EMBED_BATCH]
                fresh.extend(_openai_embed(batch, api_key))
            if len(fresh) != len(misses):
                print(f"  WARN: relevance check skipped — embedding count "
                      f"{len(fresh)} != input count {len(misses)}",
                      file=sys.stderr)
                return [], True
            for t, v in zip(misses, fresh):
                cache[_embed_key(t)] = v
            _save_embed_cache(cache)
        if len(misses) < len(texts):
            print(f"  relevance: {len(texts) - len(misses)}/{len(texts)} "
                  "embeddings from cache")
        all_vecs: list[list[float]] = [cache[k] for k in keys]
    except RuntimeError as e:
        print(f"  WARN: relevance check skipped — {e}", file=sys.stderr)
        return [], True

    problem_vec = all_vecs[0]
    issues: list[PaperIssue] = []
    for p, vec in zip(papers, all_vecs[1:]):
        sim = _cosine(problem_vec, vec)
        if sim < threshold:
            issues.append(PaperIssue(
                paper_id=p.get("paper_id", "?"),
                openalex_id=p.get("openalex_id"),
                check="relevance",
                detail=f"cosine sim {sim:.3f} < threshold {threshold:.3f}",
            ))
    return issues, False


# ---------------------------------------------------------------------------
# Check 2: encoding corruption
# ---------------------------------------------------------------------------


def _encoding_problems(text: str) -> list[str]:
    if not text:
        return ["empty text"]
    found: list[str] = []
    if REPLACEMENT_CHAR_RE.search(text):
        found.append("contains U+FFFD replacement character")
    mojibake_hits = [t for t in MOJIBAKE_TOKENS if t in text]
    if mojibake_hits:
        found.append(f"mojibake markers ({len(mojibake_hits)} kind(s), e.g. {mojibake_hits[0]!r})")
    tag_match = HTML_TAG_RE.search(text)
    if tag_match:
        found.append(f"contains HTML/JATS tags (e.g. {tag_match.group(0)!r})")
    entities = HTML_ENTITY_RE.findall(text)
    if entities:
        found.append(f"contains undecoded HTML entities (e.g. {entities[0]!r})")
    if CONTROL_CHAR_RE.search(text):
        found.append("contains control characters")
    qmark_count = text.count("?")
    if qmark_count >= 10 and (qmark_count / max(len(text), 1)) > 0.05:
        pct = qmark_count / len(text) * 100
        found.append(f"excessive '?' chars ({qmark_count}, {pct:.1f}% of text)")
    return found


def check_encoding(papers: list[dict]) -> list[PaperIssue]:
    issues: list[PaperIssue] = []
    for p in papers:
        pid = p.get("paper_id", "?")
        oid = p.get("openalex_id")
        text = f"{p.get('title') or ''}\n{p.get('abstract') or ''}"
        for problem in _encoding_problems(text):
            issues.append(PaperIssue(
                paper_id=pid, openalex_id=oid,
                check="encoding", detail=problem,
            ))
    return issues


# ---------------------------------------------------------------------------
# Check 3: citation/structure validity
# ---------------------------------------------------------------------------


def _structure_problems(p: dict) -> list[str]:
    found: list[str] = []
    oid = p.get("openalex_id")
    if not oid:
        found.append("missing openalex_id")
    elif not OPENALEX_ID_RE.match(str(oid)):
        found.append(f"openalex_id fails W-id regex: {oid!r}")
    title = (p.get("title") or "").strip()
    if not title:
        found.append("missing title")
    elif len(title) < MIN_TITLE_LEN:
        found.append(f"title too short ({len(title)} chars < {MIN_TITLE_LEN})")
    year = p.get("year")
    if year is None:
        found.append("missing year")
    else:
        try:
            y = int(year)
            if y < MIN_YEAR or y > MAX_YEAR:
                found.append(f"year out of range: {y}")
        except (TypeError, ValueError):
            found.append(f"year not numeric: {year!r}")
    abstract = p.get("abstract") or ""
    word_count = len(abstract.split())
    if word_count < MIN_ABSTRACT_WORDS:
        found.append(f"abstract too short ({word_count} words < {MIN_ABSTRACT_WORDS})")
    return found


def check_structure(papers: list[dict]) -> list[PaperIssue]:
    issues: list[PaperIssue] = []
    for p in papers:
        pid = p.get("paper_id", "?")
        oid = p.get("openalex_id")
        for problem in _structure_problems(p):
            issues.append(PaperIssue(
                paper_id=pid, openalex_id=oid,
                check="structure", detail=problem,
            ))
    return issues


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def validate_corpus(
    papers: list[dict],
    problem: str,
    relevance_threshold: float = 0.30,
    max_fail_rate: float = 0.30,
    api_key: str | None = None,
    strict: bool = False,
    relevance_mode: str = "auto",
    min_passed: int = 0,
) -> CorpusGateReport:
    """Run all three checks. Returns a report with drop_ids (papers that
    failed any check) and an abort flag if data-quality failures exceed
    max_fail_rate, if too few papers survive, or if strict mode is enabled and
    relevance was skipped. High relevance drop rates are allowed when the
    surviving corpus is still large enough; this keeps broad exploratory runs
    from being blocked before discovery can happen.

    relevance_mode selects the relevance-check backend:
        "auto"      — pick codex > anthropic > openai > none based on local auth
        "openai"    — text-embedding-3-small cosine (needs OPENAI_API_KEY)
        "anthropic" — Claude Haiku LLM-judge (needs ANTHROPIC_API_KEY)
        "codex"     — Codex CLI LLM-judge (needs local codex auth)
        "none"      — skip relevance unconditionally
    """
    from pipeline.relevance import (
        check_relevance_anthropic,
        check_relevance_codex,
        resolve_mode,
    )

    mode = resolve_mode(relevance_mode)
    rel_issues: list[PaperIssue] = []
    rel_skipped = True
    print(
        f"  relevance backend: {mode} "
        f"(threshold {relevance_threshold:.2f})",
        flush=True,
    )

    def _wrap(raw_issues):
        """Convert (pid, score, detail) tuples into PaperIssue objects."""
        return [
            PaperIssue(
                paper_id=pid,
                openalex_id=next(
                    (p.get("openalex_id") for p in papers if p.get("paper_id") == pid),
                    None,
                ),
                check="relevance",
                detail=detail,
            )
            for pid, _score, detail in raw_issues
        ]

    if mode == "openai":
        oai_key = api_key or os.environ.get("OPENAI_API_KEY")
        rel_issues, rel_skipped = check_relevance(
            papers, problem, relevance_threshold, oai_key,
        )
    elif mode == "anthropic":
        a_key = os.environ.get("ANTHROPIC_API_KEY")
        raw_issues, rel_skipped = check_relevance_anthropic(
            papers, problem, relevance_threshold, a_key,
        )
        rel_issues = _wrap(raw_issues)
    elif mode == "codex":
        raw_issues, rel_skipped = check_relevance_codex(
            papers, problem, relevance_threshold,
        )
        rel_issues = _wrap(raw_issues)
    # else: "none" — leave rel_skipped=True and rel_issues=[]
    enc_issues = check_encoding(papers)
    str_issues = check_structure(papers)

    quality_drop_ids = {i.paper_id for i in enc_issues + str_issues}
    quality_failed = len(quality_drop_ids)
    quality_fail_rate = quality_failed / len(papers) if papers else 0.0

    all_issues = rel_issues + enc_issues + str_issues
    drop_ids = {i.paper_id for i in all_issues}
    by_check = {
        "relevance": len(rel_issues),
        "encoding": len(enc_issues),
        "structure": len(str_issues),
    }
    total = len(papers)
    failed = len(drop_ids)
    passed = total - failed
    fail_rate = failed / total if total else 0.0

    report = CorpusGateReport(
        total=total,
        passed=passed,
        failed=failed,
        fail_rate=fail_rate,
        by_check=by_check,
        drop_ids=drop_ids,
        quality_failed=quality_failed,
        quality_fail_rate=quality_fail_rate,
        min_passed=min_passed,
        issues=all_issues,
    )
    if rel_skipped:
        report.skipped_checks.append("relevance")

    if strict and rel_skipped:
        report.aborted = True
        report.abort_reason = (
            "relevance check skipped (no API key or transient error) "
            "and --strict-gate is enabled; configure a relevance backend "
            "(Codex CLI auth or an API key), pass --relevance-mode none, "
            "or pass --no-strict-gate to proceed"
        )
        return report

    if quality_fail_rate > max_fail_rate:
        report.aborted = True
        report.abort_reason = (
            f"data-quality fail rate {quality_fail_rate:.1%} exceeds "
            f"max_fail_rate {max_fail_rate:.1%}; corpus quality is insufficient "
            f"for reliable composition"
        )
    elif min_passed > 0 and passed < min_passed:
        report.aborted = True
        report.abort_reason = (
            f"validated corpus has {passed} papers after filtering; need at least "
            f"{min_passed} for reliable composition"
        )
    return report


def print_report(report: CorpusGateReport) -> None:
    """Human-readable summary printed at the end of Step 1.5."""
    print(f"  total: {report.total}")
    print(f"  passed: {report.passed}")
    print(f"  failed: {report.failed} ({report.fail_rate:.1%})")
    print(f"  by check: relevance={report.by_check.get('relevance', 0)}"
          f", encoding={report.by_check.get('encoding', 0)}"
          f", structure={report.by_check.get('structure', 0)}")
    if report.quality_failed:
        print(f"  data-quality failed: {report.quality_failed} "
              f"({report.quality_fail_rate:.1%})")
    if report.skipped_checks:
        print(f"  ⚠️ SKIPPED CHECKS: {', '.join(report.skipped_checks)} "
              f"— corpus quality conclusions may be weaker than reported")
    by_paper: dict[str, list[PaperIssue]] = {}
    for i in report.issues:
        by_paper.setdefault(i.paper_id, []).append(i)
    if by_paper:
        sample_limit = 10
        sample = list(by_paper.items())[:sample_limit]
        print(f"  first {len(sample)} failing papers:")
        for pid, paper_issues in sample:
            detail = "; ".join(f"[{i.check}] {i.detail}" for i in paper_issues)
            print(f"    {pid}: {detail}")
        if len(by_paper) > sample_limit:
            print(f"    ... and {len(by_paper) - sample_limit} more")
    if report.aborted:
        print(f"  ABORTED: {report.abort_reason}")
