"""Pluggable relevance-check backends for the Step 1.5 corpus quality gate.

The gate's relevance arm previously hard-wired ``text-embedding-3-small``
behind ``OPENAI_API_KEY``. This module factors it into pluggable backends so
the gate can run against whatever auth the user has available:

  - codex      : LLM-judge via the codex CLI (uses whatever auth codex has
                 locally configured — no env-var requirement). Same prompt
                 shape as the anthropic backend; routes through the
                 existing _invoke_codex helper in pipeline/llm_io.py.
  - openai     : embedding cosine similarity (existing path; see
                 corpus_gate._openai_embed)
  - anthropic  : LLM-judge via Anthropic messages API. No embeddings; instead
                 prompts Claude Haiku to score each paper 0.0-1.0 on
                 relevance to the problem statement, in batches.

All backends return a list of (paper_id, score, detail) tuples plus a
`skipped` flag that callers convert to PaperIssue / drop-id sets in
corpus_gate.

The LLM-judge paths are intentionally simple: one batched call per ~20
papers, structured JSON output validated by Pydantic, single-shot retry
on parse failure.
"""

from __future__ import annotations

import json
import math
import os
import sys
import time

import requests
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from pipeline.llm_io import _extract_json_block

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"
DEFAULT_ANTHROPIC_MODEL = "claude-haiku-4-5-20251001"
DEFAULT_BATCH_SIZE = 20
DEFAULT_TIMEOUT_S = 90
DEFAULT_MAX_RETRIES = 2
PAPER_SUMMARY_CHARCAP = 1200


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class PaperRelevance(BaseModel):
    model_config = ConfigDict(extra="allow")
    paper_id: str
    score: float = Field(ge=0.0, le=1.0)
    reason: str = ""


class RelevanceScores(BaseModel):
    model_config = ConfigDict(extra="allow")
    scores: list[PaperRelevance]


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------


JUDGE_PROMPT_TEMPLATE = """You are scoring scientific papers on their relevance to a research problem.

PROBLEM:
{problem}

PAPERS:
{papers_block}

YOUR TASK
For each paper above, return a relevance score from 0.0 (entirely unrelated) to
1.0 (directly addresses the problem). Be calibrated: most papers in a wide
topic net will fall below 0.5. A score of 0.7+ means the paper is squarely on
topic; 0.3-0.6 means adjacent or partial overlap; below 0.3 means weak or
incidental connection.

Also include a one-line reason. Be terse.

OUTPUT — JSON ONLY, no prose, no markdown fences:
{{
  "scores": [
    {{"paper_id": "P1", "score": 0.0, "reason": "<one line>"}},
    ...
  ]
}}
"""


def _papers_block(papers: list[dict]) -> str:
    """Compact per-paper block. Truncates abstracts to keep prompts cheap."""
    lines = []
    for p in papers:
        pid = p.get("paper_id", "?")
        title = (p.get("title") or "").strip()
        abstract = (p.get("abstract") or "").strip()
        topic = p.get("source_topic") or ""
        snippet = abstract[:PAPER_SUMMARY_CHARCAP]
        if topic:
            lines.append(f"[{pid}] (topic: {topic}) {title}")
        else:
            lines.append(f"[{pid}] {title}")
        lines.append(f"     {snippet}")
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Anthropic backend
# ---------------------------------------------------------------------------


def _post_anthropic(
    api_key: str,
    prompt: str,
    model: str = DEFAULT_ANTHROPIC_MODEL,
    max_tokens: int = 4096,
    timeout: int = DEFAULT_TIMEOUT_S,
) -> str:
    """Single Anthropic messages API call. Returns the assistant text."""
    headers = {
        "x-api-key": api_key,
        "anthropic-version": ANTHROPIC_VERSION,
        "content-type": "application/json",
    }
    body = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
    }
    backoff = 2.0
    last_err: str | None = None
    for _ in range(DEFAULT_MAX_RETRIES + 1):
        try:
            resp = requests.post(
                ANTHROPIC_API_URL, headers=headers, json=body, timeout=timeout,
            )
            if resp.status_code == 200:
                data = resp.json()
                blocks = data.get("content", [])
                text_parts = [b.get("text", "") for b in blocks if b.get("type") == "text"]
                return "".join(text_parts)
            last_err = f"HTTP {resp.status_code}: {resp.text[:200]}"
        except requests.RequestException as e:
            last_err = f"{type(e).__name__}: {e}"
        time.sleep(backoff)
        backoff *= 2
    raise RuntimeError(f"Anthropic messages API failed: {last_err}")


def _parse_judge_response(raw: str) -> RelevanceScores | None:
    """Extract and validate the judge's JSON."""
    block = _extract_json_block(raw) if raw else None
    if block is None:
        return None
    try:
        return RelevanceScores.model_validate(json.loads(block))
    except (json.JSONDecodeError, ValidationError):
        return None


_REPAIR_FRAMING = (
    "Your previous response was not valid JSON matching the required "
    "schema. Return ONLY a JSON object with a 'scores' array; each entry "
    "has paper_id, score (0.0-1.0), reason. No prose, no fences.\n\n"
)


def _score_papers_judged(
    papers: list[dict],
    problem: str,
    transport,
    label: str,
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> dict[str, float]:
    """Shared LLM-judge batch loop for every relevance backend.

    ``transport(prompt) -> str | None`` is the only backend-specific piece
    (Anthropic HTTP vs codex subprocess). Parse failures get one repair
    attempt; papers the judge drops are backfilled to 1.0 so a parser slip
    never silently loses papers.
    """
    out: dict[str, float] = {}
    total_batches = math.ceil(len(papers) / batch_size)
    for batch_idx, i in enumerate(range(0, len(papers), batch_size)):
        batch = papers[i:i + batch_size]
        prompt = JUDGE_PROMPT_TEMPLATE.format(
            problem=problem.strip(),
            papers_block=_papers_block(batch),
        )
        raw = transport(prompt)
        parsed = _parse_judge_response(raw) if raw else None
        if parsed is None:
            repair = (
                _REPAIR_FRAMING
                + (f"Previous response:\n{raw}\n\n" if raw else "")
                + "Original prompt:\n" + prompt
            )
            raw2 = transport(repair)
            parsed = _parse_judge_response(raw2) if raw2 else None
        if parsed is not None:
            for s in parsed.scores:
                out[s.paper_id] = float(s.score)
        # Backfill any missing paper IDs (model dropped them, or the whole
        # batch failed to parse) with 1.0 — over-include rather than
        # silently drop.
        for p in batch:
            out.setdefault(p.get("paper_id", "?"), 1.0)
        print(
            f"  {label} relevance batch {batch_idx + 1}/{total_batches}: "
            f"scored {len(out)}/{len(papers)} papers",
            flush=True,
        )
    return out


def _issues_from_scores(
    papers: list[dict],
    scores: dict[str, float],
    threshold: float,
) -> list[tuple[str, float, str]]:
    issues: list[tuple[str, float, str]] = []
    for p in papers:
        pid = p.get("paper_id", "?")
        score = scores.get(pid, 1.0)
        if score < threshold:
            issues.append((
                pid,
                score,
                f"LLM-judge score {score:.2f} < threshold {threshold:.2f}",
            ))
    return issues


def score_papers_anthropic(
    papers: list[dict],
    problem: str,
    api_key: str,
    *,
    model: str = DEFAULT_ANTHROPIC_MODEL,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> dict[str, float]:
    """Returns paper_id -> score (0.0..1.0) using the Anthropic LLM-judge.

    Raises RuntimeError on persistent API failure; the caller wraps that as
    a soft "skipped" outcome. Missing IDs backfill to 1.0 (over-include
    rather than silently drop).
    """
    return _score_papers_judged(
        papers, problem,
        lambda prompt: _post_anthropic(api_key, prompt, model=model),
        "anthropic", batch_size=batch_size,
    )


def check_relevance_anthropic(
    papers: list[dict],
    problem: str,
    threshold: float,
    api_key: str | None,
) -> tuple[list[tuple[str, float, str]], bool]:
    """Run the Anthropic LLM-judge relevance check.

    Returns (issues, skipped). issues is a list of (paper_id, score, detail)
    tuples for papers whose score < threshold. skipped=True if the backend
    couldn't run.

    Recommended threshold for LLM-judge scores: start at the pipeline default
    of 0.30 for broad cross-disciplinary corpora, then raise toward 0.45-0.55
    only when the topic list is already tightly focused.
    """
    if not api_key:
        return [], True
    if not papers or not problem.strip():
        return [], True
    try:
        scores = score_papers_anthropic(papers, problem, api_key)
    except RuntimeError as e:
        print(f"  WARN: anthropic relevance skipped — {e}", file=sys.stderr)
        return [], True
    return _issues_from_scores(papers, scores, threshold), False


# ---------------------------------------------------------------------------
# Codex backend
# ---------------------------------------------------------------------------


def score_papers_codex(
    papers: list[dict],
    problem: str,
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    reasoning: str = "medium",
) -> dict[str, float]:
    """Same judge loop as score_papers_anthropic but transported over the
    codex CLI. Authentication comes from whatever codex has locally
    configured, so no env vars are required. Missing IDs backfill to 1.0."""
    from pipeline.llm_io import _invoke_codex

    def transport(prompt: str) -> str | None:
        ok, stdout, _stderr = _invoke_codex(prompt, reasoning)
        return stdout if ok else None

    return _score_papers_judged(papers, problem, transport, "codex",
                                batch_size=batch_size)


def check_relevance_codex(
    papers: list[dict],
    problem: str,
    threshold: float,
) -> tuple[list[tuple[str, float, str]], bool]:
    """Run the codex LLM-judge relevance check. No env-var requirement;
    relies on codex CLI auth.

    Returns (issues, skipped). issues is a list of (paper_id, score, detail)
    tuples for papers whose score < threshold.
    """
    import shutil
    if shutil.which("codex") is None:
        return [], True
    if not papers or not problem.strip():
        return [], True
    try:
        scores = score_papers_codex(papers, problem)
    except Exception as e:  # codex subprocess can raise anything; be defensive
        print(f"  WARN: codex relevance skipped — {e}", file=sys.stderr)
        return [], True
    return _issues_from_scores(papers, scores, threshold), False


# ---------------------------------------------------------------------------
# Dispatch helper
# ---------------------------------------------------------------------------


def resolve_mode(requested: str | None) -> str:
    """Resolve a user-requested mode against available env vars + CLIs.

    'auto' (the default) picks the first available backend in order:
        codex     (codex binary on PATH)
        anthropic (ANTHROPIC_API_KEY in env)
        openai    (OPENAI_API_KEY in env)
        none      (no key / no codex; relevance will be skipped)

    Explicit modes ('openai' / 'anthropic' / 'codex' / 'none') are returned as-is.
    """
    if requested and requested != "auto":
        return requested
    import shutil
    if shutil.which("codex") is not None:
        return "codex"
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic"
    if os.environ.get("OPENAI_API_KEY"):
        return "openai"
    return "none"


# Re-export the cosine helper so callers don't have to import from two
# places. Lazy import inside the function avoids the corpus_gate <->
# relevance import cycle.
def cosine(a: list[float], b: list[float]) -> float:
    from pipeline.corpus_gate import _cosine
    return _cosine(a, b)
