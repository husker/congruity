"""LLM I/O wrappers: artifact-saving codex invocation + schema-validated parsing.

This module wraps subprocess-based `codex exec` calls so that every stage:
  - persists prompt, stdout, stderr, parsed JSON, and meta.json under
    <run_dir>/llm_calls/<stage>/<NN>/
  - validates output against a Pydantic model with one shot at JSON
    repair before giving up
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Generic, TypeVar

from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)

# (ok, stdout, stderr) — back-compat shape; (ok, stdout, stderr, artifact_dir)
# is preferred so parse_or_repair can stamp parse_status on the repair
# artifact's own meta.json rather than leaving it stuck at "pending".
RepairCall = Callable[[str], tuple]


@dataclass
class ParseResult(Generic[T]):
    status: str  # "ok" | "repaired" | "failed"
    value: T | None
    errors: list[str] = field(default_factory=list)
    raw: str = ""


def _extract_json_block(text: str) -> str | None:
    """Pull the first JSON object out of LLM output. Mirrors the legacy
    _parse_json_block heuristic in compose.py: fenced ```json ... ``` blocks
    first, then the first balanced {...}."""
    if not text:
        return None
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        return fenced.group(1)
    brace = re.search(r"(\{.*\})", text, re.DOTALL)
    if not brace:
        return None
    body = brace.group(1)
    depth = 0
    end = -1
    for i, c in enumerate(body):
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                end = i
                break
    if end > 0:
        return body[: end + 1]
    return None


def _validate(raw: str, model_cls: type[T]) -> tuple[T | None, list[str]]:
    block = _extract_json_block(raw)
    if block is None:
        return None, ["could not extract JSON block from response"]
    try:
        data = json.loads(block)
    except json.JSONDecodeError as e:
        return None, [f"JSONDecodeError: {e}"]
    try:
        return model_cls.model_validate(data), []
    except ValidationError as e:
        return None, [str(e)]


def parse_or_repair(
    raw: str,
    model_cls: type[T],
    repair_call: RepairCall | None,
) -> ParseResult[T]:
    """Try to parse `raw` into `model_cls`. On failure, optionally invoke
    `repair_call` once with a repair prompt; if that also fails, return
    a `failed` result with accumulated error messages.

    repair_call signature: (repair_prompt: str) -> (ok, stdout, stderr).
    """
    value, errors = _validate(raw, model_cls)
    if value is not None:
        return ParseResult(status="ok", value=value, errors=[], raw=raw)

    if repair_call is None:
        return ParseResult(status="failed", value=None, errors=errors, raw=raw)

    repair_prompt = (
        "Your previous response could not be parsed against the required "
        "JSON schema. Here is the validation error:\n\n"
        f"{errors[0]}\n\n"
        "Your previous response (verbatim):\n"
        f"{raw}\n\n"
        "Return ONLY a corrected JSON object that conforms to the schema. "
        "No prose, no fences."
    )
    repair_result = repair_call(repair_prompt)
    # repair_call may return either (ok, stdout, stderr) [back-compat] or
    # (ok, stdout, stderr, artifact_dir). When we know the artifact_dir we
    # stamp parse_status on its meta.json so the audit trail records the
    # repair outcome instead of leaving it at the default "pending".
    if len(repair_result) == 4:
        ok, stdout, _stderr, repair_artifact_dir = repair_result
    else:
        ok, stdout, _stderr = repair_result
        repair_artifact_dir = None

    if not ok:
        errors.append("repair call returned non-ok")
        if repair_artifact_dir is not None:
            update_artifact_meta(repair_artifact_dir, parse_status="repair_codex_failed")
        return ParseResult(status="failed", value=None, errors=errors, raw=raw)

    value2, errors2 = _validate(stdout, model_cls)
    if value2 is not None:
        if repair_artifact_dir is not None:
            update_artifact_meta(repair_artifact_dir, parse_status="repair_ok")
        return ParseResult(status="repaired", value=value2, errors=[], raw=stdout)
    if repair_artifact_dir is not None:
        update_artifact_meta(
            repair_artifact_dir,
            parse_status="repair_failed",
            validation_errors=errors2,
        )
    return ParseResult(status="failed", value=None, errors=errors + errors2, raw=stdout)


CODEX_TIMEOUT_S = 1500
# One extra attempt on a transient non-zero exit (network/auth hiccup).
# Timeouts are NOT retried: callers size them as wall-clock budgets
# (e.g. autosearch's CODEGEN_TIMEOUT_S) and a retry would double-spend.
CODEX_RETRIES = 1
CODEX_RETRY_BACKOFF_S = 5.0


def _invoke_codex(prompt: str, reasoning: str,
                  timeout_s: int | None = None) -> tuple[bool, str, str]:
    """Real codex invocation. Mockable via monkeypatch in tests."""
    repo_root = Path(__file__).resolve().parent.parent
    timeout = timeout_s or CODEX_TIMEOUT_S
    last: tuple[bool, str, str] = (False, "", "")
    for attempt in range(1 + CODEX_RETRIES):
        try:
            result = subprocess.run(
                [
                    "codex",
                    "exec",
                    "--skip-git-repo-check",
                    "-",
                    "-C",
                    str(repo_root),
                    "-s",
                    "read-only",
                    "-c",
                    f'model_reasoning_effort="{reasoning}"',
                ],
                input=prompt,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            if result.returncode == 0:
                return (True, result.stdout, result.stderr)
            last = (False, result.stdout, result.stderr)
        except subprocess.TimeoutExpired:
            return (False, "", f"codex timeout ({timeout}s)")
        except FileNotFoundError:
            return (False, "", "codex binary not found")
        if attempt < CODEX_RETRIES:
            time.sleep(CODEX_RETRY_BACKOFF_S * (attempt + 1))
    return last


def _next_seq_dir(stage_dir: Path) -> Path:
    """Claim <stage_dir>/<NN>/ atomically, where NN is the next index.

    mkdir(exist_ok=False) is the claim: when parallel callers race on the
    same stage (scoring batches, per-composition verification), the loser
    gets FileExistsError and takes the next index instead of sharing a dir.
    """
    stage_dir.mkdir(parents=True, exist_ok=True)
    n = sum(1 for p in stage_dir.iterdir() if p.is_dir()) + 1
    while True:
        candidate = stage_dir / f"{n:02d}"
        try:
            candidate.mkdir()
            return candidate
        except FileExistsError:
            n += 1


VALID_EFFORTS = ("minimal", "low", "medium", "high")


def stage_reasoning(stage: str, default: str = "high") -> str:
    """Per-stage reasoning-effort override via CONGRUITY_EFFORT_<STAGE>.

    The stage name is uppercased with non-alphanumerics collapsed to "_"
    and any per-composition suffix stripped, so `verify_l1_C3` reads
    CONGRUITY_EFFORT_VERIFY_L1. Cheap-model-shaped stages (L1 consistency,
    L4 classification) can be dialed down without a code change, e.g.
    CONGRUITY_EFFORT_VERIFY_L1=medium. Defaults preserve the calibrated
    high-effort behavior; invalid values are ignored.
    """
    key = re.sub(r"[^A-Za-z0-9]+", "_", stage).upper()
    key = re.sub(r"_C\d+$", "", key)  # verify_l1_C3 -> VERIFY_L1
    value = os.environ.get(f"CONGRUITY_EFFORT_{key}", "").strip().lower()
    return value if value in VALID_EFFORTS else default


def run_codex_with_artifacts(
    stage: str,
    prompt: str,
    run_dir: Path,
    reasoning: str = "high",
    timeout_s: int | None = None,
) -> tuple[bool, str, str, Path]:
    """Invoke codex and persist prompt, stdout, stderr, and meta.json
    under <run_dir>/llm_calls/<stage>/<NN>/. Returns
    (ok, stdout, stderr, artifact_dir).

    ``reasoning`` is the caller's default; a CONGRUITY_EFFORT_<STAGE> env
    var overrides it (see stage_reasoning).
    """
    reasoning = stage_reasoning(stage, reasoning)
    stage_dir = run_dir / "llm_calls" / stage
    artifact_dir = _next_seq_dir(stage_dir)

    (artifact_dir / "prompt.txt").write_text(prompt)
    t0 = time.monotonic()
    # Positional 2-arg call when no timeout override keeps monkeypatched
    # 2-arg fakes (tests) working.
    if timeout_s is None:
        ok, stdout, stderr = _invoke_codex(prompt, reasoning)
    else:
        ok, stdout, stderr = _invoke_codex(prompt, reasoning, timeout_s)
    duration_s = time.monotonic() - t0

    (artifact_dir / "stdout.txt").write_text(stdout)
    (artifact_dir / "stderr.txt").write_text(stderr)
    meta = {
        "stage": stage,
        "reasoning": reasoning,
        "exit_code": 0 if ok else 1,
        "duration_s": round(duration_s, 3),
        # Cost proxies: codex doesn't report token usage on stdout, so we
        # estimate at ~4 chars/token. Good enough to rank stages by spend
        # and to trend cost-per-run; not billing-grade.
        "prompt_chars": len(prompt),
        "stdout_chars": len(stdout),
        "est_prompt_tokens": len(prompt) // 4,
        "est_output_tokens": len(stdout) // 4,
        "parse_status": "pending",  # caller updates after schema validation
    }
    (artifact_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return ok, stdout, stderr, artifact_dir


def aggregate_llm_usage(run_dir: Path) -> dict:
    """Sum per-call meta.json cost proxies for a run.

    Returns {"calls", "duration_s", "est_prompt_tokens", "est_output_tokens",
    "by_stage": {stage: {...same keys...}}}. Stages with a per-composition
    suffix (verify_l1_C3) are grouped under their base name (verify_l1).
    """
    totals = {"calls": 0, "duration_s": 0.0,
              "est_prompt_tokens": 0, "est_output_tokens": 0}
    by_stage: dict[str, dict] = {}
    calls_root = run_dir / "llm_calls"
    if not calls_root.is_dir():
        return {**totals, "by_stage": by_stage}
    for meta_path in sorted(calls_root.glob("*/*/meta.json")):
        try:
            meta = json.loads(meta_path.read_text())
        except json.JSONDecodeError:
            continue
        base = re.sub(r"_C\d+$", "", str(meta.get("stage", meta_path.parent.parent.name)))
        bucket = by_stage.setdefault(base, {"calls": 0, "duration_s": 0.0,
                                            "est_prompt_tokens": 0,
                                            "est_output_tokens": 0})
        for tgt in (totals, bucket):
            tgt["calls"] += 1
            tgt["duration_s"] = round(tgt["duration_s"] + meta.get("duration_s", 0.0), 3)
            tgt["est_prompt_tokens"] += meta.get("est_prompt_tokens", 0)
            tgt["est_output_tokens"] += meta.get("est_output_tokens", 0)
    return {**totals, "by_stage": by_stage}


def update_artifact_meta(artifact_dir: Path, **fields) -> None:
    """Merge fields into <artifact_dir>/meta.json. Used by callers to record
    parse_status, validation_errors, etc., after parse_or_repair runs."""
    meta_path = artifact_dir / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    meta.update(fields)
    meta_path.write_text(json.dumps(meta, indent=2))
