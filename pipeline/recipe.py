"""Stage 6: verification recipe generation.

Fires on WEAK + PURELY_TESTABLE compositions (the actionable case).
Produces a concrete experiment plan: named baselines, benchmark dataset,
primary + secondary metrics, minimum meaningful effect size, expected
runtime, likely failure mode, stop rule.

Output validated against pipeline.schemas.VerificationRecipe and persisted
under <run_dir>/llm_calls/recipe_<cid>/NN/ like every other LLM stage.
"""

from __future__ import annotations

from pathlib import Path

from pipeline.llm_io import (
    parse_or_repair,
    run_codex_with_artifacts,
    update_artifact_meta,
)
from pipeline.schemas import VerificationRecipe

RECIPE_PROMPT_TEMPLATE = """You are designing a small, runnable verification experiment for a
proposed compositional synthesis that L3 marked WEAK and L4 marked PURELY_TESTABLE.

PROBLEM: {problem}

SYNTHESIS CLAIM:
{claim}

CONSTITUENT PAPERS:
{papers_block}

L3 ADVERSARIAL VERDICT REASON:
{l3_reason}

YOUR TASK
This composition imports a technique from one field into another. The ONLY result
worth reporting is whether the import beats the DESTINATION field's OWN best-known
method — not a generic or weak baseline. So:

  - Name `home_field_sota`: the strongest standard method a specialist in the
    destination field would already reach for on this exact problem.
  - `named_baselines` MUST include that home-field SOTA (plus any others). If the
    composition can't be designed to test against it, say so in likely_failure_mode.
  - `min_meaningful_effect_size` and `stop_rule` are defined RELATIVE TO the
    home-field SOTA — beating only generic baselines is a near-miss, not a win.

Emit a strict JSON object with these fields and no others. Keep each field tight; the
engineer reading this should be able to start implementing within an hour.

OUTPUT — JSON ONLY:
{{
  "home_field_sota": "<the destination field's strongest standard method for this problem>",
  "named_baselines": ["<home-field SOTA (required)>", "<other canonical method>", "..."],
  "benchmark_dataset": "<existing public dataset + retrieval link>",
  "metric_primary": "<single string, e.g. recall@10>",
  "metric_secondary": ["<one or two secondary metrics>"],
  "min_meaningful_effect_size": "<numeric, RELATIVE TO home_field_sota, e.g. +0.02 recall@10 over SOTA>",
  "expected_runtime": "<range; CPU/GPU; e.g. 30 minutes single CPU>",
  "likely_failure_mode": "<one short paragraph>",
  "stop_rule": "<abandon as clean negative if it cannot beat home_field_sota by min effect>"
}}
"""


def _papers_block(composition: dict) -> str:
    lines = []
    for p in composition.get("papers", []):
        lines.append(f"- {p.get('paper_id', '?')}: {p.get('role', '?')}")
    return "\n".join(lines)


def generate_recipe(composition: dict, state: dict, run_dir: Path) -> dict:
    """Build a verification recipe for one composition.

    Returns the recipe dict (model_dump output). On parse failure returns
    a dict with an `error` key; callers decide whether to retry or skip.
    """
    cid = composition.get("id", "unknown")
    stage = f"recipe_{cid}"
    l3 = composition.get("verification", {}).get("L3", {})
    prompt = RECIPE_PROMPT_TEMPLATE.format(
        problem=state.get("problem", ""),
        claim=composition.get("synthesis_claim", ""),
        papers_block=_papers_block(composition),
        l3_reason=l3.get("verdict_reason", "") or l3.get("error", ""),
    )

    ok, stdout, stderr, artifact_dir = run_codex_with_artifacts(
        stage=stage, prompt=prompt, run_dir=run_dir, reasoning="high",
    )
    if not ok:
        update_artifact_meta(artifact_dir, parse_status="codex_failed")
        return {"error": stderr}

    def _repair(p: str) -> tuple[bool, str, str]:
        return run_codex_with_artifacts(
            stage=stage + "_repair", prompt=p, run_dir=run_dir, reasoning="high",
        )[:3]

    result = parse_or_repair(stdout, VerificationRecipe, repair_call=_repair)
    update_artifact_meta(
        artifact_dir,
        parse_status=result.status,
        validation_errors=result.errors,
    )
    if result.value is None:
        return {"error": "schema_validation_failed", "validation_errors": result.errors}
    return result.value.model_dump()
