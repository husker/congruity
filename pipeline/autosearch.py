"""Autosearch — the discovery loop on top of pack-bench verification.

One-shot verification (pipeline/autoverify.py) generates a candidate, grades
it once, and stops; no discovery system works that way. Autosearch wraps the
pack bench in a small evolutionary loop:

  1. Generate a population of candidate modules from the composition, each
     with a distinct emphasis.
  2. Evaluate every candidate on the pack's `search_command` — the TRAIN
     split (cheap, reduced budget).
  3. Select elites by signed train delta vs the home-field SOTA; refine each
     elite via codex with its per-instance score table (diagnose-and-improve).
  4. Repeat for a bounded number of generations; stop early on a train win.
  5. Confirm the single best candidate ONCE on `confirm_command` — the
     HOLDOUT split (disjoint instances, unseen solver seeds, full budget) —
     and record THAT result as the ledger label.

Anti-overfitting rules: the holdout is never evaluated during search, the
confirmation runs exactly once, and the recorded outcome comes only from it.
Codegen budget is population + generations x elites calls; every candidate
and score is persisted under experiments/auto/<run>-<cid>/search/.

Resumability: candidate files and an eval cache (evals.json, keyed by code
hash) persist in the search dir. A restarted run reuses existing candidate
files instead of re-paying codegen, and reuses cached bench results instead
of re-running the bench. Delete the search dir for a from-scratch rerun.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from datetime import datetime
from pathlib import Path

from pipeline.autoverify import (
    AUTO_EXPERIMENTS,
    CANDIDATE_PROMPT,
    COMPOSITIONS,
    _format_papers,
    apply_record_claims,
    build_outcome,
    execute_pack_bench,
    load_composition,
    parse_min_effect,
    record,
    structural_check,
)

# Rotating emphases keep the initial population diverse instead of K near-identical
# modules.
EMPHASES = [
    "Implement the composition faithfully but minimally: the smallest module "
    "that realizes its core mechanisms.",
    "Implement ONLY the composition's single most promising mechanism; drop "
    "validation/bookkeeping components that cannot reduce the metric.",
    "Hybridize the composition's core mechanism with the strongest local baseline: "
    "use the baseline as the engine and the mechanism to guide or restart it.",
    "Pursue the interpretation of the composition that differs MOST from the "
    "home-field SOTA's strategy, even if riskier.",
]

# Codegen must answer fast: candidates are population members, not one-shot
# masterpieces. A call that can't emit a module in this window is killed and
# counted as a failed candidate (C2's first run lost 50 minutes to two
# 1500s codex timeouts at high reasoning).
CODEGEN_TIMEOUT_S = 600
CODEGEN_SPEED_RULES = (
    "\nRESPONSE DISCIPLINE: Write the module directly and respond immediately. "
    "Do NOT run, test, or iterate on code. Do NOT explore the repository beyond "
    "what the interface names. Keep the module under ~250 lines. A working "
    "simple implementation now beats an elaborate one later.\n"
)

FUNCTION_PROMPT = """You are evolving ONE function inside a frozen, human-written solver
skeleton (FunSearch-style: the skeleton guarantees validity and budget
accounting; only the priority function between the markers is yours).

PROBLEM: {problem}

COMPOSITION (the idea to draw on):
{claim}

THE SKELETON (read the driver to understand exactly how your function is called):
{skeleton}

HARD RULES:
  1. Output ONLY the replacement for the region between
     `# <PRIORITY-FUNCTION>` and `# </PRIORITY-FUNCTION>`: a complete
     `def priority(...)` with the SAME signature. No markers, no fences,
     no imports, no other top-level code.
  2. Pure function of its arguments — no globals, no I/O, no randomness
     beyond the provided `noise` argument.
  3. Fast: it runs millions of times per attempt.
{fulltext}
VARIANT DIRECTIVE: {emphasis}
"""

FUNCTION_REFINE_PROMPT = """You previously wrote the priority function below for a frozen solver
skeleton. Improve it using the per-instance results.

CURRENT PRIORITY FUNCTION:
{code}

PER-INSTANCE RESULTS (metric: {metric}, {direction} is better):
{score_table}

Same hard rules: output ONLY a complete `def priority(...)` with the same
signature — no markers, no fences, no other code.
{fulltext}
"""

_SKELETON_MARK_OPEN = "# <PRIORITY-FUNCTION>"
_SKELETON_MARK_CLOSE = "# </PRIORITY-FUNCTION>"


def splice_skeleton(skeleton: str, function_code: str) -> str:
    """Replace the marked region of a skeleton with codegen's function."""
    i = skeleton.index(_SKELETON_MARK_OPEN)
    j = skeleton.index(_SKELETON_MARK_CLOSE) + len(_SKELETON_MARK_CLOSE)
    body = function_code.strip()
    if "def priority(" not in body:
        raise ValueError("codegen output does not define priority()")
    return (skeleton[:i] + _SKELETON_MARK_OPEN + "\n" + body + "\n"
            + _SKELETON_MARK_CLOSE + skeleton[j:])


def extract_priority(candidate_code: str) -> str:
    """The marked region of a spliced candidate (for refine prompts)."""
    i = candidate_code.index(_SKELETON_MARK_OPEN) + len(_SKELETON_MARK_OPEN)
    j = candidate_code.index(_SKELETON_MARK_CLOSE)
    return candidate_code[i:j].strip()


REFINE_PROMPT = """You previously wrote a CANDIDATE module for a benchmark. Improve it.

THE COMPOSITION IT IMPLEMENTS:
{claim}

CANDIDATE INTERFACE (unchanged, your module MUST still satisfy exactly this):
{candidate_interface}

CURRENT CANDIDATE SOURCE:
```python
{code}
```

TRAIN RESULTS (metric: {metric}; {direction} is better):
{score_table}

Diagnose where the candidate loses or merely ties the home-field SOTA, then emit
ONE improved module. Make a real algorithmic change targeted at the weakest
instances — not a parameter tweak. Keep what works.

HARD RULES (same as before):
  1. Emit ONLY the Python file contents, no markdown fences, no commentary.
  2. Importing the domain's existing baseline implementations is allowed.
  3. No network, file writes, subprocesses, or stdout. Deterministic in `seed`.
  4. Output is re-validated by the harness; invalid output fails the run.
"""


def fitness(result: dict) -> float | None:
    """Signed train delta vs the home-field SOTA (>0 = candidate better).

    None for crashed/contract-broken/non-finite results, so failures sort last
    and can never be selected as elites.
    """
    if "_error" in result:
        return None
    scores = result.get("scores") or {}
    comp = result.get("composition_method")
    sota = result.get("home_field_sota_method")
    if comp not in scores or sota not in scores:
        return None
    c, s = scores[comp], scores[sota]
    if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in (c, s)):
        return None
    return (c - s) if result.get("higher_is_better", True) else (s - c)


def record_fitness(result: dict, n: int) -> float | None:
    """Best value the candidate achieved at target dimension/size n.

    Read from the bench's record_claims (params.n == n). None when absent,
    crashed, or non-finite — failures sort last, same as fitness().
    """
    if "_error" in result:
        return None
    vals = [c.get("value") for c in (result.get("record_claims") or [])
            if (c.get("params") or {}).get("n") == n]
    vals = [v for v in vals if isinstance(v, (int, float)) and math.isfinite(v)]
    return max(vals) if vals else None


def score_table(result: dict) -> str:
    """Per-instance candidate-vs-SOTA table for the refine prompt."""
    per = result.get("per_instance") or {}
    comp = result.get("composition_method", "candidate")
    sota = result.get("home_field_sota_method", "")
    cand_by, sota_by = per.get(comp, {}), per.get(sota, {})
    if not cand_by or not sota_by:
        return f"  overall: candidate={result.get('scores', {}).get(comp)} " \
               f"sota={result.get('scores', {}).get(sota)} (no per-instance breakdown)"
    lines = [f"  {'instance':16s} {'candidate':>10s} {'sota':>10s}"]
    for inst in sorted(set(cand_by) | set(sota_by)):
        c = cand_by.get(inst, float("nan"))
        s = sota_by.get(inst, float("nan"))
        lines.append(f"  {inst:16s} {c:10.2f} {s:10.2f}")
    return "\n".join(lines)


def _strip_fences(code: str) -> str:
    code = code.strip()
    if code.startswith("```"):
        code = re.sub(r"^```[a-z]*\n|\n```$", "", code)
    return code


def _codegen(prompt: str, dest: Path, stage: str) -> Path:
    from pipeline.llm_io import run_codex_with_artifacts, update_artifact_meta
    ok, stdout, _stderr, art = run_codex_with_artifacts(
        stage=stage, prompt=prompt + CODEGEN_SPEED_RULES, run_dir=dest.parent,
        reasoning="medium", timeout_s=CODEGEN_TIMEOUT_S)
    dest.write_text(_strip_fences(stdout))
    update_artifact_meta(art, parse_status="ok" if ok else "codex_failed")
    return dest


def generate_initial(comp: dict, recipe: dict, pack, problem: str,
                     dest: Path, emphasis: str, stage: str,
                     extra_context: str = "",
                     skeleton: str | None = None) -> Path:
    if skeleton is not None:
        prompt = FUNCTION_PROMPT.format(
            problem=problem, claim=comp.get("synthesis_claim", ""),
            skeleton=skeleton, fulltext=extra_context, emphasis=emphasis)
        _codegen(prompt, dest, stage)
        dest.write_text(splice_skeleton(skeleton, dest.read_text()))
        return dest
    shape = pack.pack.problem_shape
    prompt = CANDIDATE_PROMPT.format(
        problem=problem,
        claim=comp.get("synthesis_claim", ""),
        papers=_format_papers(comp),
        home_field_sota=recipe.get("home_field_sota", "(unspecified)"),
        metric=pack.evaluator.metric_primary,
        objective=shape.objective,
        constraints="\n".join(f"    - {c}" for c in shape.constraints),
        gotchas=pack.gotchas.strip() or "(none)",
        candidate_interface=pack.evaluator.candidate_interface,
    ) + extra_context + f"\nVARIANT DIRECTIVE: {emphasis}\n"
    return _codegen(prompt, dest, stage)


def refine(comp: dict, pack, parent_code: str, parent_result: dict,
           dest: Path, stage: str, extra_context: str = "",
           skeleton: str | None = None) -> Path:
    if skeleton is not None:
        prompt = FUNCTION_REFINE_PROMPT.format(
            code=extract_priority(parent_code),
            metric=pack.evaluator.metric_primary,
            direction="higher" if pack.evaluator.higher_is_better else "lower",
            score_table=score_table(parent_result), fulltext=extra_context)
        _codegen(prompt, dest, stage)
        dest.write_text(splice_skeleton(skeleton, dest.read_text()))
        return dest
    prompt = REFINE_PROMPT.format(
        claim=comp.get("synthesis_claim", ""),
        candidate_interface=pack.evaluator.candidate_interface,
        code=parent_code[:16000],
        metric=pack.evaluator.metric_primary,
        direction="higher" if pack.evaluator.higher_is_better else "lower",
        score_table=score_table(parent_result),
    ) + extra_context
    return _codegen(prompt, dest, stage)


# Seeding-experiment arms (roadmap #5). "composition" is the normal mode and
# the only one that writes the ledger — the other two are controls that answer
# "does literature seeding actually out-discover the alternatives?".
ARM_CLAIMS = {
    "unseeded": (
        "(control arm — NO literature seed) Design the best method you can "
        "from first principles for this problem. Ignore any notion of "
        "implementing a published composition; just win on the metric."),
    "evolve-sota": (
        "(control arm — evolve the SOTA) Start from the home-field SOTA "
        "algorithm itself and make ONE principled algorithmic modification "
        "that could improve the metric. Each candidate should try a "
        "different modification."),
}


def orchestrate_search(run_id: str, cid: str, now: datetime, *, pack_id: str,
                       population: int = 4, generations: int = 3,
                       elites: int = 2,
                       seed_candidate: str | None = None,
                       arm: str = "composition",
                       record_dim: int | None = None,
                       fulltext: bool = False,
                       function_skeleton: str | None = None) -> dict:
    """Run the search loop; record exactly one ledger outcome (from HOLDOUT).

    Non-"composition" arms are experiment controls: they run the identical
    loop and budget but skip the ledger (their result is not an outcome OF
    the composition) — the summary lands in the search dir instead.
    """
    if arm not in ("composition", *ARM_CLAIMS):
        return {"error": f"unknown arm {arm!r}"}
    # Record mode optimizes tail behavior (best length at one target n) and
    # NEVER writes the ledger: its output is a stronger method + certificate
    # to hand to the record miner, not a labeled outcome.
    write_ledger = arm == "composition" and record_dim is None
    comp = load_composition(run_id, cid)
    if comp is None:
        return {"error": f"composition {run_id}/{cid} not found"}
    recipe = (comp.get("verification") or {}).get("recipe") or {}
    if not recipe:
        return {"error": "no recipe on composition (run compose --with-recipes first)"}
    # Control arms see the same problem/recipe/pack but not the composition.
    prompt_comp = comp if arm == "composition" else {
        "synthesis_claim": ARM_CLAIMS[arm], "papers": []}
    problem = json.loads(
        (COMPOSITIONS / run_id / "state.json").read_text()).get("problem", "")

    # Full-text mode: OA method excerpts from the constituent papers, shown
    # to codegen alongside the abstract-level composition. Control arms skip
    # it (they don't see the composition either).
    skeleton_text: str | None = None
    if function_skeleton:
        skeleton_text = Path(function_skeleton).read_text()
        if _SKELETON_MARK_OPEN not in skeleton_text:
            return {"error": f"skeleton has no {_SKELETON_MARK_OPEN} marker: "
                             f"{function_skeleton}"}
        if not seed_candidate:
            seed_candidate = function_skeleton  # the skeleton is runnable as-is

    fulltext_block = ""
    if fulltext and arm == "composition":
        from pipeline.fulltext import format_fulltext_block, fulltext_for_composition
        corpus_path = COMPOSITIONS / run_id / "corpus.json"
        corpus = json.loads(corpus_path.read_text()) if corpus_path.exists() else []
        excerpts = fulltext_for_composition(comp, corpus, COMPOSITIONS / run_id)
        print(f"  fulltext: method excerpts for {len(excerpts)}/"
              f"{len(comp.get('papers', []))} constituent paper(s)"
              + (f" ({', '.join(sorted(excerpts))})" if excerpts else ""), flush=True)
        fulltext_block = format_fulltext_block(excerpts, comp)

    from pipeline.computable import load_pack, preflight_pack
    try:
        pack = load_pack(pack_id)
    except (FileNotFoundError, ValueError) as e:
        return {"error": str(e)}
    pre = preflight_pack(pack_id)
    if not pre.search_ready:
        return {"error": f"pack {pack_id!r} is not search-ready "
                         "(evaluator.search_command / confirm_command)"}
    record_target: float | None = None
    if record_dim is not None:
        if not pack.evaluator.record_command:
            return {"error": f"pack {pack_id!r} has no evaluator.record_command "
                             "(required for --record-dim)"}
        from pipeline.computable import compare_to_record
        probe = compare_to_record(pack, {"n": record_dim}, 0.0) or {}
        record_target = probe.get("record_value")  # None = uncovered dim
        print(f"  record mode: dim {record_dim}, published record "
              f"{record_target if record_target is not None else 'none on file'}")

    suffix = "search" if arm == "composition" else f"search_{arm}"
    if record_dim is not None:
        suffix = f"{suffix}_record{record_dim}"
    search_dir = AUTO_EXPERIMENTS / f"{run_id}-{cid}" / suffix
    search_dir.mkdir(parents=True, exist_ok=True)
    min_eff = parse_min_effect(recipe.get("min_meaningful_effect_size"))

    history: list[dict] = []
    seen_hashes: set[str] = set()
    # (fitness, path, result) for every evaluated candidate; None fitness sorts last.
    evaluated: list[tuple[float | None, Path, dict]] = []

    # Resume support: bench results keyed by candidate code hash survive a
    # killed run. On restart, existing candidate files skip codegen and
    # cached hashes skip the bench. Delete the search dir to run fresh.
    evals_path = search_dir / "evals.json"
    eval_cache: dict[str, dict] = {}
    if evals_path.exists():
        try:
            eval_cache = json.loads(evals_path.read_text())
        except json.JSONDecodeError:
            eval_cache = {}
    if eval_cache:
        print(f"  resuming: {len(eval_cache)} cached evaluation(s) from a "
              f"prior run in {evals_path}", flush=True)

    def finish(summary: dict) -> dict:
        summary["arm"] = arm
        (search_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        return summary

    def evaluate(path: Path, gen: int, label: str) -> float | None:
        code_hash = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
        if code_hash in seen_hashes:
            history.append({"gen": gen, "candidate": path.name,
                            "skipped": "duplicate code"})
            return None
        seen_hashes.add(code_hash)
        cached = code_hash in eval_cache
        if cached:
            result = eval_cache[code_hash]
        else:
            template = (pack.evaluator.record_command if record_dim is not None
                        else pack.evaluator.search_command)
            result = execute_pack_bench(
                template.format(candidate=str(path), n=record_dim))
            eval_cache[code_hash] = result
            evals_path.write_text(json.dumps(eval_cache, indent=2))
        f = (record_fitness(result, record_dim) if record_dim is not None
             else fitness(result))
        evaluated.append((f, path, result))
        history.append({"gen": gen, "candidate": path.name, "label": label,
                        "fitness": f,
                        "cached": cached,
                        "error": result.get("_error"),
                        "scores": result.get("scores")})
        (search_dir / "history.json").write_text(json.dumps(history, indent=2))
        print(f"  [gen {gen}] {path.name} ({label}): fitness="
              f"{f if f is not None else 'FAILED'}"
              f"{' (cached)' if cached else ''}", flush=True)
        return f

    # Generation 0: diverse initial population (+ optional human seed).
    print(f"autosearch {run_id}/{cid} on pack {pack_id} (arm={arm}): "
          f"population={population} generations={generations} elites={elites}")
    if seed_candidate:
        seed_path = Path(seed_candidate)
        if not seed_path.exists():
            return {"error": f"seed candidate not found: {seed_path}"}
        evaluate(seed_path, 0, "seed")
    for i in range(population):
        emphasis = EMPHASES[i % len(EMPHASES)]
        dest = search_dir / f"g0_c{i}.py"
        if not (dest.exists() and dest.stat().st_size > 0):
            generate_initial(prompt_comp, recipe, pack, problem, dest, emphasis,
                             stage=f"autosearch_{cid}_{arm}_g0c{i}",
                             extra_context=fulltext_block,
                             skeleton=skeleton_text)
        evaluate(dest, 0, f"initial:{i}")

    def best_first() -> list[tuple[float | None, Path, dict]]:
        return sorted(evaluated,
                      key=lambda t: t[0] if t[0] is not None else -math.inf,
                      reverse=True)

    # Refinement generations.
    for gen in range(1, generations + 1):
        ranked = [t for t in best_first() if t[0] is not None]
        if record_dim is not None:
            # Only an outright new record stops a record search early.
            if (ranked and record_target is not None
                    and ranked[0][0] is not None and ranked[0][0] > record_target):
                print(f"  early stop: EXCEEDED published record at gen {gen - 1} "
                      f"({ranked[0][0]:.0f} > {record_target:.0f})")
                break
        elif ranked and ranked[0][0] is not None and ranked[0][0] >= max(min_eff, 1e-12):
            print(f"  early stop: train win at gen {gen - 1} "
                  f"(delta {ranked[0][0]:.4f} >= min effect {min_eff})")
            break
        if not ranked:
            print("  no viable candidates to refine; stopping search")
            break
        for j, (_f, parent_path, parent_result) in enumerate(ranked[:elites]):
            dest = search_dir / f"g{gen}_e{j}.py"
            if not (dest.exists() and dest.stat().st_size > 0):
                refine(prompt_comp, pack, parent_path.read_text(), parent_result,
                       dest, stage=f"autosearch_{cid}_{arm}_g{gen}e{j}",
                       extra_context=fulltext_block,
                       skeleton=skeleton_text)
            evaluate(dest, gen, f"refined-from:{parent_path.name}")

    ranked = best_first()
    if record_dim is not None:
        # Record mode terminal: no holdout, no ledger. Report the best
        # method + certificate and hand off to the record miner.
        if not ranked or ranked[0][0] is None:
            return finish({"mode": "record", "record_dim": record_dim,
                           "error": "no candidate produced a finite record fitness",
                           "candidates_evaluated": len(evaluated)})
        best_f, best_path, best_result = ranked[0]
        cert = (best_result.get("best_sequences") or {}).get(f"q{record_dim}")
        cert_path = None
        if cert:
            cert_path = search_dir / f"record_q{record_dim}_len{int(best_f)}.json"
            cert_path.write_text(json.dumps(
                {"n": record_dim, "length": int(best_f), "seq": cert,
                 "candidate": str(best_path), "run_id": run_id, "cid": cid,
                 "at": now.isoformat()}, indent=2))
        from pipeline.computable import compare_to_record
        cmp = compare_to_record(pack, {"n": record_dim}, float(best_f)) or {}
        print(f"  record search done: best {best_f:.0f} at Q{record_dim} "
              f"({cmp.get('verdict', 'no-record')}, record {record_target})")
        print(f"  next: mine with it —\n"
              f"    uv run python -m experiments.snake_in_box.record_run "
              f"--candidate {best_path} --dims {record_dim} --workers 4")
        return finish({"mode": "record", "record_dim": record_dim,
                       "best_fitness": best_f,
                       "record_value": record_target,
                       "gap": (best_f - record_target) if record_target is not None else None,
                       "verdict": cmp.get("verdict", "no-record"),
                       "best_candidate": str(best_path),
                       "certificate": str(cert_path) if cert_path else None,
                       "candidates_evaluated": len(evaluated)})

    if not ranked or ranked[0][0] is None:
        # Nothing ever produced a finite train fitness: quarantined invalid.
        outcome = build_outcome(run_id, cid, comp, recipe,
                                {"scores": {}, "n_trials": 0}, now,
                                audit=_pack_audit(pack_id))
        outcome.valid, outcome.grade = False, "invalid"
        outcome.invalid_reason = "autosearch: no candidate produced a finite train fitness"
        outcome.confidence = "quarantined"
        if write_ledger:
            record(outcome)
        return finish({"grade": "invalid",
                       "reason": outcome.invalid_reason,
                       "evaluated": len(evaluated), "pack": pack_id})

    best_fitness, best_path, _ = ranked[0]
    assert best_fitness is not None  # guaranteed by the early return above

    # ONE holdout confirmation; this result — not the train one — is the label.
    print(f"  confirming {best_path.name} on holdout "
          f"(train fitness {best_fitness:.4f})")
    # Confirmation runs the full-budget bench plus the SOTA comparator AND the
    # candidate's ablation; candidates with heavy per-eval overhead can take
    # 700x the comparator's wall-clock at equal eval budget (heilbronn C2:
    # 6s vs 75min at 240k evals). Size confirm_command by measured wall-clock;
    # this cap is the backstop, not the budget.
    confirm_result = execute_pack_bench(
        pack.evaluator.confirm_command.format(candidate=str(best_path)),
        timeout_s=10800)

    audit = _pack_audit(pack_id)
    invalid_reason = ""
    if "_error" in confirm_result:
        invalid_reason = confirm_result["_error"]
    else:
        violation = structural_check(confirm_result, pack)
        if violation:
            invalid_reason = f"pack contract violation: {violation}"

    search_notes = (f"autosearch: {len(evaluated)} candidates over "
                    f"{generations} max generations; best train delta "
                    f"{best_fitness:.4f}; label from one holdout confirmation")

    if invalid_reason:
        outcome = build_outcome(run_id, cid, comp, recipe,
                                {"scores": {}, "n_trials": 0}, now, audit=audit)
        outcome.valid, outcome.invalid_reason, outcome.grade = False, invalid_reason, "invalid"
        outcome.confidence = "quarantined"
        outcome.notes = search_notes
        if write_ledger:
            record(outcome)
        return finish({"grade": "invalid", "reason": invalid_reason,
                       "best_candidate": str(best_path), "pack": pack_id})

    confirm_result["notes"] = (str(confirm_result.get("notes", "")) +
                               " | " + search_notes)
    outcome = build_outcome(run_id, cid, comp, recipe, confirm_result, now,
                            audit=audit)
    apply_record_claims(outcome, confirm_result, pack)
    if write_ledger:
        record(outcome)
    return finish({
        "grade": outcome.grade, "valid": outcome.valid,
        "confidence": outcome.confidence,
        "beats_home_field": outcome.beats_home_field,
        "train_delta_best": round(best_fitness, 4),
        "holdout_delta": round(outcome.delta_vs_home, 4),
        "composition": outcome.composition_score,
        "home_field_sota": outcome.home_field_score,
        "candidates_evaluated": len(evaluated),
        "best_candidate": str(best_path), "pack": pack_id,
    })


def _pack_audit(pack_id: str) -> dict:
    return {
        "fair": True, "sota_faithful": True, "issues": [],
        "notes": f"pack-bench mode: harness, baselines, and home-field SOTA "
                 f"from computable pack '{pack_id}' (human-written); "
                 f"label from held-out split only",
    }
