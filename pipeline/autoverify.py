"""Automated verification loop — brick #2 of the moat (the flywheel enabler).

Runs a composition against a computable pack's human-written benchmark and
records a labeled outcome — at scale, so the verified-transfer catalog (and the
outcome predictor) accrue without a manual session per composition.

Design constraint: auto-generated benchmarks proved to be a liability — codex
scaffolds invented strawman SOTAs, ignored local baselines, and yielded 0
trustworthy labels (findings/autoverify-flywheel-verdict-20260530.md,
findings/graph-coloring-domain-20260611.md). So the harness is never generated:

  - The benchmark harness, datasets, baselines, validator, and the home-field
    SOTA come from a computable pack (computable_packs/<id>/), written and
    tested by a human. Fairness holds by construction.
  - Codegen is ONE contained step (generate_candidate) that may only emit the
    candidate solver module behind the pack's declared interface. A broken
    candidate becomes an invalid, quarantined outcome — there is no repair loop.
  - A validity gate rejects results that didn't include the home-field SOTA,
    didn't run, or returned non-finite/degenerate scores.
  - Every recorded label is tagged `provenance="auto"` and `confidence`, kept
    DISTINCT from human-verified findings/INDEX.md. Auto-labels are leads, not
    proof; a human-verified subset stays the gold standard.

This module is the deterministic spine (contract + grade + ledger) plus that
one guarded codegen step.
"""

from __future__ import annotations

import json
import random
import re
import shlex
import subprocess
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

# Cross-platform exclusive file lock for the outcomes-ledger append.
# POSIX uses fcntl.flock; Windows uses msvcrt.locking; neither present ->
# best-effort no lock (rather than crashing import on an unknown platform).
try:
    import fcntl as _fcntl
except ImportError:  # pragma: no cover - Windows
    _fcntl = None
try:
    import msvcrt as _msvcrt
except ImportError:  # pragma: no cover - POSIX
    _msvcrt = None


@contextmanager
def _exclusive_lock(f):
    """Hold an exclusive advisory lock on open file ``f`` for the block.

    On Windows the lock is a 1-byte region at offset 0 used purely as a
    mutex token; writes still land at EOF because ``f`` is opened in append
    mode. If no locking primitive is available the append proceeds unlocked.
    """
    if _fcntl is not None:
        _fcntl.flock(f.fileno(), _fcntl.LOCK_EX)
        try:
            yield
        finally:
            _fcntl.flock(f.fileno(), _fcntl.LOCK_UN)
    elif _msvcrt is not None:
        f.seek(0)
        _msvcrt.locking(f.fileno(), _msvcrt.LK_LOCK, 1)
        try:
            yield
        finally:
            f.seek(0)
            _msvcrt.locking(f.fileno(), _msvcrt.LK_UNLCK, 1)
    else:  # pragma: no cover - exotic platform
        yield

REPO = Path(__file__).resolve().parent.parent
COMPOSITIONS = REPO / "data" / "pipeline" / "compositions"
OUTCOMES = REPO / "data" / "pipeline" / "outcomes.jsonl"
AUTO_EXPERIMENTS = REPO / "experiments" / "auto"

# A pack bench's --json output (last stdout line) must match this shape.
RESULT_CONTRACT = {
    "metric": "str — name of the scored metric",
    "higher_is_better": "bool",
    "scores": "dict[str,float] — method label -> metric value; MUST include both labels below",
    "composition_method": "str — the key in scores for the composition's method",
    "home_field_sota_method": "str — the key in scores for the home-field SOTA",
    "n_trials": "int — independent trials/seeds aggregated",
    "notes": "str — one line",
}


@dataclass
class Outcome:
    run_id: str
    cid: str
    metric: str
    higher_is_better: bool
    scores: dict
    composition_method: str
    home_field_sota_method: str
    composition_score: float
    home_field_score: float
    delta_vs_home: float          # signed improvement over home-field SOTA (>0 = better)
    min_effect: float
    beats_home_field: bool
    grade: str                    # transfer-win | partial | negative | invalid
    n_trials: int
    transfer_distance: int        # distinct source fields (0 if unknown)
    valid: bool
    invalid_reason: str
    provenance: str               # "auto"
    confidence: str               # quarantined | low | medium (auto ceiling; human="high")
    audit: dict                   # fairness verdict (fair / sota_faithful / issues)
    timestamp: str
    notes: str


# --------------------------------------------------------------------------- #
# Parsing helpers
# --------------------------------------------------------------------------- #
def parse_min_effect(s: str | float | None) -> float:
    """Best-effort numeric min effect size from the recipe string (default 0.0)."""
    if isinstance(s, (int, float)):
        return float(s)
    if not s:
        return 0.0
    m = re.search(r"[-+]?\d*\.?\d+", str(s))
    return abs(float(m.group())) if m else 0.0


def load_composition(run_id: str, cid: str) -> dict | None:
    sp = COMPOSITIONS / run_id / "state.json"
    if not sp.exists():
        return None
    return next((c for c in json.loads(sp.read_text()).get("compositions", [])
                 if c.get("id") == cid), None)


# --------------------------------------------------------------------------- #
# Grading — the rigor core
# --------------------------------------------------------------------------- #
def paired_deltas(result: dict) -> list[float] | None:
    """Per-(instance, seed) candidate-vs-SOTA deltas, sign-adjusted (>0 = better).

    Built from the bench's `per_trial` traces; pairs where either side failed
    are dropped. None when traces are absent (older benches).
    """
    per = result.get("per_trial") or {}
    cand_by = per.get(result.get("composition_method"))
    sota_by = per.get(result.get("home_field_sota_method"))
    if not cand_by or not sota_by:
        return None
    hib = bool(result.get("higher_is_better", True))
    deltas: list[float] = []
    for inst, cand_ks in cand_by.items():
        for c, s in zip(cand_ks, sota_by.get(inst) or []):
            if c is None or s is None:
                continue
            deltas.append(float(c - s) if hib else float(s - c))
    return deltas or None


def bootstrap_ci(deltas: list[float], iters: int = 2000,
                 alpha: float = 0.05, seed: int = 0) -> tuple[float, float]:
    """Seeded percentile bootstrap CI for the mean of paired deltas."""
    rng = random.Random(seed)
    n = len(deltas)
    means = sorted(
        sum(rng.choice(deltas) for _ in range(n)) / n for _ in range(iters))
    lo = means[int(alpha / 2 * iters)]
    hi = means[min(iters - 1, int((1 - alpha / 2) * iters))]
    return lo, hi


# Paired trials needed before the significance requirement kicks in; below
# this the bootstrap is meaningless and the mean-only rule applies.
MIN_TRIALS_FOR_CI = 6


def grade(result: dict, recipe: dict) -> tuple[str, bool, str]:
    """Return (grade, valid, invalid_reason).

    Validity gate first; then degenerate (wrapper) detection; then grade the
    composition vs the home-field SOTA — with a paired-bootstrap significance
    requirement when per-trial traces are available.
    """
    scores = result.get("scores") or {}
    comp = result.get("composition_method")
    sota = result.get("home_field_sota_method")
    if not isinstance(scores, dict) or len(scores) < 2:
        return "invalid", False, "fewer than 2 method scores"
    if comp not in scores:
        return "invalid", False, f"composition method '{comp}' not in scores"
    if sota not in scores:
        return "invalid", False, f"home-field SOTA '{sota}' not in scores"
    vals = [scores[comp], scores[sota]]
    if any(not isinstance(v, (int, float)) or v != v or abs(v) == float("inf")
           for v in vals):
        return "invalid", False, "non-finite composition/SOTA score"
    if (result.get("n_trials") or 0) < 1:
        return "invalid", False, "n_trials < 1"

    # Wrapper detection: identical per-trial behavior to the SOTA on every
    # instance and seed means the candidate IS the SOTA (or trivially defers
    # to it). A tie that close is not evidence the idea added anything.
    per = result.get("per_trial") or {}
    if per.get(comp) and per.get(comp) == per.get(sota):
        return "degenerate", True, ""

    hib = bool(result.get("higher_is_better", True))
    c, s = float(scores[comp]), float(scores[sota])
    min_eff = parse_min_effect(recipe.get("min_meaningful_effect_size"))

    deltas = paired_deltas(result)
    if deltas is not None and len(deltas) >= MIN_TRIALS_FOR_CI:
        delta = sum(deltas) / len(deltas)
        ci_lo, _ = bootstrap_ci(deltas)
        # A win must be big enough AND statistically separated from zero.
        beats_home = delta >= min_eff and delta > 0 and ci_lo > 0
    else:
        delta = (c - s) if hib else (s - c)    # >0 means composition is better
        beats_home = delta >= min_eff and delta > 0

    others = [v for k, v in scores.items()
              if k not in (comp, sota, result.get("ablated_method"))
              and isinstance(v, (int, float))]
    beats_any_other = any((c > o) if hib else (c < o) for o in others)

    if beats_home:
        return "transfer-win", True, ""
    if beats_any_other or (delta > 0):
        return "partial", True, ""
    return "negative", True, ""


def transfer_distance_of(comp: dict, run_id: str) -> int:
    """Distinct source fields of the composition's papers (0 if unavailable)."""
    try:
        from scripts.transfer_distance import build_topic_field_map, load_corpus_fields
        pf = load_corpus_fields(COMPOSITIONS / run_id, build_topic_field_map())
        fields = {pf.get(p.get("paper_id")) for p in comp.get("papers", [])}
        return len({f for f in fields if f and f != "?"})
    except Exception:
        return 0


def confidence_from_audit(valid: bool, audit: dict | None) -> str:
    """Deterministic confidence for an auto-label (human verification = 'high').

    Auto ceiling is 'medium', only when the harness is fair AND the home-field
    SOTA is faithful (not a strawman) — in pack-bench mode both hold by
    construction because the harness is human-written. An invalid result
    quarantines the label regardless of the grade.
    """
    if not valid:
        return "quarantined"
    a = audit or {}
    if a.get("fair") is False:
        return "quarantined"
    if a.get("fair") and a.get("sota_faithful"):
        return "medium"
    return "low"


def build_outcome(run_id: str, cid: str, comp: dict, recipe: dict,
                  result: dict, now: datetime, audit: dict | None = None) -> Outcome:
    g, valid, reason = grade(result, recipe)
    scores = result.get("scores") or {}
    comp_m = result.get("composition_method", "")
    sota_m = result.get("home_field_sota_method", "")
    hib = bool(result.get("higher_is_better", True))
    cs = float(scores.get(comp_m, float("nan"))) if comp_m in scores else float("nan")
    ss = float(scores.get(sota_m, float("nan"))) if sota_m in scores else float("nan")
    delta = (cs - ss) if hib else (ss - cs)

    notes = str(result.get("notes", ""))
    abl = result.get("ablated_method")
    if abl and abl in scores and isinstance(scores[abl], (int, float)) \
            and scores[abl] == scores[abl]:
        a = float(scores[abl])
        mech = (cs - a) if hib else (a - cs)  # >0: the mechanism itself helps
        notes += (f" | ablation: candidate={cs:.4g} ablated={a:.4g} "
                  f"mechanism_delta={mech:+.4g}")
    return Outcome(
        run_id=run_id, cid=cid,
        metric=str(result.get("metric", "")),
        higher_is_better=hib, scores=scores,
        composition_method=comp_m, home_field_sota_method=sota_m,
        composition_score=cs, home_field_score=ss,
        delta_vs_home=delta if delta == delta else 0.0,
        min_effect=parse_min_effect(recipe.get("min_meaningful_effect_size")),
        beats_home_field=(g == "transfer-win"),
        grade=g, n_trials=int(result.get("n_trials", 0)),
        transfer_distance=transfer_distance_of(comp, run_id),
        valid=valid, invalid_reason=reason,
        provenance="auto", confidence=confidence_from_audit(valid, audit),
        audit=audit or {}, timestamp=now.isoformat(),
        notes=notes,
    )


def record(outcome: Outcome) -> None:
    """Append one outcome to the ledger under an exclusive advisory lock.

    Concurrent flywheel runs append to the same file; without the lock,
    interleaved writes could corrupt the JSONL that calibration reads.
    """
    OUTCOMES.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(asdict(outcome)) + "\n"
    with OUTCOMES.open("a") as f, _exclusive_lock(f):
        f.write(line)
        f.flush()


# --------------------------------------------------------------------------- #
# Candidate codegen — the ONE guarded step
# --------------------------------------------------------------------------- #
CANDIDATE_PROMPT = """You are implementing the CANDIDATE method for an existing benchmark. The harness,
datasets, baselines, validator, and the home-field SOTA are human-written and
already exist — you must NOT reimplement, modify, or work around any of them.
Your entire job is ONE Python module implementing the composition's idea behind
the declared interface.

PROBLEM: {problem}

COMPOSITION (the idea to implement):
{claim}

CONSTITUENT PAPERS (roles in the composition):
{papers}

RECIPE (context only — the harness already enforces the comparison):
  home_field_sota: {home_field_sota}
  metric_primary: {metric}

PROBLEM SHAPE (from the domain pack):
  objective: {objective}
  constraints:
{constraints}

DOMAIN GOTCHAS:
{gotchas}

CANDIDATE INTERFACE (your module MUST satisfy exactly this):
{candidate_interface}

HARD RULES:
  1. Emit ONE self-contained Python module satisfying the interface above.
     Emit ONLY the Python file contents, no markdown fences.
  2. Importing the domain's existing baseline implementations to build on them
     is allowed and encouraged where the composition calls for it.
  3. No network access, no file writes, no subprocesses, nothing printed to
     stdout, no reading of harness internals beyond the declared interface.
  4. Determinism: respect the seed parameter; no time- or os-based randomness.
  5. Your output is re-validated by the harness; invalid output fails the run.
"""


def _format_papers(comp: dict) -> str:
    lines = []
    for p in comp.get("papers", []):
        pid = p.get("paper_id") or p.get("id") or "?"
        role = p.get("role", "")
        quote = (p.get("quote") or "").strip()
        line = f"- {pid}: {role}"
        if quote:
            line += f' — "{quote[:200]}"'
        lines.append(line)
    return "\n".join(lines) or "- (none listed)"


def generate_candidate(comp: dict, recipe: dict, pack, problem: str,
                       dest: Path, cid: str = "C") -> Path:
    """Generate the candidate solver module via codex into `dest`.

    The only codegen in autoverify: the prompt confines codex to the pack's
    candidate interface; the harness, baselines, and SOTA are never generated.
    """
    from pipeline.llm_io import run_codex_with_artifacts, update_artifact_meta
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
        candidate_interface=(
            pack.evaluator.candidate_interface
            + (f"\n\nExact required signature: "
               f"{pack.evaluator.candidate_signature.render()}"
               if pack.evaluator.candidate_signature else "")
        ),
    )
    ok, stdout, _stderr, art = run_codex_with_artifacts(
        stage=f"autoverify_candidate_{cid}", prompt=prompt,
        run_dir=dest.parent, reasoning="high")
    code = stdout.strip()
    if code.startswith("```"):
        code = re.sub(r"^```[a-z]*\n|\n```$", "", code)
    dest.write_text(code)
    update_artifact_meta(art, parse_status="ok" if ok else "codex_failed")
    return dest


# --------------------------------------------------------------------------- #
# Pack-bench execution
# --------------------------------------------------------------------------- #
def execute_pack_bench(command: str, timeout_s: int | None = None) -> dict:
    """Run a pack's candidate_command; return its parsed result dict or an error.

    Wall-clock cap defaults to 900s; override with CONGRUITY_BENCH_TIMEOUT_S
    (an over-engineered candidate that can't finish the pack's evaluation in
    the budget is a legitimate invalid, and a shorter cap resolves it faster).

    The result is parsed from the LAST stdout line, which the pack bench must
    emit as RESULT_CONTRACT JSON (its --json mode). The command runs in its
    own process group and the WHOLE group is killed on timeout — wrapper
    commands like `uv run` otherwise leave the real bench python orphaned
    and burning CPU (observed: a 5400s-killed heilbronn confirm left its
    grandchild alive for 6+ hours).

    Cross-platform: on POSIX the command is tokenized with shlex and run in a
    new session (killed via killpg); on Windows shlex is skipped (its POSIX
    escaping mangles ``C:\\`` paths) — the string is handed to CreateProcess
    in a new process group and the tree is killed via ``taskkill /T``.
    """
    import os
    import signal

    if timeout_s is None:
        timeout_s = int(os.environ.get("CONGRUITY_BENCH_TIMEOUT_S", "900"))
    if os.name == "nt":
        args: str | list[str] = command  # CreateProcess parses the string
        popen_kwargs = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    else:
        args = shlex.split(command)
        popen_kwargs = {"start_new_session": True}
    try:
        proc = subprocess.Popen(
            args, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, cwd=str(REPO), **popen_kwargs)
    except FileNotFoundError as e:
        return {"_error": f"command not found: {e}"}
    try:
        stdout, stderr = proc.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                capture_output=True)
        else:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
        proc.wait()
        return {"_error": "timeout"}
    if proc.returncode != 0:
        return {"_error": f"exit {proc.returncode}: {(stderr or '')[-300:]}"}
    last = stdout.strip().splitlines()[-1] if stdout.strip() else ""
    try:
        return json.loads(last)
    except json.JSONDecodeError:
        return {"_error": f"unparseable result: {last[:200]}"}


def apply_record_claims(outcome: Outcome, result: dict, pack) -> None:
    """Re-grade an outcome against the pack's cached record table (Tier B).

    Benches may emit ``record_claims``: [{"params": {...}, "value": float}].
    Beating a best_known record upgrades the grade to "new-record" — a
    certifiable novel discovery (still confidence<=medium until a human
    checks the certificate against the cited source). "Beating" a
    proven_optimal record is impossible, so it invalidates and quarantines
    the outcome: the objective conventions don't match.
    """
    claims = result.get("record_claims") or []
    if not claims or getattr(pack, "records", None) is None:
        return
    from pipeline.computable import compare_to_record
    notes: list[str] = []
    for claim in claims:
        try:
            value = float(claim.get("value"))
        except (TypeError, ValueError):
            continue
        cmp = compare_to_record(pack, claim.get("params") or {}, value)
        label = cmp.get("record_id") or str(claim.get("params"))
        notes.append(f"record[{label}]: {cmp['verdict']}"
                     f" (margin {cmp.get('margin', 0.0):+.3g})")
        if cmp["verdict"] == "exceeds-proven-optimal-CHECK-OBJECTIVE":
            outcome.valid = False
            outcome.grade = "invalid"
            outcome.invalid_reason = (
                "record claim exceeds a proven-optimal value — objective "
                "conventions mismatch, not a discovery")
            outcome.confidence = "quarantined"
        elif (cmp["verdict"] == "new-record" and outcome.valid
              and outcome.grade != "invalid"):
            outcome.grade = "new-record"
            outcome.beats_home_field = True
    if notes:
        outcome.notes += " | " + "; ".join(notes)


def structural_check(result: dict, pack) -> str:
    """Pack-contract check on a bench result; '' if clean, else the violation.

    Replaces the old LLM fairness audit: with a human-written harness the only
    remaining failure surface is the bench emitting a result that contradicts
    the pack's declared evaluator (grade() handles score-level validity).
    """
    metric = result.get("metric", "")
    if metric != pack.evaluator.metric_primary:
        return (f"metric {metric!r} != pack metric "
                f"{pack.evaluator.metric_primary!r}")
    if bool(result.get("higher_is_better", True)) != pack.evaluator.higher_is_better:
        return "higher_is_better disagrees with pack evaluator"
    return ""


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #
def orchestrate(run_id: str, cid: str, now: datetime, *, pack_id: str,
                candidate_path: str | None = None,
                reuse_candidate: bool = False) -> dict:
    """Pack-bench verification: candidate -> pack bench -> grade -> record.

    The pack provides the trusted harness; the candidate is either a
    human-written module (`candidate_path`) or one contained codex generation.
    Any crash, contract violation, or invalid output records a quarantined
    invalid outcome — there is no repair loop.
    """
    comp = load_composition(run_id, cid)
    if comp is None:
        return {"error": f"composition {run_id}/{cid} not found"}
    recipe = (comp.get("verification") or {}).get("recipe") or {}
    if not recipe:
        return {"error": "no recipe on composition (run compose --with-recipes first)"}
    problem = json.loads((COMPOSITIONS / run_id / "state.json").read_text()).get("problem", "")

    from pipeline.computable import load_pack, preflight_pack
    try:
        pack = load_pack(pack_id)
    except (FileNotFoundError, ValueError) as e:
        return {"error": str(e)}
    pre = preflight_pack(pack_id)
    if not pre.ai_verifiable:
        return {"error": f"pack {pack_id!r} is not ai_verifiable; missing: {pre.missing}"}
    if not pre.candidate_ready:
        return {"error": f"pack {pack_id!r} declares no candidate interface "
                         "(evaluator.candidate_command / candidate_interface)"}

    if candidate_path:
        cand = Path(candidate_path)
        if not cand.exists():
            return {"error": f"candidate module not found: {cand}"}
    else:
        out_dir = AUTO_EXPERIMENTS / f"{run_id}-{cid}"
        out_dir.mkdir(parents=True, exist_ok=True)
        cand = out_dir / "candidate.py"
        if not (reuse_candidate and cand.exists()):
            cand = generate_candidate(comp, recipe, pack, problem, cand, cid=cid)

    audit = {
        "fair": True, "sota_faithful": True, "issues": [],
        "notes": f"pack-bench mode: harness, baselines, and home-field SOTA "
                 f"from computable pack '{pack_id}' (human-written)",
    }

    command = pack.evaluator.candidate_command.format(candidate=str(cand))
    result = execute_pack_bench(command)

    invalid_reason = ""
    if "_error" in result:
        invalid_reason = result["_error"]
    else:
        violation = structural_check(result, pack)
        if violation:
            invalid_reason = f"pack contract violation: {violation}"

    if invalid_reason:
        outcome = build_outcome(run_id, cid, comp, recipe,
                                {"scores": {}, "n_trials": 0}, now, audit=audit)
        outcome.valid, outcome.invalid_reason, outcome.grade = False, invalid_reason, "invalid"
        outcome.confidence = "quarantined"
        record(outcome)
        return {"grade": "invalid", "reason": invalid_reason,
                "pack": pack_id, "candidate": str(cand), "command": command}

    outcome = build_outcome(run_id, cid, comp, recipe, result, now, audit=audit)
    apply_record_claims(outcome, result, pack)
    record(outcome)
    return {
        "grade": outcome.grade, "valid": outcome.valid,
        "confidence": outcome.confidence,
        "beats_home_field": outcome.beats_home_field,
        "composition": outcome.composition_score,
        "home_field_sota": outcome.home_field_score,
        "delta_vs_home": round(outcome.delta_vs_home, 4),
        "transfer_distance": outcome.transfer_distance,
        "pack": pack_id, "candidate": str(cand),
    }
