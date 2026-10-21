"""`pipeline pack forge` — evidence-first automation of the measuring stick.

The May 30 flywheel verdict established that codex *re-implementing* a
domain's SOTA produces strawmen. Forge automates pack creation anyway, by
moving trust from generation to things a machine can CHECK:

  draft      codex drafts the pack files, a bench module, validator attack
             cases, and a SOTA provenance claim — every claim carries a
             source (paper, repo, published number).
  pin        the SOTA's official implementation is cloned and pinned to an
             exact commit SHA. Retrieval, not re-implementation.
  reproduce  the acceptance test generation can't fake: run the pinned
             implementation's reproduce command and require it to match the
             PUBLISHED number within the declared tolerance.
  battery    attack the drafted validator: every case's good_solution must
             pass, bad_solution must fail, and declared values must
             recompute. A validator that survives is trustworthy without
             trusting the model that wrote it.
  records    fetch the declared record-table sources and snapshot them with
             provenance. Extraction into records.json stays human — a
             fabricated "best known" fails silently and poisons the ledger.
  evidence   assemble everything into forge.json + evidence.md.

Nothing forge produces can touch the ledger: packs live in
data/pipeline/forge/<id>/ until `pipeline pack promote <id> --approve <name>`
— the human signature — copies them into computable_packs/ after a final
doctor gate. autoverify/autosearch only ever load from computable_packs/,
so an unpromoted pack cannot label an outcome.

Every stage persists its result in forge.json; re-running skips stages that
already passed (delete the workspace or use --redo to re-run).
"""

from __future__ import annotations

import importlib.util
import json
import os
import shlex
import shutil
import signal
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import requests
from pydantic import BaseModel, ConfigDict, Field

from pipeline.computable import PACKS_DIR, load_pack, preflight_pack
from pipeline.llm_io import parse_or_repair, run_codex_with_artifacts

REPO = Path(__file__).resolve().parent.parent
FORGE_DIR = REPO / "data" / "pipeline" / "forge"
EXPERIMENTS_DIR = REPO / "experiments"

REPRODUCE_TIMEOUT_S = 3600
FETCH_THROTTLE_S = 1.1

STAGES = ("draft", "pin", "reproduce", "battery", "records", "evidence")


# --------------------------------------------------------------------------
# Draft schema — what the codex draft call must return
# --------------------------------------------------------------------------


class PublishedResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    benchmark: str = Field(min_length=1)     # the paper's own benchmark
    metric: str = Field(min_length=1)
    value: float
    rel_tolerance: float = Field(gt=0, le=0.5, default=0.05)
    source: str = Field(min_length=1)        # paper + table/figure citation


class SotaProvenance(BaseModel):
    model_config = ConfigDict(extra="forbid")
    method_name: str = Field(min_length=1)
    paper_title: str = Field(min_length=1)
    paper_url: str = Field(min_length=1)
    repo_url: str = Field(min_length=1)      # official/reference implementation
    repo_ref: str = ""                       # optional branch/tag hint
    # Runs inside the pinned clone; LAST stdout line must be {"value": float}.
    reproduce_command: str = Field(min_length=1)
    published_result: PublishedResult


class ValidatorCase(BaseModel):
    """One attack case for the drafted validator."""

    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=1)
    instance: Any
    good_solution: Any                       # validator must accept
    bad_solution: Any                        # validator must reject
    good_value: float | None = None          # optional recomputation check


class ForgeDraft(BaseModel):
    model_config = ConfigDict(extra="allow")
    pack: dict
    evaluator: dict
    baselines: dict
    datasets: dict
    sota: dict
    gotchas_md: str = ""
    # Full bench module source. Must expose
    # validate(instance, solution) -> (bool, float) and implement the
    # RESULT_CONTRACT --json mode described in the prompt.
    bench_py: str = Field(min_length=1)
    sota_provenance: SotaProvenance
    validator_cases: list[ValidatorCase] = Field(min_length=2)
    record_sources: list[str] = []


FORGE_PROMPT = """You are drafting a "computable pack" — an executable verification
harness — for the congruity research-verification system.

DOMAIN / PROBLEM:
{domain}

A pack makes one problem domain automatically verifiable. It consists of five
JSON files (schemas below), a gotchas note, and a bench module. The single
most important rule: every factual claim (SOTA method, published numbers,
record sources) must carry a real, checkable source. A later stage will clone
the repo you name and REQUIRE its reproduce command to match the published
number you cite — so cite only what genuinely reproduces. If you are not
confident a claim is real, say so in notes rather than inventing one.

Return ONLY a JSON object (no prose, no fences) with these keys:

- "pack": {{id, display_name, description, problem_shape: {{objective,
  input_representation, output_representation, constraints[],
  failure_modes[], verification_kind (one of: exact_oracle, roundtrip,
  benchmark, backtest, record_comparison, property_test)}},
  recommended_problem_template, compose_query_hints[]}}
  The id MUST be "{pack_id}".
- "evaluator": {{metric_primary, higher_is_better, validator, smoke_command,
  full_command, pass_criteria, candidate_command, candidate_interface,
  candidate_signature: {{function, args[], returns}}, search_command,
  confirm_command}}. Commands use
  "uv run python -m experiments.{pack_id}.bench ..." and candidate/search/
  confirm commands must contain the literal placeholder {{candidate}}.
  Keep train and holdout splits disjoint.
- "baselines": {{baselines: [{{id, name, level (naive|strong_reproducible|
  record_cached), command, notes}}]}}
- "datasets": {{datasets: [{{id, name, kind, availability, command, notes}}]}}
- "sota": {{bar_levels: [{{level, available, comparator, source, notes}}]}}
- "gotchas_md": domain traps injected into candidate codegen prompts.
- "bench_py": the COMPLETE bench module source for
  experiments/{pack_id}/bench.py. Requirements:
    * import and call cap_threads() from experiments._harness BEFORE numpy
    * expose validate(instance, solution) -> (bool, float) — re-derives the
      metric from the solution; never trusts a solver's claim
    * --smoke, --json, --candidate PATH, --split train|holdout flags
    * with --json, print RESULT_CONTRACT JSON as the LAST stdout line via
      experiments._harness.emit_result_contract (keys: metric,
      higher_is_better, scores, composition_method, home_field_sota_method,
      n_trials, notes)
- "sota_provenance": {{method_name, paper_title, paper_url, repo_url,
  repo_ref, reproduce_command, published_result: {{benchmark, metric, value,
  rel_tolerance, source}}}}. reproduce_command runs inside a fresh clone of
  repo_url and must print {{"value": <float>}} as its LAST stdout line.
- "validator_cases": >= 2 attack cases: {{name, instance, good_solution,
  bad_solution, good_value (optional)}} — good must pass validate(), bad
  must fail, good_value (when given) must match the recomputed metric.
  All values must be plain JSON.
- "record_sources": URLs of authoritative record tables for this domain
  (may be empty). These are snapshotted for HUMAN extraction — do not
  fabricate record values anywhere.
"""


# --------------------------------------------------------------------------
# Workspace / state
# --------------------------------------------------------------------------


def workspace_for(pack_id: str, root: Path = FORGE_DIR) -> Path:
    return root / pack_id


def _state_path(ws: Path) -> Path:
    return ws / "forge.json"


def load_state(ws: Path) -> dict:
    p = _state_path(ws)
    if p.exists():
        try:
            return json.loads(p.read_text())
        except json.JSONDecodeError:
            pass
    return {"stages": {}}


def _save_state(ws: Path, state: dict) -> None:
    ws.mkdir(parents=True, exist_ok=True)
    _state_path(ws).write_text(json.dumps(state, indent=2))


def _mark(state: dict, stage: str, ok: bool, **detail) -> None:
    state["stages"][stage] = {
        "ok": ok,
        "at": datetime.now(timezone.utc).isoformat(),
        **detail,
    }


# --------------------------------------------------------------------------
# Stages
# --------------------------------------------------------------------------


def stage_draft(pack_id: str, domain: str, ws: Path, state: dict) -> bool:
    """Codex drafts the pack + bench + provenance; everything is persisted
    and schema-validated (JSON schema here, strict pack schemas on load)."""
    prompt = FORGE_PROMPT.format(domain=domain, pack_id=pack_id,
                                 candidate="{candidate}")
    ok, stdout, _stderr, _art = run_codex_with_artifacts(
        stage="forge_draft", prompt=prompt, run_dir=ws, reasoning="high")
    if not ok:
        _mark(state, "draft", False, error="codex call failed")
        return False

    def repair_call(repair_prompt: str):
        return run_codex_with_artifacts(
            stage="forge_draft_repair", prompt=repair_prompt,
            run_dir=ws, reasoning="high")

    parsed = parse_or_repair(stdout, ForgeDraft, repair_call)
    if parsed.value is None:
        _mark(state, "draft", False, error="; ".join(parsed.errors[:3]))
        return False
    draft = parsed.value

    pack_dir = ws / "packs" / pack_id
    pack_dir.mkdir(parents=True, exist_ok=True)
    for name, obj in (("pack.json", draft.pack),
                      ("evaluator.json", draft.evaluator),
                      ("baselines.json", draft.baselines),
                      ("datasets.json", draft.datasets),
                      ("sota.json", draft.sota)):
        (pack_dir / name).write_text(json.dumps(obj, indent=2) + "\n")
    (pack_dir / "gotchas.md").write_text(draft.gotchas_md or f"# Gotchas — {pack_id}\n")
    (ws / "bench.py").write_text(draft.bench_py)
    (ws / "validator_cases.json").write_text(json.dumps(
        [c.model_dump() for c in draft.validator_cases], indent=2))
    (ws / "sota_provenance.json").write_text(
        draft.sota_provenance.model_dump_json(indent=2))
    (ws / "record_sources.json").write_text(json.dumps(draft.record_sources, indent=2))

    # The drafted files must survive the strict pack schemas as-is.
    try:
        load_pack(pack_id, root=ws / "packs")
    except (ValueError, FileNotFoundError) as e:
        _mark(state, "draft", False, error=f"pack schema validation: {e}")
        return False
    pre = preflight_pack(pack_id, root=ws / "packs")
    _mark(state, "draft", True,
          candidate_ready=pre.candidate_ready, search_ready=pre.search_ready,
          validator_cases=len(draft.validator_cases),
          record_sources=len(draft.record_sources))
    return True


def stage_pin(ws: Path, state: dict) -> bool:
    """Clone the declared SOTA implementation and pin the exact commit SHA."""
    prov = SotaProvenance.model_validate_json(
        (ws / "sota_provenance.json").read_text())
    dest = ws / "sota_impl"
    if dest.exists():
        shutil.rmtree(dest)
    cmd = ["git", "clone", "--depth", "1"]
    if prov.repo_ref:
        cmd += ["--branch", prov.repo_ref]
    cmd += [prov.repo_url, str(dest)]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        _mark(state, "pin", False, error=result.stderr[-300:],
              repo_url=prov.repo_url)
        return False
    sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=dest,
                         capture_output=True, text=True).stdout.strip()
    _mark(state, "pin", True, repo_url=prov.repo_url, sha=sha)
    return True


def _run_in_group(command: str, cwd: Path, timeout_s: int) -> tuple[int, str, str]:
    """Run in its own process group; SIGKILL the group on timeout."""
    proc = subprocess.Popen(shlex.split(command), stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True, cwd=str(cwd),
                            start_new_session=True)
    try:
        stdout, stderr = proc.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        proc.wait()
        return -1, "", f"timeout after {timeout_s}s"
    return proc.returncode, stdout, stderr


def stage_reproduce(ws: Path, state: dict,
                    timeout_s: int = REPRODUCE_TIMEOUT_S) -> bool:
    """The acceptance test: the pinned implementation must reproduce its own
    published number within the declared tolerance. This is the check that
    distinguishes a faithful SOTA from a strawman."""
    prov = SotaProvenance.model_validate_json(
        (ws / "sota_provenance.json").read_text())
    impl = ws / "sota_impl"
    if not impl.exists():
        _mark(state, "reproduce", False, error="no pinned implementation (run pin)")
        return False
    code, stdout, stderr = _run_in_group(prov.reproduce_command, impl, timeout_s)
    if code != 0:
        _mark(state, "reproduce", False,
              error=f"exit {code}: {(stderr or '')[-300:]}")
        return False
    last = stdout.strip().splitlines()[-1] if stdout.strip() else ""
    try:
        got = float(json.loads(last)["value"])
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        _mark(state, "reproduce", False,
              error=f"last stdout line is not {{\"value\": float}}: {last[:120]}")
        return False
    expected = prov.published_result.value
    rel_err = abs(got - expected) / max(abs(expected), 1e-12)
    ok = rel_err <= prov.published_result.rel_tolerance
    _mark(state, "reproduce", ok, expected=expected, got=got,
          rel_err=round(rel_err, 6),
          rel_tolerance=prov.published_result.rel_tolerance,
          source=prov.published_result.source)
    return ok


def _load_bench_module(bench_path: Path):
    spec = importlib.util.spec_from_file_location("forge_bench", bench_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load bench module from {bench_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def stage_battery(ws: Path, state: dict) -> bool:
    """Attack the drafted validator with the drafted cases.

    good_solution must pass, bad_solution must fail, and any declared
    good_value must match the recomputed metric. A validator that survives
    is trustworthy independent of the model that wrote it.
    """
    try:
        mod = _load_bench_module(ws / "bench.py")
    except Exception as e:  # drafted code is untrusted
        _mark(state, "battery", False, error=f"bench import failed: {e}")
        return False
    validate = getattr(mod, "validate", None)
    if not callable(validate):
        _mark(state, "battery", False,
              error="bench.py does not expose validate(instance, solution)")
        return False

    cases = json.loads((ws / "validator_cases.json").read_text())
    results, all_ok = [], True
    for case in cases:
        name = case.get("name", "?")
        entry = {"name": name}
        try:
            good_ok, good_val = validate(case["instance"], case["good_solution"])
            bad_ok, _ = validate(case["instance"], case["bad_solution"])
            entry["good_accepted"] = bool(good_ok)
            entry["bad_rejected"] = not bad_ok
            entry["ok"] = bool(good_ok) and not bad_ok
            expected = case.get("good_value")
            if expected is not None:
                match = abs(float(good_val) - float(expected)) <= 1e-9 + 1e-6 * abs(float(expected))
                entry["value_matches"] = match
                entry["ok"] = entry["ok"] and match
        except Exception as e:  # validator crash on an attack case = fail
            entry["ok"] = False
            entry["error"] = str(e)[:200]
        results.append(entry)
        all_ok = all_ok and entry["ok"]
    _mark(state, "battery", all_ok, cases=results)
    return all_ok


def stage_records(ws: Path, state: dict,
                  fetch: Callable[[str], str] | None = None) -> bool:
    """Snapshot the declared record-table sources with provenance.

    Extraction into records.json is deliberately NOT automated: record
    values feed grading, and a fabricated best-known fails silently. The
    snapshots + source URLs make the human curation step a transcription
    job instead of a research job.
    """
    sources = json.loads((ws / "record_sources.json").read_text())
    snap_dir = ws / "records"
    snap_dir.mkdir(exist_ok=True)
    fetched, failed = [], []
    for i, url in enumerate(sources):
        if fetch is None:
            time.sleep(FETCH_THROTTLE_S)
        try:
            body = fetch(url) if fetch else _default_fetch(url)
            dest = snap_dir / f"source_{i:02d}.html"
            dest.write_text(body)
            (snap_dir / f"source_{i:02d}.meta.json").write_text(json.dumps({
                "url": url,
                "fetched_at": datetime.now(timezone.utc).isoformat(),
                "bytes": len(body),
            }, indent=2))
            fetched.append(url)
        except Exception as e:
            failed.append({"url": url, "error": str(e)[:200]})
    # No sources is fine (not every domain has record tables); a declared
    # source that failed to fetch is a problem worth flagging.
    ok = not failed
    _mark(state, "records", ok, fetched=fetched, failed=failed)
    return ok


def _default_fetch(url: str) -> str:
    resp = requests.get(url, timeout=60,
                        headers={"User-Agent": "congruity-pack-forge/1.0"})
    resp.raise_for_status()
    return resp.text


def stage_evidence(pack_id: str, ws: Path, state: dict) -> bool:
    """Render the evidence bundle a human reviews at promotion time."""
    lines = [f"# Forge evidence — {pack_id}", ""]
    prov_path = ws / "sota_provenance.json"
    if prov_path.exists():
        prov = json.loads(prov_path.read_text())
        pub = prov.get("published_result", {})
        lines += [
            "## SOTA provenance",
            f"- method: **{prov.get('method_name')}** — [{prov.get('paper_title')}]({prov.get('paper_url')})",
            f"- implementation: {prov.get('repo_url')} @ `{state['stages'].get('pin', {}).get('sha', 'UNPINNED')}`",
            f"- published: {pub.get('metric')} = {pub.get('value')} on {pub.get('benchmark')} ({pub.get('source')})",
            "",
        ]
    lines += ["## Stage results", ""]
    for stage in STAGES:
        info = state["stages"].get(stage)
        if not info:
            lines.append(f"- {stage}: NOT RUN")
            continue
        mark = "PASS" if info.get("ok") else "FAIL"
        detail = {k: v for k, v in info.items() if k not in ("ok", "at", "cases")}
        lines.append(f"- {stage}: **{mark}** — {json.dumps(detail, default=str)[:300]}")
    battery = state["stages"].get("battery", {})
    if battery.get("cases"):
        lines += ["", "## Validator battery", ""]
        for c in battery["cases"]:
            lines.append(f"- {'PASS' if c.get('ok') else 'FAIL'} {c.get('name')}"
                         + (f" — {c.get('error')}" if c.get("error") else ""))
    lines += [
        "",
        "## Reviewer checklist (before `pack promote`)",
        "",
        "- [ ] The cited paper/repo is the real home-field SOTA for this domain",
        "- [ ] The reproduce command measures the same thing as the published number",
        "- [ ] The validator cases cover the domain's known cheating modes",
        "- [ ] Record snapshots (if any) transcribed into records.json BY HAND with sources",
        "",
    ]
    (ws / "evidence.md").write_text("\n".join(lines))
    _mark(state, "evidence", True, path=str(ws / "evidence.md"))
    return True


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------


def forge(pack_id: str, domain: str, *, root: Path = FORGE_DIR,
          redo: list[str] | None = None,
          reproduce_timeout_s: int = REPRODUCE_TIMEOUT_S,
          fetch: Callable[[str], str] | None = None) -> dict:
    """Run all forge stages, skipping ones that already passed."""
    ws = workspace_for(pack_id, root)
    ws.mkdir(parents=True, exist_ok=True)
    state = load_state(ws)
    state["pack_id"] = pack_id
    state["domain"] = domain
    redo = redo or []

    def _need(stage: str) -> bool:
        return stage in redo or not state["stages"].get(stage, {}).get("ok")

    runners: list[tuple[str, Callable[[], bool]]] = [
        ("draft", lambda: stage_draft(pack_id, domain, ws, state)),
        ("pin", lambda: stage_pin(ws, state)),
        ("reproduce", lambda: stage_reproduce(ws, state, reproduce_timeout_s)),
        ("battery", lambda: stage_battery(ws, state)),
        ("records", lambda: stage_records(ws, state, fetch=fetch)),
        ("evidence", lambda: stage_evidence(pack_id, ws, state)),
    ]
    for stage, run in runners:
        if not _need(stage):
            print(f"forge {pack_id}: {stage} — already passed, skipping")
            continue
        print(f"forge {pack_id}: running {stage} ...", flush=True)
        try:
            ok = run()
        except Exception as e:  # a stage crash is a stage failure, not a forge crash
            _mark(state, stage, False, error=f"{type(e).__name__}: {e}")
            ok = False
        _save_state(ws, state)
        print(f"forge {pack_id}: {stage} — {'PASS' if ok else 'FAIL'}")
        # evidence renders whatever exists; earlier failures stop the chain
        # so the bundle reflects a definite state.
        if not ok and stage != "evidence":
            stage_evidence(pack_id, ws, state)
            _save_state(ws, state)
            break
    _save_state(ws, state)
    return state


REQUIRED_FOR_PROMOTION = ("draft", "pin", "reproduce", "battery")


def promote(pack_id: str, approve: str, *, root: Path = FORGE_DIR,
            packs_root: Path = PACKS_DIR,
            experiments_root: Path = EXPERIMENTS_DIR,
            waive: list[str] | None = None,
            run_doctor: bool = True) -> dict:
    """The human signature: copy a forged pack into computable_packs/.

    Refuses unless every required stage passed (or was explicitly waived
    with a recorded reason-holder name). Runs `pack doctor` on the promoted
    files as the final gate and rolls the copy back if it fails.
    """
    if not approve.strip():
        raise ValueError("promotion requires --approve <your name>")
    ws = workspace_for(pack_id, root)
    state = load_state(ws)
    waive = waive or []
    missing = [s for s in REQUIRED_FOR_PROMOTION
               if not state["stages"].get(s, {}).get("ok") and s not in waive]
    if missing:
        raise ValueError(
            f"cannot promote {pack_id}: stage(s) not passed: {missing}. "
            f"Fix and re-forge, or waive explicitly with --waive.")

    src_pack = ws / "packs" / pack_id
    dst_pack = packs_root / pack_id
    dst_bench_dir = experiments_root / pack_id
    if dst_pack.exists() or dst_bench_dir.exists():
        raise FileExistsError(f"{dst_pack} or {dst_bench_dir} already exists")

    shutil.copytree(src_pack, dst_pack)
    dst_bench_dir.mkdir(parents=True)
    shutil.copy2(ws / "bench.py", dst_bench_dir / "bench.py")

    if run_doctor:
        from pipeline.pack_tools import doctor
        report = doctor(pack_id, root=packs_root, run_smoke=True)
        if not report.ok:
            shutil.rmtree(dst_pack)
            shutil.rmtree(dst_bench_dir)
            raise RuntimeError(
                f"promotion doctor gate failed; rolled back.\n{report.render()}")

    provenance = {
        "forged": True,
        "approved_by": approve,
        "approved_at": datetime.now(timezone.utc).isoformat(),
        "waived_stages": waive,
        "evidence": str(ws / "evidence.md"),
        "sota_pin": state["stages"].get("pin", {}),
        "reproduce": state["stages"].get("reproduce", {}),
    }
    (dst_pack / "provenance.json").write_text(json.dumps(provenance, indent=2))
    state["promoted"] = provenance
    _save_state(ws, state)
    return provenance
