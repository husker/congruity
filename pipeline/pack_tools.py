"""Computable-pack tooling: `pack doctor` and `pack new`.

Preflight (pipeline/computable.py) only checks that declared fields are
non-empty — it never runs anything, so a pack with a broken smoke command or
a bench that doesn't exist passes preflight and then wastes a full compose
run before failing. `doctor` closes that gap:

  - structural preflight (existing checks)
  - static command checks: `{candidate}` placeholders present where
    required, and any `python -m experiments.x.bench` module referenced by
    a command actually exists on disk
  - live smoke run (opt-out with --no-smoke): executes evaluator.smoke_command
    in its own process group with a timeout, requiring exit 0
  - records.json sanity: unique ids, params present

`scaffold` (pack new) writes a template pack directory plus a bench skeleton
built on experiments/_harness.py, so a new domain starts contract-compliant
instead of hand-copying an old bench.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import signal
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from pipeline.computable import PACKS_DIR, load_pack, preflight_pack

REPO = Path(__file__).resolve().parent.parent
SMOKE_TIMEOUT_S = 300


@dataclass
class DoctorReport:
    pack_id: str
    ok: bool
    preflight: dict = field(default_factory=dict)
    checks: list[tuple[str, bool, str]] = field(default_factory=list)

    def add(self, name: str, passed: bool, detail: str = "") -> None:
        self.checks.append((name, passed, detail))
        if not passed:
            self.ok = False

    def render(self) -> str:
        lines = [f"pack doctor: {self.pack_id} — {'OK' if self.ok else 'PROBLEMS FOUND'}"]
        for name, passed, detail in self.checks:
            mark = "ok " if passed else "FAIL"
            lines.append(f"  [{mark}] {name}" + (f" — {detail}" if detail else ""))
        return "\n".join(lines)


_BENCH_MODULE_PAT = re.compile(r"-m\s+(experiments(?:\.\w+)+)")


def _referenced_bench_modules(commands: list[str]) -> set[str]:
    mods: set[str] = set()
    for cmd in commands:
        for m in _BENCH_MODULE_PAT.finditer(cmd or ""):
            mods.add(m.group(1))
    return mods


def _run_smoke(command: str, timeout_s: int) -> tuple[bool, str]:
    """Run a smoke command in its own process group; kill the group on
    timeout (same orphan-grandchild defense as autoverify.execute_pack_bench).

    Cross-platform the same way as autoverify.execute_pack_bench: POSIX
    shlex-tokenizes + new session + killpg; Windows keeps the string (shlex's
    POSIX escaping mangles ``C:\\`` paths) + new process group + taskkill /T.
    """
    if os.name == "nt":
        args: str | list[str] = command
        popen_kwargs = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    else:
        args = shlex.split(command)
        popen_kwargs = {"start_new_session": True}
    try:
        proc = subprocess.Popen(
            args, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, cwd=str(REPO), **popen_kwargs)
    except FileNotFoundError as e:
        return False, f"command not found: {e}"
    try:
        _stdout, stderr = proc.communicate(timeout=timeout_s)
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
        return False, f"timeout after {timeout_s}s"
    if proc.returncode != 0:
        return False, f"exit {proc.returncode}: {(stderr or '')[-200:]}"
    return True, ""


def doctor(pack_id: str, *, root: Path = PACKS_DIR, run_smoke: bool = True,
           smoke_timeout_s: int = SMOKE_TIMEOUT_S) -> DoctorReport:
    """Structural + executable health check for one pack."""
    report = DoctorReport(pack_id=pack_id, ok=True)

    pre = preflight_pack(pack_id, root=root)
    report.preflight = {
        "computable": pre.computable,
        "ai_verifiable": pre.ai_verifiable,
        "candidate_ready": pre.candidate_ready,
        "search_ready": pre.search_ready,
        "record_ready": pre.record_ready,
    }
    report.add("preflight: computable", pre.computable,
               "; ".join(pre.missing) if not pre.computable else "")
    if not pre.computable:
        return report  # nothing else is checkable

    pack = load_pack(pack_id, root=root)
    ev = pack.evaluator

    # Static placeholder checks (preflight covers these for candidate/search;
    # repeat here so the doctor output is one complete picture).
    for label, cmd, needs_placeholder in (
        ("smoke_command", ev.smoke_command, False),
        ("full_command", ev.full_command, False),
        ("candidate_command", ev.candidate_command, True),
        ("search_command", ev.search_command, True),
        ("confirm_command", ev.confirm_command, True),
    ):
        if not cmd:
            continue
        if needs_placeholder and "{candidate}" not in cmd:
            report.add(f"{label} has {{candidate}} placeholder", False, cmd)
        else:
            report.add(f"{label} declared", True)

    # Every bench module referenced by any command must exist on disk.
    commands = [ev.smoke_command, ev.full_command, ev.candidate_command,
                ev.search_command, ev.confirm_command]
    commands += [b.command for b in pack.baselines.baselines]
    for mod in sorted(_referenced_bench_modules(commands)):
        path = REPO / (mod.replace(".", "/") + ".py")
        report.add(f"bench module {mod}", path.exists(),
                   "" if path.exists() else f"missing file {path}")

    # Typed candidate signature (optional but recommended for candidate packs).
    if ev.candidate_command:
        has_sig = ev.candidate_signature is not None
        report.add("candidate_signature declared", True,
                   ev.candidate_signature.render() if has_sig
                   else "not declared (prose interface only) — consider adding "
                        "evaluator.candidate_signature")

    # records.json sanity.
    if pack.records is not None:
        ids = [r.id for r in pack.records.records]
        report.add("records: unique ids", len(ids) == len(set(ids)),
                   "" if len(ids) == len(set(ids)) else "duplicate record ids")
        report.add("records: params present",
                   all(r.params for r in pack.records.records))

    # Live smoke run — the check preflight can't do.
    if run_smoke:
        ok, detail = _run_smoke(ev.smoke_command, smoke_timeout_s)
        report.add("smoke run (exit 0)", ok, detail)
    else:
        report.add("smoke run", True, "skipped (--no-smoke)")

    return report


# --------------------------------------------------------------------------
# Scaffolder
# --------------------------------------------------------------------------

_PACK_TEMPLATE = {
    "id": "{pack_id}",
    "display_name": "{Pack Display Name}",
    "description": "One-paragraph description of the problem this pack verifies.",
    "problem_shape": {
        "objective": "minimize|maximize <metric> subject to <constraints>",
        "input_representation": "what a problem instance looks like",
        "output_representation": "what a solution looks like",
        "constraints": ["every constraint the validator enforces"],
        "failure_modes": ["how candidates typically cheat or break"],
        "verification_kind": "exact_oracle",
    },
    "recommended_problem_template": (
        "Develop a novel method for ... that beats <home-field SOTA> on ..."),
    "compose_query_hints": ["query one", "query two"],
}

_EVALUATOR_TEMPLATE = {
    "metric_primary": "<metric name>",
    "higher_is_better": False,
    "validator": "how solutions are re-checked (never trust the solver's claim)",
    "smoke_command": "uv run python -m experiments.{pack_id}.bench --smoke",
    "full_command": "uv run python -m experiments.{pack_id}.bench",
    "pass_criteria": "what counts as beating the home-field SOTA",
    # NB: these templates are rendered by string-replacing "{pack_id}" only;
    # "{candidate}" must survive verbatim for autoverify to substitute later.
    "candidate_command": ("uv run python -m experiments.{pack_id}.bench "
                          "--candidate {candidate} --json"),
    "candidate_interface": (
        "The candidate module must expose solve(instance, seed) -> solution. "
        "Describe types precisely; this text is injected into codegen prompts."),
    "candidate_signature": {
        "function": "solve",
        "args": ["instance: <type>", "seed: int"],
        "returns": "<type>",
    },
    "search_command": ("uv run python -m experiments.{pack_id}.bench "
                       "--candidate {candidate} --split train --json"),
    "confirm_command": ("uv run python -m experiments.{pack_id}.bench "
                        "--candidate {candidate} --split holdout --json"),
}

_BASELINES_TEMPLATE = {
    "baselines": [
        {"id": "naive", "name": "Naive baseline", "level": "naive",
         "command": "uv run python -m experiments.{pack_id}.bench --methods naive",
         "notes": ""},
        {"id": "sota", "name": "Home-field SOTA (human-verified)",
         "level": "strong_reproducible",
         "command": "uv run python -m experiments.{pack_id}.bench --methods sota",
         "notes": "Cite the paper/implementation this reproduces."},
    ]
}

_DATASETS_TEMPLATE = {
    "datasets": [
        {"id": "synthetic", "name": "Deterministic synthetic instances",
         "kind": "synthetic", "availability": "generated in-repo",
         "command": "", "notes": "Keep train and holdout instance sets disjoint."},
    ]
}

_SOTA_TEMPLATE = {
    "bar_levels": [
        {"level": "naive", "available": True,
         "comparator": "naive baseline in this pack", "source": "", "notes": ""},
        {"level": "strong_reproducible", "available": True,
         "comparator": "<named method>", "source": "<paper/URL>", "notes": ""},
        {"level": "record_cached", "available": False,
         "comparator": "published record table", "source": "",
         "notes": "Fill records.json ONLY with human-verified published values."},
    ]
}

_GOTCHAS_TEMPLATE = """# Gotchas — {pack_id}

Notes injected into candidate codegen prompts. List domain traps:
degenerate solutions the validator must reject, numerical pitfalls,
common ways candidates accidentally cheat.
"""

_BENCH_TEMPLATE = '''"""Bench for the {pack_id} pack. Emits RESULT_CONTRACT JSON with --json.

Skeleton generated by `pipeline pack new`. Fill in instances, baselines, and
the validator; keep train/holdout splits disjoint and never trust a
candidate's self-reported score.
"""

from __future__ import annotations

from experiments._harness import cap_threads

cap_threads()  # BEFORE numpy import — see experiments/_harness.py

import argparse  # noqa: E402
import sys  # noqa: E402

from experiments._harness import emit_result_contract, make_rng  # noqa: E402


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", default=None)
    parser.add_argument("--split", choices=["train", "holdout"], default="train")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    rng = make_rng(1234)
    _ = rng  # TODO: build instances, run methods, validate outputs

    scores = {{"candidate": float("nan"), "sota": float("nan")}}
    if args.json:
        emit_result_contract(
            metric="<metric>", higher_is_better=False, scores=scores,
            composition_method="candidate", home_field_sota_method="sota",
            n_trials=0, notes="TODO: implement")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
'''


def scaffold(pack_id: str, *, root: Path = PACKS_DIR,
             experiments_root: Path | None = None) -> list[Path]:
    """Write a template pack directory + bench skeleton. Refuses to overwrite."""
    if not re.fullmatch(r"[a-z][a-z0-9_]*", pack_id):
        raise ValueError("pack id must be lowercase snake_case")
    pack_dir = root / pack_id
    if pack_dir.exists():
        raise FileExistsError(f"{pack_dir} already exists")
    exp_root = experiments_root or (REPO / "experiments")
    bench_dir = exp_root / pack_id
    if bench_dir.exists():
        raise FileExistsError(f"{bench_dir} already exists")

    def _fmt(obj):
        return json.loads(json.dumps(obj).replace("{pack_id}", pack_id))

    pack_dir.mkdir(parents=True)
    written = []
    for name, template in (("pack.json", _PACK_TEMPLATE),
                           ("evaluator.json", _EVALUATOR_TEMPLATE),
                           ("baselines.json", _BASELINES_TEMPLATE),
                           ("datasets.json", _DATASETS_TEMPLATE),
                           ("sota.json", _SOTA_TEMPLATE)):
        p = pack_dir / name
        p.write_text(json.dumps(_fmt(template), indent=2) + "\n")
        written.append(p)
    g = pack_dir / "gotchas.md"
    g.write_text(_GOTCHAS_TEMPLATE.format(pack_id=pack_id))
    written.append(g)

    bench_dir.mkdir(parents=True)
    b = bench_dir / "bench.py"
    b.write_text(_BENCH_TEMPLATE.format(pack_id=pack_id))
    written.append(b)
    return written
