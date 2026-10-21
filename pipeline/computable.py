"""Computable problem packs.

A computable pack is an executable verification contract for one problem shape.
It tells Congruity which outputs are valid, which metric to use, which baselines
are available, and what success bars can be checked by code.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

PACKS_DIR = Path(__file__).resolve().parent.parent / "computable_packs"
BAR_LEVELS = ("naive", "strong_reproducible", "record_cached")
VERIFICATION_KINDS = (
    "exact_oracle",
    "roundtrip",
    "benchmark",
    "backtest",
    "record_comparison",
    "property_test",
)


class _StrictBase(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ProblemShape(_StrictBase):
    objective: str = Field(min_length=1)
    input_representation: str = Field(min_length=1)
    output_representation: str = Field(min_length=1)
    constraints: list[str] = Field(min_length=1)
    failure_modes: list[str] = Field(min_length=1)
    verification_kind: Literal[
        "exact_oracle",
        "roundtrip",
        "benchmark",
        "backtest",
        "record_comparison",
        "property_test",
    ]


class PackMeta(_StrictBase):
    id: str = Field(min_length=1)
    display_name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    problem_shape: ProblemShape
    recommended_problem_template: str = Field(min_length=1)
    compose_query_hints: list[str] = Field(min_length=1)


class CandidateSignature(_StrictBase):
    """Typed candidate entry point, checkable before the bench runs.

    Complements the prose ``candidate_interface`` (which is injected into
    codegen prompts): the signature lets tooling statically verify a
    generated candidate exposes the declared function instead of finding
    out at bench runtime.
    """

    function: str = Field(min_length=1)          # e.g. "solve"
    args: list[str] = Field(min_length=1)        # e.g. ["adj: list[set[int]]", "seed: int"]
    returns: str = Field(min_length=1)           # e.g. "list[int]"

    def render(self) -> str:
        return f"def {self.function}({', '.join(self.args)}) -> {self.returns}"


class EvaluatorSpec(_StrictBase):
    metric_primary: str = Field(min_length=1)
    higher_is_better: bool
    validator: str = Field(min_length=1)
    smoke_command: str = Field(min_length=1)
    full_command: str = Field(min_length=1)
    pass_criteria: str = Field(min_length=1)
    # Pack-bench autoverify (optional): a command template containing
    # "{candidate}" that runs the pack bench with a candidate module plugged
    # in and prints RESULT_CONTRACT JSON as the last stdout line, plus the
    # prose interface contract the candidate module must satisfy.
    candidate_command: str = ""
    candidate_interface: str = ""
    # Optional typed form of candidate_interface (see CandidateSignature).
    candidate_signature: CandidateSignature | None = None
    # Optional record-search command template with "{candidate}" and "{n}":
    # runs the bench at one target dimension/size for record-gap fitness
    # (autosearch --record-dim). Distinct from search/confirm: record mode
    # optimizes tail behavior at a single n, not mean over a split.
    record_command: str = ""
    # Autosearch (optional): cheap fitness on the TRAIN split for iterating
    # candidates, and the one-shot HOLDOUT confirmation (disjoint instances,
    # unseen seeds) whose result becomes the recorded label. Both are
    # "{candidate}" templates like candidate_command.
    search_command: str = ""
    confirm_command: str = ""


class BaselineSpec(_StrictBase):
    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    level: Literal["naive", "strong_reproducible", "record_cached"]
    command: str
    notes: str


class BaselinesFile(_StrictBase):
    baselines: list[BaselineSpec] = Field(min_length=1)


class DatasetSpec(_StrictBase):
    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    kind: str = Field(min_length=1)
    availability: str = Field(min_length=1)
    command: str
    notes: str


class DatasetsFile(_StrictBase):
    datasets: list[DatasetSpec] = Field(min_length=1)


class SotaSpec(_StrictBase):
    level: Literal["naive", "strong_reproducible", "record_cached"]
    available: bool
    comparator: str = Field(min_length=1)
    source: str
    notes: str


class SotaFile(_StrictBase):
    bar_levels: list[SotaSpec] = Field(min_length=1)


class RecordEntry(_StrictBase):
    """One published best-known (or proven-optimal) value, with provenance."""

    id: str = Field(min_length=1)
    params: dict[str, int | float | str]
    value: float
    status: Literal["proven_optimal", "best_known"]
    exact_form: str = ""
    found_by: str = ""
    source: str = Field(min_length=1)
    precision_note: str = ""


class RecordsFile(_StrictBase):
    """Cached record table for Tier-B (certifiable-record) verification.

    Beating a `best_known` entry — with a checker-verified certificate — IS a
    novel discovery. Beating a `proven_optimal` entry is impossible and means
    the objective conventions don't match; the comparison flags it instead of
    celebrating it.
    """

    domain: str = Field(min_length=1)
    metric: str = Field(min_length=1)
    higher_is_better: bool
    tolerance: float = 1e-9
    records: list[RecordEntry] = Field(min_length=1)
    notes: str = ""


class ComputablePack(_StrictBase):
    pack: PackMeta
    evaluator: EvaluatorSpec
    baselines: BaselinesFile
    datasets: DatasetsFile
    sota: SotaFile
    gotchas: str = ""
    records: RecordsFile | None = None


class PreflightResult(_StrictBase):
    pack_id: str
    computable: bool
    ai_verifiable: bool
    candidate_ready: bool = False
    search_ready: bool = False
    record_ready: bool = False
    missing: list[str] = Field(default_factory=list)
    bar_levels_available: list[str] = Field(default_factory=list)
    recommended_compose_constraints: list[str] = Field(default_factory=list)
    smoke_command: str = ""


def list_packs(root: Path = PACKS_DIR) -> list[str]:
    """Return computable pack ids in deterministic order."""
    if not root.exists():
        return []
    return sorted(p.name for p in root.iterdir() if p.is_dir())


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as e:
        raise ValueError(f"{path}: invalid JSON: {e}") from e


def _validate_file(path: Path, model: type[_StrictBase]) -> _StrictBase:
    try:
        return model.model_validate(_read_json(path))
    except ValidationError as e:
        raise ValueError(f"{path}: schema validation failed: {e}") from e


def load_pack(name: str, root: Path = PACKS_DIR) -> ComputablePack:
    """Load and validate one computable pack."""
    d = root / name
    if not d.is_dir():
        raise FileNotFoundError(f"computable pack {name!r} not found at {d}")

    try:
        pack = _validate_file(d / "pack.json", PackMeta)
        evaluator = _validate_file(d / "evaluator.json", EvaluatorSpec)
        baselines = _validate_file(d / "baselines.json", BaselinesFile)
        datasets = _validate_file(d / "datasets.json", DatasetsFile)
        sota = _validate_file(d / "sota.json", SotaFile)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"{d}: missing required file {Path(e.filename).name}") from e

    if pack.id != name:
        raise ValueError(f"{d}: pack.json id {pack.id!r} does not match directory {name!r}")

    gotchas_path = d / "gotchas.md"
    gotchas = gotchas_path.read_text() if gotchas_path.exists() else ""
    records_path = d / "records.json"
    records = _validate_file(records_path, RecordsFile) if records_path.exists() else None
    return ComputablePack(
        pack=pack,
        evaluator=evaluator,
        baselines=baselines,
        datasets=datasets,
        sota=sota,
        gotchas=gotchas,
        records=records,
    )


def pack_to_dict(pack: ComputablePack) -> dict:
    """JSON-serializable representation of a loaded pack."""
    return pack.model_dump()


def _tokens(text: str) -> set[str]:
    return {
        t
        for t in re.findall(r"[a-z0-9]+", text.lower())
        if len(t) >= 3
    }


def _pack_search_text(pack: ComputablePack) -> str:
    p = pack.pack
    s = p.problem_shape
    parts = [
        p.id,
        p.display_name,
        p.description,
        p.recommended_problem_template,
        s.objective,
        s.input_representation,
        s.output_representation,
        s.verification_kind,
        " ".join(s.constraints),
        " ".join(s.failure_modes),
        " ".join(p.compose_query_hints),
        " ".join(b.name for b in pack.baselines.baselines),
        " ".join(d.name for d in pack.datasets.datasets),
    ]
    return " ".join(parts)


def match_problem(problem: str, root: Path = PACKS_DIR) -> list[dict]:
    """Rank packs by deterministic token overlap against the problem text."""
    query = _tokens(problem)
    matches = []
    for name in list_packs(root):
        pack = load_pack(name, root=root)
        haystack = _tokens(_pack_search_text(pack))
        overlap = sorted(query & haystack)
        score = len(overlap)
        matches.append({
            "pack_id": name,
            "display_name": pack.pack.display_name,
            "score": score,
            "matched_terms": overlap,
        })
    matches.sort(key=lambda m: (-m["score"], m["pack_id"]))
    return matches


def _missing_files(pack_dir: Path) -> list[str]:
    required = ["pack.json", "evaluator.json", "baselines.json", "datasets.json", "sota.json"]
    return [name for name in required if not (pack_dir / name).exists()]


def preflight_pack(name: str, root: Path = PACKS_DIR) -> PreflightResult:
    """Return a structural computability verdict for one pack."""
    pack_dir = root / name
    missing = _missing_files(pack_dir)
    if missing:
        return PreflightResult(
            pack_id=name,
            computable=False,
            ai_verifiable=False,
            missing=missing,
            bar_levels_available=[],
            recommended_compose_constraints=[],
            smoke_command="",
        )

    try:
        pack = load_pack(name, root=root)
    except (FileNotFoundError, ValueError) as e:
        return PreflightResult(
            pack_id=name,
            computable=False,
            ai_verifiable=False,
            missing=[str(e)],
            bar_levels_available=[],
            recommended_compose_constraints=[],
            smoke_command="",
        )

    available_bars = [
        b.level
        for b in pack.sota.bar_levels
        if b.available
    ]
    has_strong_bar = "strong_reproducible" in available_bars
    has_strong_baseline = any(
        b.level == "strong_reproducible" and bool(b.command)
        for b in pack.baselines.baselines
    )
    has_dataset = bool(pack.datasets.datasets)
    has_smoke = bool(pack.evaluator.smoke_command)
    computable = bool(has_smoke and pack.evaluator.validator and pack.evaluator.metric_primary)
    ai_verifiable = computable and has_dataset and has_strong_baseline and has_strong_bar
    constraints = [
        f"objective: {pack.pack.problem_shape.objective}",
        f"validator: {pack.evaluator.validator}",
        f"metric: {pack.evaluator.metric_primary}",
        f"pass: {pack.evaluator.pass_criteria}",
    ]

    missing_fields = []
    if not has_smoke:
        missing_fields.append("evaluator.smoke_command")
    if not has_dataset:
        missing_fields.append("datasets.datasets")
    if not has_strong_baseline:
        missing_fields.append("baselines.strong_reproducible.command")
    if not has_strong_bar:
        missing_fields.append("sota.strong_reproducible.available")

    candidate_ready = bool(
        pack.evaluator.candidate_command
        and "{candidate}" in pack.evaluator.candidate_command
        and pack.evaluator.candidate_interface
    )
    search_ready = bool(
        candidate_ready
        and pack.evaluator.search_command
        and "{candidate}" in pack.evaluator.search_command
        and pack.evaluator.confirm_command
        and "{candidate}" in pack.evaluator.confirm_command
    )

    return PreflightResult(
        pack_id=name,
        computable=computable,
        ai_verifiable=ai_verifiable,
        candidate_ready=candidate_ready,
        search_ready=search_ready,
        record_ready=pack.records is not None,
        missing=missing_fields,
        bar_levels_available=available_bars,
        recommended_compose_constraints=constraints,
        smoke_command=pack.evaluator.smoke_command,
    )


def compare_to_record(pack: ComputablePack, params: dict,
                      value: float) -> dict:
    """Compare an achieved value against the pack's cached record table.

    Verdicts:
      new-record           — beats a best_known entry beyond tolerance
      exceeds-proven-optimal-CHECK-OBJECTIVE — "beats" a proven optimum,
                             which is impossible: the objective conventions
                             don't match; this is a bug, not a discovery
      matches-record       — within tolerance of the record
      below-record         — worse than the record
      no-record            — no entry with these params
    """
    if pack.records is None:
        return {"verdict": "no-record", "reason": "pack has no records.json"}
    rec = next((r for r in pack.records.records if r.params == params), None)
    if rec is None:
        return {"verdict": "no-record", "params": params}
    hib = pack.records.higher_is_better
    tol = pack.records.tolerance
    margin = (value - rec.value) if hib else (rec.value - value)
    if margin > tol:
        verdict = ("exceeds-proven-optimal-CHECK-OBJECTIVE"
                   if rec.status == "proven_optimal" else "new-record")
    elif margin >= -tol:
        verdict = "matches-record"
    else:
        verdict = "below-record"
    return {
        "verdict": verdict,
        "record_id": rec.id,
        "record_value": rec.value,
        "achieved_value": value,
        "margin": margin,
        "status": rec.status,
        "source": rec.source,
        "precision_note": rec.precision_note,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Computable problem pack inspection")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list", help="list available computable packs")

    inspect_p = sub.add_parser("inspect", help="print merged pack JSON")
    inspect_p.add_argument("name")

    match_p = sub.add_parser("match", help="rank packs for a free-form problem")
    match_p.add_argument("--problem", required=True)

    preflight_p = sub.add_parser("preflight", help="check whether a pack is AI-verifiable")
    preflight_p.add_argument("name")

    rec_p = sub.add_parser(
        "record-compare",
        help="compare an achieved value against a pack's cached record table")
    rec_p.add_argument("name")
    rec_p.add_argument("--value", type=float, required=True)
    rec_p.add_argument("--params", required=True,
                       help="comma-separated key=value, e.g. n=11")

    args = parser.parse_args(argv if argv is not None else sys.argv[1:])

    try:
        if args.cmd == "list":
            for name in list_packs():
                pack = load_pack(name)
                print(f"{name}\t{pack.pack.display_name}")
            return 0
        if args.cmd == "inspect":
            print(json.dumps(pack_to_dict(load_pack(args.name)), indent=2))
            return 0
        if args.cmd == "match":
            print(json.dumps(match_problem(args.problem), indent=2))
            return 0
        if args.cmd == "preflight":
            print(json.dumps(preflight_pack(args.name).model_dump(), indent=2))
            return 0
        if args.cmd == "record-compare":
            params: dict = {}
            for kv in args.params.split(","):
                k, _, v = kv.partition("=")
                try:
                    params[k.strip()] = int(v)
                except ValueError:
                    try:
                        params[k.strip()] = float(v)
                    except ValueError:
                        params[k.strip()] = v.strip()
            print(json.dumps(
                compare_to_record(load_pack(args.name), params, args.value),
                indent=2))
            return 0
    except (FileNotFoundError, ValueError) as e:
        print(str(e), file=sys.stderr)
        return 1

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
