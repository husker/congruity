"""Domain pack loader + apply CLI.

A domain pack is a directory under ``domain_packs/<name>/`` containing:

  topic_queries.json       — curated OpenAlex topic IDs for this domain
  problem_templates.json   — problem statement templates with example fills
  known_baselines.json     — problem-family → canonical baselines list
  evaluation_datasets.json — datasets per problem family
  transient_deps.json      — recommended dev-only deps per family
  gotchas.md               — recurring traps a new user should skim

Usage::

    uv run python -m pipeline.domain_pack list
    uv run python -m pipeline.domain_pack apply <name> --problem "..." [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PACKS_DIR = Path(__file__).resolve().parent.parent / "domain_packs"


def list_packs() -> list[str]:
    if not PACKS_DIR.exists():
        return []
    return sorted(p.name for p in PACKS_DIR.iterdir() if p.is_dir())


def load_pack(name: str) -> dict:
    d = PACKS_DIR / name
    if not d.is_dir():
        raise FileNotFoundError(f"domain pack {name!r} not found at {d}")
    pack: dict = {}
    for fname in [
        "topic_queries",
        "problem_templates",
        "known_baselines",
        "evaluation_datasets",
        "transient_deps",
    ]:
        p = d / f"{fname}.json"
        pack[fname] = json.loads(p.read_text()) if p.exists() else None
    gotchas_path = d / "gotchas.md"
    pack["gotchas"] = gotchas_path.read_text() if gotchas_path.exists() else ""
    return pack


def apply_pack(name: str, problem: str, dry_run: bool = True) -> dict:
    """Stamp the chosen pack into a structured summary that a future compose
    run can consume. Currently always dry-run; live integration with
    compose_one is a follow-up — the value of even the dry-run summary
    is that it surfaces baselines / datasets / gotchas in one place
    before the user kicks off the LLM-heavy stages."""
    pack = load_pack(name)
    gotchas = pack.get("gotchas", "") or ""
    return {
        "domain": name,
        "problem": problem,
        "topic_queries": pack.get("topic_queries") or [],
        "known_baselines_hint": pack.get("known_baselines") or {},
        "evaluation_datasets_hint": pack.get("evaluation_datasets") or {},
        "transient_deps_hint": pack.get("transient_deps") or {},
        "gotchas_preview": gotchas[:500] + ("..." if len(gotchas) > 500 else ""),
        "dry_run": dry_run,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Domain pack inspection + apply")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="list available packs")
    ap = sub.add_parser("apply", help="apply a pack to a problem")
    ap.add_argument("name")
    ap.add_argument("--problem", required=True)
    ap.add_argument("--dry-run", action="store_true", default=True,
                    help="print what would be applied (default; live apply not yet wired)")
    args = p.parse_args(argv if argv is not None else sys.argv[1:])
    if args.cmd == "list":
        for name in list_packs():
            print(name)
        return 0
    if args.cmd == "apply":
        out = apply_pack(args.name, args.problem, dry_run=args.dry_run)
        print(json.dumps(out, indent=2))
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
