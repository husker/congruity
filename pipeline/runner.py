"""End-to-end pipeline orchestrator + CLI.

Ties Layers 1–4 together:
    1. fetch topics (cached)
    2. filter pairs by cross-citation density (cached)
    3. run Codex discovery on top-N passing pairs
    4. rank + write digest

CLI:
    uv run python -m pipeline --help
    uv run python -m pipeline run               # full batch
    uv run python -m pipeline run --dry-run     # skip codex, validate pipeline shape
    uv run python -m pipeline run --max-pairs 5 # cap codex calls (budget control)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from pipeline._openalex import extract_id
from pipeline.compose import add_compose_args, run_compose_from_args
from pipeline.digest import render_digest, write_digest
from pipeline.discover import discover_one
from pipeline.pair_filter import (
    DENSITY_CACHE,
    DEFAULT_SAMPLE,
    DEFAULT_THRESHOLD,
    DEFAULT_TOP_N_TOPICS,
    build_density_matrix,
    fetch_all_samples,
)
from pipeline.rank import score_bridge
from pipeline.topics import load_or_fetch
from pipeline.web_ui import build_site as build_web_site

DEFAULT_MAX_PAIRS = 50
DEFAULT_CONFIDENCE_FLOOR = 7
DEFAULT_TOP_N = 10
PIPELINE_DATA = Path(__file__).resolve().parent.parent / "data" / "pipeline"
FINDINGS_DIR = Path(__file__).resolve().parent.parent / "findings"


def cmd_topics(args: argparse.Namespace) -> int:
    topics = load_or_fetch(force=args.force)
    print(f"{len(topics)} CS topics available.")
    return 0


def cmd_filter(args: argparse.Namespace) -> int:
    topics = load_or_fetch()
    if not topics:
        print("No topics. Run `python -m pipeline topics` first.", file=sys.stderr)
        return 1
    selected = sorted(topics, key=lambda t: -t.get("works_count", 0))[:args.top_n_topics]
    print(f"Sampling {len(selected)} topics × {args.sample_size} papers...")
    samples = fetch_all_samples([t["id"] for t in selected], args.sample_size)
    print("Computing density matrix...")
    pairs = build_density_matrix(samples, threshold=args.threshold)
    passing = [p for p in pairs if p["passes_threshold"]]
    name_by_id = {extract_id(t["id"]): t["display_name"] for t in selected}
    for p in pairs:
        p["a_name"] = name_by_id.get(p["a"], p["a"])
        p["b_name"] = name_by_id.get(p["b"], p["b"])
    DENSITY_CACHE.parent.mkdir(parents=True, exist_ok=True)
    DENSITY_CACHE.write_text(json.dumps({
        "threshold": args.threshold,
        "sample_size": args.sample_size,
        "topic_count": len(selected),
        "total_pairs": len(pairs),
        "passing_pairs": len(passing),
        "pairs": pairs,
    }, indent=2))
    print(f"Passing pairs: {len(passing)}/{len(pairs)}.")
    print(f"Saved: {DENSITY_CACHE}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    """Full batch: topics → filter → discover → rank → digest."""
    started_at = datetime.now(timezone.utc).isoformat()
    print("\n=== congruity v1 batch pipeline ===")
    print(f"Started: {started_at}")
    print(f"Config: max-pairs={args.max_pairs}, threshold={args.threshold:.0%}, "
          f"confidence-floor={args.confidence_floor}, dry-run={args.dry_run}\n")

    # Layer 1
    topics = load_or_fetch()
    if not topics:
        print("ERROR: no topics", file=sys.stderr)
        return 1
    selected = sorted(topics, key=lambda t: -t.get("works_count", 0))[:args.top_n_topics]
    name_by_id = {extract_id(t["id"]): t["display_name"] for t in selected}
    print(f"Layer 1: {len(selected)} topics selected (top by works_count)\n")

    # Layer 2 — use cached density matrix if present, else compute
    pairs_data: dict | None = None
    if DENSITY_CACHE.exists() and not args.refilter:
        pairs_data = json.loads(DENSITY_CACHE.read_text())
        # Reuse cache only if topic pool matches
        if pairs_data.get("topic_count") != len(selected):
            print("  cache topic_count mismatch; refiltering")
            pairs_data = None
    if pairs_data is None:
        print(f"Layer 2: sampling {args.sample_size} papers per topic + density...")
        samples = fetch_all_samples([t["id"] for t in selected], args.sample_size)
        pairs = build_density_matrix(samples, threshold=args.threshold)
        for p in pairs:
            p["a_name"] = name_by_id.get(p["a"], p["a"])
            p["b_name"] = name_by_id.get(p["b"], p["b"])
        pairs_data = {
            "threshold": args.threshold,
            "sample_size": args.sample_size,
            "topic_count": len(selected),
            "total_pairs": len(pairs),
            "passing_pairs": sum(1 for p in pairs if p["passes_threshold"]),
            "pairs": pairs,
        }
        DENSITY_CACHE.write_text(json.dumps(pairs_data, indent=2))

    passing = [p for p in pairs_data["pairs"] if p["passes_threshold"]]
    print(f"Layer 2: {len(passing)} pairs pass density threshold (of "
          f"{pairs_data['total_pairs']} total)\n")

    if not passing:
        print("No pairs passed. Nothing to discover. Exiting.")
        return 0

    # Cap to max_pairs (lowest density first = most-separated)
    to_run = passing[:args.max_pairs]
    print(f"Layer 3: running discovery on top {len(to_run)} most-separated pairs"
          f"{' (dry run)' if args.dry_run else ''}\n")

    discovery_results: list[dict] = []
    pair_lookup: dict[str, dict] = {}
    for i, pp in enumerate(to_run, 1):
        a_topic = next((t for t in selected if extract_id(t["id"]) == pp["a"]), None)
        b_topic = next((t for t in selected if extract_id(t["id"]) == pp["b"]), None)
        if not a_topic or not b_topic:
            continue
        print(f"[{i}/{len(to_run)}] {a_topic['display_name']} ↔ {b_topic['display_name']}")
        print(f"  density: A={pp['density']['a_ratio']:.1%} B={pp['density']['b_ratio']:.1%}")
        result = discover_one(a_topic, b_topic, dry_run=args.dry_run)
        discovery_results.append(result)
        pair_lookup[result["pair_id"]] = pp
        print(f"  status: {result.get('status')}")
        if result.get("confidence") is not None:
            print(f"  confidence: {result['confidence']}/10")
        check = result.get("citation_check")
        if check:
            print(f"  citations: {check['grounded']} grounded, "
                  f"{check['confabulated']} confabulated → "
                  f"{'PASS' if check['passes'] else 'FAIL'}")
        print()
        time.sleep(0.5)

    # Layer 4 — rank + digest
    print("Layer 4: scoring + digest")
    scored = [score_bridge(r, pair_lookup.get(r["pair_id"], {})) for r in discovery_results]

    run_meta = {
        "topics_count": len(selected),
        "pairs_total": pairs_data["total_pairs"],
        "pairs_passing": len(passing),
        "pairs_processed": len(to_run),
        "started_at": started_at,
        "ended_at": datetime.now(timezone.utc).isoformat(),
        "dry_run": args.dry_run,
    }
    digest = render_digest(scored, run_meta,
                           top_n=args.top_n,
                           confidence_floor=args.confidence_floor)
    out_path = (args.output if args.output else
                FINDINGS_DIR / f"digest-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}.md")
    write_digest(digest, out_path)

    # Persist full run for inspection / autoplan aggregation
    run_path = PIPELINE_DATA / "runs.jsonl"
    run_path.parent.mkdir(parents=True, exist_ok=True)
    with run_path.open("a") as f:
        f.write(json.dumps({
            "meta": run_meta,
            "results": scored,
        }) + "\n")

    print(f"\nDigest: {out_path}")
    print(f"Run log: {run_path}")
    print("Done.")
    return 0


def cmd_compose(args: argparse.Namespace) -> int:
    """Compositional synthesis on a topic list + problem statement.

    Delegates to ``pipeline.compose.run_compose_from_args`` so the
    top-level documented CLI exposes the same surface as the module form.
    """
    return run_compose_from_args(args)


def cmd_web(args: argparse.Namespace) -> int:
    """Build the static product-facing web UI from saved compose runs."""
    manifest = build_web_site(args.root, args.out, args.index)
    print(f"wrote {manifest['index']}")
    print(f"wrote {len(manifest['run_pages'])} run page(s)")
    return 0


def cmd_autosearch(args: argparse.Namespace) -> int:
    """Evolutionary candidate search on a pack bench; one holdout-labeled outcome."""
    from datetime import datetime, timezone

    from pipeline.autosearch import orchestrate_search
    summary = orchestrate_search(
        args.run, args.cid, datetime.now(timezone.utc), pack_id=args.pack,
        population=args.population, generations=args.generations,
        elites=args.elites, seed_candidate=args.candidate_seed,
        arm=args.arm, record_dim=args.record_dim, fulltext=args.fulltext,
        function_skeleton=args.function_skeleton)
    print(json.dumps(summary, indent=2))
    return 0 if "error" not in summary else 1


def cmd_pack(args: argparse.Namespace) -> int:
    """Computable-pack tooling: doctor, new, forge, promote."""
    from pipeline.computable import list_packs
    from pipeline.pack_tools import doctor, scaffold

    if args.pack_cmd == "forge":
        from pipeline.pack_forge import STAGES, forge, workspace_for
        state = forge(args.pack_id, args.domain,
                      redo=args.redo or [],
                      reproduce_timeout_s=args.reproduce_timeout)
        ws = workspace_for(args.pack_id)
        print(f"\nevidence: {ws / 'evidence.md'}")
        all_ok = all(state["stages"].get(s, {}).get("ok") for s in STAGES)
        if all_ok:
            print(f"all stages passed — review the evidence, then run:\n"
                  f"  pipeline pack promote {args.pack_id} --approve \"<your name>\"")
        return 0 if all_ok else 1

    if args.pack_cmd == "promote":
        from pipeline.pack_forge import promote
        try:
            prov = promote(args.pack_id, args.approve,
                           waive=args.waive or [],
                           run_doctor=not args.no_doctor)
        except (ValueError, FileExistsError, RuntimeError) as e:
            print(f"pack promote: {e}")
            return 1
        print(json.dumps(prov, indent=2))
        print(f"\npromoted: computable_packs/{args.pack_id}/ is now live for "
              "autoverify/autosearch. Remember: records.json entries must be "
              "hand-transcribed from the snapshots with sources.")
        return 0

    if args.pack_cmd == "new":
        try:
            written = scaffold(args.pack_id)
        except (ValueError, FileExistsError) as e:
            print(f"pack new: {e}")
            return 1
        for p in written:
            print(f"wrote {p}")
        print("\nNext: fill in the JSON templates + bench, then run "
              f"`pipeline pack doctor {args.pack_id}`. Record tables "
              "(records.json) must contain only human-verified published "
              "values.")
        return 0

    ids = [args.pack_id] if args.pack_id else list_packs()
    all_ok = True
    for pid in ids:
        report = doctor(pid, run_smoke=not args.no_smoke,
                        smoke_timeout_s=args.smoke_timeout)
        print(report.render())
        print()
        all_ok = all_ok and report.ok
    return 0 if all_ok else 1


def cmd_autoverify(args: argparse.Namespace) -> int:
    """Verify one composition on a computable pack's bench; record the outcome."""
    from datetime import datetime, timezone

    from pipeline.autoverify import orchestrate
    summary = orchestrate(args.run, args.cid, datetime.now(timezone.utc),
                          pack_id=args.pack, candidate_path=args.candidate,
                          reuse_candidate=args.reuse_candidate)
    print(json.dumps(summary, indent=2))
    return 0 if "error" not in summary else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="pipeline", description="congruity v1 batch discovery")
    sub = p.add_subparsers(dest="cmd", required=True)

    pt = sub.add_parser("topics", help="Fetch / cache CS topics")
    pt.add_argument("--force", action="store_true")
    pt.set_defaults(func=cmd_topics)

    pf = sub.add_parser("filter", help="Compute cross-cite density matrix")
    pf.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE)
    pf.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    pf.add_argument("--top-n-topics", type=int, default=DEFAULT_TOP_N_TOPICS)
    pf.set_defaults(func=cmd_filter)

    pr = sub.add_parser("run", help="Full batch: topics → filter → discover → digest (equivalence-finding)")
    pr.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE)
    pr.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    pr.add_argument("--top-n-topics", type=int, default=DEFAULT_TOP_N_TOPICS)
    pr.add_argument("--max-pairs", type=int, default=DEFAULT_MAX_PAIRS,
                    help="Max pairs to run through Codex (budget cap)")
    pr.add_argument("--top-n", type=int, default=DEFAULT_TOP_N,
                    help="Top-N bridges in the digest")
    pr.add_argument("--confidence-floor", type=int, default=DEFAULT_CONFIDENCE_FLOOR,
                    help="Min Codex confidence for digest inclusion")
    pr.add_argument("--dry-run", action="store_true",
                    help="Skip Codex calls; validate pipeline shape only")
    pr.add_argument("--refilter", action="store_true",
                    help="Re-run Layer 2 even if cached")
    pr.add_argument("--output", type=Path, default=None)
    pr.set_defaults(func=cmd_run)

    pc = sub.add_parser(
        "compose",
        help="Compositional synthesis: find subsets of papers that compose into a stronger solution",
    )
    add_compose_args(pc, required=True)
    pc.set_defaults(func=cmd_compose)

    pw = sub.add_parser(
        "web",
        help="Generate the static evidence dashboard and per-run pages",
    )
    pw.add_argument("--root", type=Path, default=PIPELINE_DATA / "compositions")
    pw.add_argument("--out", type=Path, default=FINDINGS_DIR / "web")
    pw.add_argument("--index", type=Path, default=FINDINGS_DIR / "INDEX.md")
    pw.set_defaults(func=cmd_web)

    pav = sub.add_parser(
        "autoverify",
        help="Run a composition on a computable pack's human-written bench and "
             "record a labeled outcome (vs the pack's home-field SOTA)")
    pav.add_argument("--run", required=True, help="compose run_id")
    pav.add_argument("--cid", required=True, help="composition id, e.g. C2")
    pav.add_argument("--pack", required=True,
                     help="computable pack id providing the trusted bench, "
                          "e.g. graph_coloring")
    pav.add_argument("--candidate", default=None,
                     help="path to a human-written candidate module; skips codegen")
    pav.add_argument("--reuse-candidate", action="store_true",
                     help="reuse an existing generated candidate.py instead of regenerating")
    pav.set_defaults(func=cmd_autoverify)

    pas = sub.add_parser(
        "autosearch",
        help="Evolutionary candidate search on a pack bench (train split), "
             "confirmed once on the holdout split; records one labeled outcome")
    pas.add_argument("--run", required=True, help="compose run_id")
    pas.add_argument("--cid", required=True, help="composition id, e.g. C2")
    pas.add_argument("--pack", required=True,
                     help="search-ready computable pack id, e.g. graph_coloring")
    pas.add_argument("--population", type=int, default=4,
                     help="initial candidates generated from the composition (default 4)")
    pas.add_argument("--generations", type=int, default=3,
                     help="max refinement generations (default 3)")
    pas.add_argument("--elites", type=int, default=2,
                     help="top candidates refined each generation (default 2)")
    pas.add_argument("--candidate-seed", default=None,
                     help="optional human-written module added to the initial population")
    pas.add_argument("--arm", default="composition",
                     choices=["composition", "unseeded", "evolve-sota"],
                     help="seeding-experiment arm; control arms (unseeded, "
                          "evolve-sota) run the same loop and budget but do "
                          "not write the outcomes ledger")
    pas.add_argument("--record-dim", type=int, default=None,
                     help="record mode: optimize best value at this target "
                          "dimension/size via the pack's record_command; no "
                          "holdout, no ledger row — output is a stronger "
                          "method + certificate for the record miner")
    pas.add_argument("--fulltext", action="store_true",
                     help="fetch OA full text for the composition's papers "
                          "and show method excerpts to codegen (needs "
                          "pdftotext; closed papers stay abstract-only)")
    pas.add_argument("--function-skeleton", default=None,
                     help="FunSearch mode: path to a frozen solver skeleton; "
                          "codegen evolves ONLY the marked priority-function "
                          "region (spliced automatically)")
    pas.set_defaults(func=cmd_autosearch)

    pk = sub.add_parser(
        "pack",
        help="Computable-pack tools: doctor (structural + smoke health check), "
             "new (scaffold a template pack + bench skeleton)")
    pksub = pk.add_subparsers(dest="pack_cmd", required=True)
    pkd = pksub.add_parser("doctor", help="Check one pack (or all packs)")
    pkd.add_argument("pack_id", nargs="?", default=None,
                     help="pack id; omit to check every pack")
    pkd.add_argument("--no-smoke", action="store_true",
                     help="skip executing the smoke command (structural checks only)")
    pkd.add_argument("--smoke-timeout", type=int, default=300)
    pkd.set_defaults(func=cmd_pack)
    pkn = pksub.add_parser("new", help="Scaffold computable_packs/<id>/ + experiments/<id>/bench.py")
    pkn.add_argument("pack_id")
    pkn.set_defaults(func=cmd_pack)
    pkf = pksub.add_parser(
        "forge",
        help="Auto-draft a pack with cited SOTA, pin + reproduce the "
             "implementation, battery-test the validator, snapshot record "
             "sources, and emit an evidence bundle. Output stays in "
             "data/pipeline/forge/<id>/ until promoted.")
    pkf.add_argument("pack_id")
    pkf.add_argument("--domain", required=True,
                     help="problem/domain description for the draft")
    pkf.add_argument("--redo", action="append", default=None,
                     metavar="STAGE",
                     help="re-run a stage even if it already passed "
                          "(draft|pin|reproduce|battery|records|evidence)")
    pkf.add_argument("--reproduce-timeout", type=int, default=3600)
    pkf.set_defaults(func=cmd_pack)
    pkp = pksub.add_parser(
        "promote",
        help="Human signature: copy a forged pack into computable_packs/ "
             "after a final doctor gate. Requires all forge stages passed.")
    pkp.add_argument("pack_id")
    pkp.add_argument("--approve", required=True,
                     help="your name — recorded in the pack's provenance.json")
    pkp.add_argument("--waive", action="append", default=None, metavar="STAGE",
                     help="promote despite a failed stage (recorded in provenance)")
    pkp.add_argument("--no-doctor", action="store_true",
                     help="skip the final doctor smoke gate (not recommended)")
    pkp.set_defaults(func=cmd_pack)

    return p


def main() -> int:
    # Load .env so API keys (Anthropic, OpenAI) reach os.environ before
    # the relevance backends look them up. Mirrors pipeline.compose.main
    # so the documented `python -m pipeline compose` path behaves the same.
    from pipeline.env import load_env
    load_env()
    args = build_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
