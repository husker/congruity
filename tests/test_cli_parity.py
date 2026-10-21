"""CLI parity: the documented `python -m pipeline compose` path and the
module form `python -m pipeline.compose` must expose the same argument
surface. Pre-fix, the runner-level subparser silently missed
--relevance-mode, --with-recipes, --with-novelty, --strict-gate, and
--skip-gate; users on the README quickstart couldn't reach the new
controls. This test pins the surface so the two cannot drift again.
"""

from __future__ import annotations

import argparse


def _flag_names(parser: argparse.ArgumentParser) -> set[str]:
    """Return the set of long-form flag names (e.g. {'--topic-list', ...})
    declared on `parser`, excluding the auto-generated -h/--help."""
    names: set[str] = set()
    for action in parser._actions:
        for opt in action.option_strings:
            if opt.startswith("--") and opt not in {"--help"}:
                names.add(opt)
    return names


def _module_parser() -> argparse.ArgumentParser:
    """Build the module CLI's parser the same way pipeline.compose.main does."""
    from pipeline.compose import add_compose_args

    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=False)
    sub.add_parser("reprocess")
    add_compose_args(p, required=False)
    return p


def _runner_compose_parser() -> argparse.ArgumentParser:
    """Build a parser equivalent to the runner's `compose` subparser."""
    from pipeline.compose import add_compose_args

    p = argparse.ArgumentParser()
    add_compose_args(p, required=True)
    return p


def test_top_level_and_module_compose_share_args():
    runner_flags = _flag_names(_runner_compose_parser())
    module_flags = _flag_names(_module_parser())
    assert runner_flags == module_flags, (
        "CLI surfaces drifted; mirror new flags in pipeline.compose."
        f"add_compose_args.\nrunner-only: {runner_flags - module_flags}\n"
        f"module-only: {module_flags - runner_flags}"
    )


def test_compose_surface_includes_new_product_controls():
    """The new flags are reachable from both CLIs — this catches the
    specific gap from the 2026-05-22 deep analysis."""
    flags = _flag_names(_runner_compose_parser())
    assert "--relevance-mode" in flags
    assert "--min-gate-papers" in flags
    assert "--recipe-top-k" in flags
    assert "--with-recipes" in flags
    assert "--with-novelty" in flags
    assert "--skip-gate" in flags
    assert "--strict-gate" in flags
    assert "--no-strict-gate" in flags
    assert "--strict-scoring" in flags
    assert "--no-strict-scoring" in flags


def test_runner_main_loads_env():
    """The runner entry point must load .env so OPENAI_API_KEY and
    ANTHROPIC_API_KEY land in os.environ before the relevance backends
    look them up. Pre-fix only pipeline.compose.main loaded .env, which
    silently broke the documented `python -m pipeline compose` path."""
    import inspect

    from pipeline import runner

    src = inspect.getsource(runner.main)
    assert "load_env" in src, (
        "runner.main must call load_env() so the documented CLI picks up "
        ".env keys before subcommand dispatch"
    )
