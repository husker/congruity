"""congruity v1 batch discovery pipeline.

Implements Layers 1–4 of the discovery architecture from
~/.gstack/projects/congruity/james-nobranch-design-20260516-synthesis.md:

    Layer 1 — topics.py        fetch OpenAlex CS topics
    Layer 2 — pair_filter.py   cross-citation density filter at scale
    Layer 3 — discover.py      Codex bridge discovery + citation check
                (uses corpus.py for shared OpenAlex paper fetching)
    Layer 4 — rank.py, digest.py  score bridges, produce markdown digest
    Runner  — runner.py        end-to-end orchestrator + CLI

Layer 5 (verification, oracle/) is out of v1 scope and remains manual.

CLI: `uv run python -m pipeline --help`
"""

__version__ = "0.0.1"
