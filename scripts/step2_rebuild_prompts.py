"""Rebuild step2 prompts from a cleaned corpus.json without re-fetching.

Run this after manually editing corpus.json (e.g. to drop a paper whose
abstract was contaminated by OpenAlex's indexing).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Import the build_prompt and redaction logic from the main script.
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from step2_codex_experiment import build_prompt  # noqa: E402


def main() -> int:
    exp_dir = SCRIPT_DIR.parent / "data" / "step2_experiment"
    with (exp_dir / "corpus.json").open() as f:
        d = json.load(f)

    pl, os_ = d["pl"], d["os"]
    print(f"PL: {len(pl)}, OS: {len(os_)}")
    if len(pl) < 1 or len(os_) < 1:
        print("Insufficient corpus.", file=sys.stderr)
        return 1

    (exp_dir / "prompt_clear.txt").write_text(build_prompt(pl, os_, redacted=False))
    (exp_dir / "prompt_redacted.txt").write_text(build_prompt(pl, os_, redacted=True))
    print("Prompts rebuilt.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
