"""Load environment variables from a repo-root .env file.

Why it lives in its own module: any CLI entry point in pipeline/ can call
``load_env()`` as its first line and pick up secrets without depending on
how the user launched the shell. The function is idempotent — calling it
more than once is a no-op — and it never overwrites existing process env
vars, so explicit ``export`` in the shell still wins.

The .env file is git-ignored (see .gitignore). A `.env.example` template
sits at repo root with the supported keys filled in as placeholders.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = REPO_ROOT / ".env"


def load_env(path: Path | None = None, override: bool = False) -> Path | None:
    """Load .env into os.environ.

    Returns the path that was loaded (or None if no file was found).
    With ``override=False`` (the default) existing env vars are preserved,
    so shells that already export a key take precedence over the file.
    """
    target = path or ENV_PATH
    if not target.exists():
        return None
    # python-dotenv is the standard; do a lazy import so importing this
    # module doesn't fail in test environments where dotenv isn't installed.
    try:
        from dotenv import load_dotenv
    except ImportError:
        return None
    load_dotenv(dotenv_path=target, override=override)
    return target
