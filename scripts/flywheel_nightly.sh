#!/bin/zsh
# Nightly flywheel cycle: one compose + autosearch pass per ready pack,
# then regenerate the web dashboard. Designed for launchd/cron — see
# docs/flywheel-scheduling.md for installation.
#
# Every long stage is resumable (compose --resume, autosearch evals.json),
# so a machine-sleep or kill mid-run costs nothing: the next night's run
# picks up where this one stopped.

set -euo pipefail
cd "$(dirname "$0")/.."

LOG_DIR="data/pipeline/nightly"
mkdir -p "$LOG_DIR"
STAMP="$(date +%Y%m%d-%H%M%S)"
LOG="$LOG_DIR/$STAMP.log"

echo "=== flywheel nightly $STAMP ===" | tee -a "$LOG"

# Health-check the packs first; skip the night if a pack is broken rather
# than paying for a corpus against a bench that can't run.
uv run python -m pipeline pack doctor --no-smoke 2>&1 | tee -a "$LOG"

# One flywheel cycle per ready pack. Add packs here as they become
# candidate+search ready (pack doctor shows readiness).
uv run python scripts/flywheel.py \
    --pack graph_coloring \
    --problem "Develop a novel local-search graph coloring method that beats TabuCol on DIMACS-style instances" \
    --paper-search "graph coloring local search;tabu search graph coloring;phase transition graph coloring" \
    2>&1 | tee -a "$LOG" || echo "flywheel graph_coloring failed (see log)" | tee -a "$LOG"

# Refresh the static dashboard from whatever landed.
uv run python -m pipeline web 2>&1 | tee -a "$LOG"

echo "=== done $(date +%H:%M:%S) ===" | tee -a "$LOG"
