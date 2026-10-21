#!/bin/zsh
# Seed portfolio at the full router config (pool 4 + cap schedule), each
# assembly saved and immediately deepened. Ledger-checkpointed per seed:
# safe to kill and rerun with the remaining seeds.
set -u
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
for s in "$@"; do
  $PY -m experiments.snake_in_box.hierarchical_router \
    --ks 9:1,8:1 --beam 6 --stop-step 1 --node-cap 20000000 \
    --final-cap 300000000 --probe-cap 5000000 --probe-top 24 \
    --leaf-pool 4 --rng-seed "$s" \
    --out "data/snake_in_box/hrouter_seed${s}.json"
  $PY -m experiments.snake_in_box.route_then_deepen \
    --in "data/snake_in_box/hrouter_seed${s}.json" \
    --widths 8,12,16 --extras 1,2 --unwind 60 \
    --polish-cap 300000000 --window-cap 50000000 --stride 2 \
    --checkpoint "data/snake_in_box/deepen_ck_s${s}.json"
done
