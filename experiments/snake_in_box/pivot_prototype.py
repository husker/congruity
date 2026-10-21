"""Pivot-move prototype for SIB (polymer Monte Carlo's workhorse, per the
untried-ideas memo). Move: pick joint i, pick sigma in S_9 (stabilizer of
v_i), map suffix w -> v_i ^ sigma(w ^ v_i). Prefix-internal and
suffix-internal legality are automatic; only prefix<->suffix interactions
need checking. Measures: global acceptance rate on our certificates and
the acceptance profile by pivot position (the 'stiffness' diagnostic).
"""
import json
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from experiments.snake_in_box.baselines import validate_snake  # noqa: E402 — bootstrap imports

N = 9


def apply_sigma(sigma, x):
    y = 0
    for b in range(N):
        if x >> b & 1:
            y |= 1 << sigma[b]
    return y


def pivot(path, i, sigma):
    """Pivot at joint i (suffix = path[i+1:]). Returns new path or None."""
    vi = path[i]
    prefix = path[: i + 1]
    pset = set(prefix)
    new_suffix = []
    for w in path[i + 1:]:
        nw = vi ^ apply_sigma(sigma, w ^ vi)
        new_suffix.append(nw)
    # collisions and chords against the prefix
    for j, w in enumerate(new_suffix):
        if w in pset:
            return None
        for b in range(N):
            u = w ^ (1 << b)
            if u in pset:
                # allowed only: joint edge (first suffix vertex - v_i)
                if not (j == 0 and u == vi):
                    return None
    return prefix + new_suffix


def profile(path, trials_per_site=200, rng=random.Random(1)):
    L = len(path)
    ok, length = validate_snake(N, path)
    assert ok
    accept = []
    total_acc = 0
    for i in range(1, L - 1):
        a = 0
        for _ in range(trials_per_site):
            sigma = list(range(N))
            rng.shuffle(sigma)
            if pivot(path, i, sigma) is not None:
                a += 1
        accept.append(a / trials_per_site)
        total_acc += a
    return accept, total_acc / (trials_per_site * (L - 2))


def main():
    for name, fp in (
            ("ours-185", "data/snake_in_box/certificates/q9_len185_joint.json"),
            ("wynn-190", "data/snake_in_box/certificates/q9_len190_wynn.json"),
            ("router-181", "data/snake_in_box/hrouter_seed3.json")):
        d = json.loads((REPO / fp).read_text())
        path = d.get("seq") or d.get("path")
        acc, overall = profile(path)
        # summarize: overall rate, and where the loose joints are
        loose = sorted(range(len(acc)), key=lambda k: -acc[k])[:8]
        print(f"{name}: {len(path)-1}e  overall acceptance {overall:.4f}")
        print(f"  loosest joints (pos: rate): "
              f"{[(k+1, round(acc[k], 3)) for k in loose]}")
        nonzero = sum(1 for a in acc if a > 0)
        print(f"  joints with ANY legal pivot: {nonzero}/{len(acc)}")


if __name__ == "__main__":
    main()
