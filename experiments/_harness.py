"""Shared harness utilities for experiments/*/bench.py.

Every bench used to re-type the same four things by hand — BLAS thread caps,
seeding schemes, wall-clock budget loops, and the flywheel's RESULT_CONTRACT
JSON — and the copies drifted (the thread-cap fix that produced a
"flywheel-wide speedup" existed in only 2 of ~30 benches). New benches should
import from here.

Usage skeleton:

    from experiments._harness import cap_threads
    cap_threads()          # BEFORE importing numpy/scipy/jax

    import numpy as np
    from experiments._harness import budget_loop, emit_result_contract, make_rng

    rng = make_rng(1234, n_points, trial)
    for _ in budget_loop(seconds=600):
        step(rng)
    emit_result_contract(
        metric="min_triangle_area", higher_is_better=True,
        scores={"composition": best, "sota_sa": baseline},
        composition_method="composition", home_field_sota_method="sota_sa",
        n_trials=trials, notes="one line",
    )
"""

from __future__ import annotations

import json
import os
import time
from typing import Iterator

_THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def cap_threads(n: int = 1) -> None:
    """Cap BLAS/OpenMP thread pools. MUST run before importing numpy.

    Parallel benches oversubscribe cores when every worker's BLAS spawns a
    full thread pool — on Heilbronn this made a single n=13 eval look like a
    6-hour hang. Explicit shell exports still win (values already in the
    environment are left alone).
    """
    for var in _THREAD_ENV_VARS:
        os.environ.setdefault(var, str(n))


def make_rng(*components: int):
    """Independent numpy Generator from a tuple of integer components.

    Replaces the ad-hoc `seed + 100_003 * n + 1000 * trial` offset schemes:
    numpy's SeedSequence hashes the component tuple, so streams for
    different (seed, n, trial, ...) tuples are decorrelated by construction.
    Imports numpy lazily so cap_threads() can be called first.
    """
    import numpy as np

    return np.random.default_rng(list(components))


def budget_loop(seconds: float, *, min_iterations: int = 1) -> Iterator[int]:
    """Yield iteration indices until the wall-clock budget is spent.

    Guarantees at least ``min_iterations`` yields so a tight budget still
    produces one measurement. Time is checked before each yield; the loop
    does not preempt a running iteration.
    """
    start = time.perf_counter()
    i = 0
    while i < min_iterations or (time.perf_counter() - start) < seconds:
        yield i
        i += 1
        if i >= min_iterations and (time.perf_counter() - start) >= seconds:
            return


REQUIRED_CONTRACT_KEYS = (
    "metric",
    "higher_is_better",
    "scores",
    "composition_method",
    "home_field_sota_method",
    "n_trials",
    "notes",
)


def build_result_contract(
    *,
    metric: str,
    higher_is_better: bool,
    scores: dict[str, float],
    composition_method: str,
    home_field_sota_method: str,
    n_trials: int,
    notes: str = "",
    **extra,
) -> dict:
    """Assemble and validate a RESULT_CONTRACT dict (see pipeline/autoverify.py).

    Raises ValueError when the two named methods are missing from scores —
    the exact malformed-output class the flywheel would otherwise quarantine
    at runtime. Extra keys (per_trial, per_instance, record_claims, ...)
    pass through.
    """
    if composition_method not in scores:
        raise ValueError(
            f"composition_method {composition_method!r} not in scores "
            f"({sorted(scores)})")
    if home_field_sota_method not in scores:
        raise ValueError(
            f"home_field_sota_method {home_field_sota_method!r} not in scores "
            f"({sorted(scores)})")
    return {
        "metric": metric,
        "higher_is_better": bool(higher_is_better),
        "scores": scores,
        "composition_method": composition_method,
        "home_field_sota_method": home_field_sota_method,
        "n_trials": int(n_trials),
        "notes": notes,
        **extra,
    }


def emit_result_contract(**kwargs) -> dict:
    """Validate, print as the FINAL stdout line (flush), and return the dict.

    autoverify.execute_pack_bench parses the last stdout line as JSON —
    anything printed after this breaks the contract.
    """
    contract = build_result_contract(**kwargs)
    print(json.dumps(contract), flush=True)
    return contract
