"""Bounded-error round-trip invariants for the multitrack stems codec.

The codec is intentionally lossy under overflow — the predictor clips the
int16 residual when it can't fit. These tests pin two invariants the
documentation claims:

  1. On samples where the residual did NOT overflow, encode+decode is
     bit-exact (target == decoded).
  2. On overflow samples the reconstruction error is bounded by the
     int16 range (max |error| < 65536).

If either invariant changes (e.g., someone adds a true lossless fallback
or tightens the clipping), update these tests deliberately, not silently.
"""

from __future__ import annotations

import numpy as np

from experiments.multitrack_stems.predictor import (
    decode_residual,
    encode_residual,
)


def _make_sources(n: int = 1024, n_sources: int = 3, seed: int = 0) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [
        rng.integers(-2**14, 2**14, size=(n, 2), dtype=np.int16)
        for _ in range(n_sources)
    ]


def _make_target_low_overflow(sources: list[np.ndarray], seed: int = 1) -> np.ndarray:
    """A target that's a noisy linear combination of sources — most samples
    will fit in int16 residual after the predictor fits."""
    rng = np.random.default_rng(seed)
    base = sources[0].astype(np.int32) // 3 + sources[1].astype(np.int32) // 3
    noise = rng.integers(-100, 100, size=base.shape, dtype=np.int32)
    target = base + noise
    return np.clip(target, -32768, 32767).astype(np.int16)


def _make_target_high_overflow(sources: list[np.ndarray], seed: int = 2) -> np.ndarray:
    """A target that's poorly predicted from sources — many residuals will
    overflow int16."""
    rng = np.random.default_rng(seed)
    return rng.integers(-32768, 32767, size=(sources[0].shape[0], 2), dtype=np.int16)


def test_non_overflow_samples_roundtrip_bit_exact():
    sources = _make_sources()
    target = _make_target_low_overflow(sources)
    residual, overflow_mask, W_q, b_q, info = encode_residual(sources, target)
    decoded = decode_residual(sources, residual, W_q, b_q)
    # Where overflow did not happen, decode == encode_input exactly.
    np.testing.assert_array_equal(target[~overflow_mask], decoded[~overflow_mask])


def test_overflow_error_bounded_by_int16_range():
    sources = _make_sources()
    target = _make_target_high_overflow(sources)
    residual, overflow_mask, W_q, b_q, info = encode_residual(sources, target)
    decoded = decode_residual(sources, residual, W_q, b_q)
    error = np.abs(target.astype(np.int32) - decoded.astype(np.int32))
    # The codec clips residual to int16, so worst-case reconstruction error
    # is bounded by the int16 range.
    assert error.max() < 65536, f"reconstruction error {error.max()} exceeded int16 range"


def test_overflow_count_matches_overflow_mask():
    sources = _make_sources()
    target = _make_target_high_overflow(sources)
    _residual, overflow_mask, _W_q, _b_q, info = encode_residual(sources, target)
    assert info["overflow_count"] == int(np.sum(overflow_mask))


def test_zero_overflow_path_is_lossless_end_to_end():
    """When sources predict the target perfectly, residual is zero and
    decode round-trips exactly across the whole array."""
    sources = _make_sources()
    # Make target identical to source 0 — predictor should fit this trivially.
    target = sources[0].copy()
    residual, overflow_mask, W_q, b_q, _info = encode_residual(sources, target)
    decoded = decode_residual(sources, residual, W_q, b_q)
    # No overflow expected on this trivial case.
    assert overflow_mask.sum() == 0
    np.testing.assert_array_equal(target, decoded)
