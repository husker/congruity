"""Linear cross-stem predictor for multitrack residual coding.

For each target stem T (int16 [N, 2]) and source stems S1, S2, S3:
    Stack sources as X = [S1 | S2 | S3]  shape [N, 6]
    Fit T = X @ W + b  via per-output-channel least squares
    Quantize W, b to integers (Q15) so encode/decode use the same arithmetic
    Predict_int = round((X @ W_q + b_q) >> 15)
    Residual_int = T - Predict_int  (int16 subtraction)

The decoder receives W_q (12 int16 + 2 int32 bias ≈ 32 bytes) plus the residual
stem, applies the same integer math, and reconstructs T.

**Lossy under overflow.** When the residual (target - prediction) falls
outside the int16 range, the encoder clips it to [-32768, 32767] and
increments overflow_count. The decoder cannot recover the original sample
in that case — the reconstruction error is bounded by twice the int16
range. In the experimental corpora this rate is well under 0.1%, so
results computed against the clipped residual are reported as-is, but the
codec is **not** bit-exact and must not be described as lossless. See
tests/test_multitrack_roundtrip.py for the formal invariants.

Optional temporal taps (k>0): for each source channel, use samples at
[t-k, t-k+1, ..., t+k] (zero-padded at edges). Costs more side-info bytes
but can capture short delays between stems (kick→bass settling, etc.).
"""

from __future__ import annotations

import numpy as np

Q_SHIFT = 15
Q_SCALE = 1 << Q_SHIFT  # 32768


def _stack_sources_with_taps(sources: list[np.ndarray], taps: int = 0) -> np.ndarray:
    """Stack sources (each [N, 2]) into X [N, 2*len(sources)*(2*taps+1)].

    With taps=0 we get 6 cols for 3 stereo sources (same-time samples only).
    With taps=2 we get 6*5 = 30 cols (samples at t-2,-1,0,+1,+2 per channel).
    """
    n = sources[0].shape[0]
    cols = []
    for src in sources:
        for c in range(src.shape[1]):
            col = src[:, c].astype(np.float64)
            for k in range(-taps, taps + 1):
                if k == 0:
                    cols.append(col)
                else:
                    shifted = np.zeros_like(col)
                    if k < 0:
                        shifted[-k:] = col[: n + k]
                    else:
                        shifted[: n - k] = col[k:]
                    cols.append(shifted)
    return np.stack(cols, axis=1)  # [N, n_cols]


def fit_linear(
    sources: list[np.ndarray],
    target: np.ndarray,
    taps: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Least-squares fit W (shape [n_cols, 2]) and b (shape [2]) on int16 arrays."""
    X = _stack_sources_with_taps(sources, taps=taps)
    Y = target.astype(np.float64)  # [N, 2]
    # Solve per output channel: append 1-column for bias
    X_aug = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)
    sol, *_ = np.linalg.lstsq(X_aug, Y, rcond=None)
    W = sol[:-1]   # [n_cols, 2]
    b = sol[-1]    # [2]
    return W, b


def _integer_predict(
    X: np.ndarray, W_q: np.ndarray, b_q: np.ndarray,
) -> np.ndarray:
    """Decoder-reproducible integer prediction. Mirrors the math in encode_residual
    so the decoder gets bit-identical pred_q values on non-overflow samples."""
    X_i = X.astype(np.int64)
    W_i = W_q.astype(np.int64)
    b_i = b_q.astype(np.int64)
    pred_q = (X_i @ W_i + b_i) >> Q_SHIFT
    return np.clip(pred_q, -32768, 32767).astype(np.int16)


def encode_residual(
    sources: list[np.ndarray],
    target: np.ndarray,
    taps: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """Compute integer residual for target stem given source stems.

    Returns (residual_int16, overflow_mask, W_quant_int16, b_quant_int32, info).
    Decoder reconstructs target as:
        pred = _integer_predict(X, W_q, b_q)
        target_approx = pred + residual_int16
    On overflow samples, the reconstruction may differ from the original.

    `info` includes per-sample variance reduction (float).
    """
    W, b = fit_linear(sources, target, taps=taps)
    X = _stack_sources_with_taps(sources, taps=taps)
    # Quantize weights
    W_q = np.round(W * Q_SCALE).astype(np.int32)
    b_q = np.round(b * Q_SCALE).astype(np.int32)
    # Float-true prediction (for variance calc)
    pred_float = X @ W + b
    pred_q = _integer_predict(X, W_q, b_q)
    residual = (target.astype(np.int32) - pred_q.astype(np.int32))
    residual_i16 = np.clip(residual, -32768, 32767).astype(np.int16)
    # Track which samples overflowed so callers (and tests) can reason about
    # them. The codec is bit-exact off the mask and bounded-error on it.
    overflow_mask = ((residual < -32768) | (residual > 32767))
    overflow_count = int(np.sum(overflow_mask))
    info = {
        "raw_var": float(np.var(target.astype(np.float64))),
        "residual_var_float": float(np.var((target.astype(np.float64) - pred_float))),
        "residual_var_int": float(np.var(residual_i16.astype(np.float64))),
        "overflow_count": overflow_count,
        "n_taps": taps,
        "n_cols": W_q.shape[0],
    }
    # Return W_q as int32 — quantized weights at scale 2^15 routinely hit
    # values >= 32768 when the underlying coefficient is near ±1, and
    # truncating to int16 silently wraps those (-32768 instead of 32768).
    # Cost is 4 bytes/weight instead of 2; that's accounted for in
    # side_info_bytes via .nbytes.
    return residual_i16, overflow_mask, W_q, b_q, info


def decode_residual(
    sources: list[np.ndarray],
    residual_i16: np.ndarray,
    W_q: np.ndarray,
    b_q: np.ndarray,
    taps: int = 0,
) -> np.ndarray:
    """Reconstruct the target stem from sources + side-info + residual.

    Returns an int16 array of the same shape as the original target. Bit-exact
    on samples where the residual fit in int16; otherwise differs by at most
    twice the int16 range from the original (see overflow_mask returned by
    encode_residual).
    """
    X = _stack_sources_with_taps(sources, taps=taps)
    pred_q = _integer_predict(X, W_q.astype(np.int32), b_q.astype(np.int32))
    target_approx = (pred_q.astype(np.int32) + residual_i16.astype(np.int32))
    return np.clip(target_approx, -32768, 32767).astype(np.int16)


def side_info_bytes(W_q: np.ndarray, b_q: np.ndarray) -> int:
    """Side-info size = quantized weight matrix + bias vector."""
    return W_q.nbytes + b_q.nbytes
