"""C5 voice-isolation verification: blind demixing prior (AuxIVA) + ratio mask
vs classical beamforming, on synthetic multi-mic mixtures.

Composition C5 (compose run 20260530-065501): "P24's IVA/NMF determined BSS +
P7's complex ratio masking + P3's deep clustering + P13's CHiME multichannel
setting -> a multichannel phase-aware separator whose BLIND demixing prior
supplies spatial separation."

We verify the cheaply-implementable CORE (skipping the deep-clustering neural
part, which would need GPU/data): determined blind source separation via AuxIVA
(Ono 2011) + a complex ratio-mask post-filter. The composition's distinctive
claim is that the blind prior supplies spatial separation WITHOUT a steering
direction — so the decisive comparison is against beamformers that DO use a DOA,
including when that DOA is imperfect:

  baselines: delay-and-sum (oracle DOA), MVDR (oracle DOA), MVDR (DOA error)
  C5-core:   AuxIVA (blind) and AuxIVA + ratio mask (blind)
  metric:    SI-SDR improvement (dB) of the target estimate, vs the target as
             received at the reference mic. Averaged over seeds.

Pure numpy + scipy. Synthetic non-stationary super-Gaussian sources (white noise
× slow random envelope, speech-band) — independent + non-stationary, so both IVA
(needs non-Gaussianity/non-stationarity) and beamformers (need distinct DOAs)
have a fair shot.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import istft, stft

FS = 8000
C = 340.0  # speed of sound (m/s)


# --------------------------------------------------------------------------- #
# Synthetic sources and anechoic array mixing
# --------------------------------------------------------------------------- #
def make_source(n: int, rng: np.random.Generator) -> np.ndarray:
    """Non-stationary, speech-band-ish source: white noise × slow envelope."""
    x = rng.standard_normal(n)
    # slow amplitude envelope (non-stationarity IVA exploits)
    env = np.abs(rng.standard_normal(n // 400 + 2))
    env = np.interp(np.linspace(0, len(env) - 1, n), np.arange(len(env)), env)
    env = env / (env.max() + 1e-9)
    x = x * (0.1 + env)
    # crude band-pass 300-3400 Hz via FFT mask
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(n, 1 / FS)
    X[(f < 300) | (f > 3400)] = 0
    return np.fft.irfft(X, n)


def _frac_delay(x: np.ndarray, delay_samples: float) -> np.ndarray:
    """Fractional delay via frequency-domain phase shift."""
    n = len(x)
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(n)
    return np.fft.irfft(X * np.exp(-2j * np.pi * f * delay_samples), n)


def mix_anechoic(sources: list[np.ndarray], angles_deg: list[float],
                 n_mics: int, mic_spacing: float, snr_db: float,
                 rng: np.random.Generator):
    """Far-field plane-wave linear array. Returns (mics[n_mics,N], images[K,n_mics,N]).

    images[k,m] = source k as received at mic m (its steered copy) — used to build
    per-method references.
    """
    N = len(sources[0])
    mics = np.zeros((n_mics, N))
    images = np.zeros((len(sources), n_mics, N))
    for k, (s, ang) in enumerate(zip(sources, angles_deg)):
        theta = np.deg2rad(ang)
        for m in range(n_mics):
            tau = m * mic_spacing * np.sin(theta) / C * FS  # samples
            sm = _frac_delay(s, tau)
            images[k, m] = sm
            mics[m] += sm
    # additive white sensor noise at target SNR (relative to total signal power)
    sig_pow = np.mean(mics ** 2)
    noise_pow = sig_pow / (10 ** (snr_db / 10))
    mics += rng.standard_normal(mics.shape) * np.sqrt(noise_pow)
    return mics, images


# --------------------------------------------------------------------------- #
# Metric: SI-SDR (scale-invariant; small best-shift search to remove global delay)
# --------------------------------------------------------------------------- #
def si_sdr(est: np.ndarray, ref: np.ndarray, max_shift: int = 8) -> float:
    best = -np.inf
    for sh in range(-max_shift, max_shift + 1):
        e = np.roll(est, sh)
        a = np.dot(e, ref) / (np.dot(ref, ref) + 1e-12)
        proj = a * ref
        noise = e - proj
        val = 10 * np.log10((np.dot(proj, proj) + 1e-12) / (np.dot(noise, noise) + 1e-12))
        best = max(best, val)
    return best


# --------------------------------------------------------------------------- #
# Beamformers (frequency domain), oracle or noisy steering
# --------------------------------------------------------------------------- #
def _steering(n_mics, mic_spacing, angle_deg, freqs):
    theta = np.deg2rad(angle_deg)
    taus = np.arange(n_mics) * mic_spacing * np.sin(theta) / C  # seconds
    # [F, M]
    return np.exp(-2j * np.pi * np.outer(freqs, taus))


def _stft_multi(mics):
    Xs = []
    for m in range(mics.shape[0]):
        f, t, Z = stft(mics[m], fs=FS, nperseg=512, noverlap=384)
        Xs.append(Z)
    return f, t, np.stack(Xs, axis=0)  # [M, F, T]


def _istft_one(Z):
    _, x = istft(Z, fs=FS, nperseg=512, noverlap=384)
    return x


def delay_and_sum(mics, n_mics, mic_spacing, angle_deg):
    f, _t, X = _stft_multi(mics)           # [M,F,T]
    d = _steering(n_mics, mic_spacing, angle_deg, f)  # [F,M]
    Y = np.einsum("fm,mft->ft", np.conj(d), X) / n_mics
    return _istft_one(Y)


def mvdr(mics, n_mics, mic_spacing, angle_deg):
    f, _t, X = _stft_multi(mics)           # [M,F,T]
    d = _steering(n_mics, mic_spacing, angle_deg, f)  # [F,M]
    Y = np.zeros((X.shape[1], X.shape[2]), dtype=complex)
    for fi in range(X.shape[1]):
        Xf = X[:, fi, :]                   # [M,T]
        R = (Xf @ Xf.conj().T) / Xf.shape[1] + 1e-3 * np.eye(n_mics)
        Rinv = np.linalg.inv(R)
        df = d[fi]                         # [M]
        w = Rinv @ df / (df.conj() @ Rinv @ df + 1e-12)
        Y[fi] = w.conj() @ Xf
    return _istft_one(Y)


# --------------------------------------------------------------------------- #
# AuxIVA (Ono 2011), determined (n_mics == n_sources), Laplacian source model
# --------------------------------------------------------------------------- #
def auxiva(mics, n_iter: int = 40):
    _f, _t, X = _stft_multi(mics)          # [M,F,T]
    M, F, T = X.shape
    Xp = X.transpose(1, 0, 2)              # [F,M,T]
    W = np.tile(np.eye(M, dtype=complex), (F, 1, 1))  # [F,M,M]
    Y = np.einsum("fnm,fmt->fnt", W, Xp)   # [F,N,T]
    eye = np.eye(M, dtype=complex)
    for _ in range(n_iter):
        # source-wise auxiliary update (iterative projection)
        r = np.sqrt(np.sum(np.abs(Y) ** 2, axis=0)) + 1e-8   # [N,T] cross-freq norm
        g = 1.0 / r                                          # Laplacian weight [N,T]
        for n in range(M):
            # V_n[f] = mean_t g[n,t] x x^H
            V = np.einsum("t,fmt,fkt->fmk", g[n], Xp, np.conj(Xp)) / T  # [F,M,M]
            WV = W @ V                                        # [F,M,M]
            rhs = np.broadcast_to(eye[n], (F, M)).copy()[:, :, None]  # [F,M,1]
            w = np.linalg.solve(WV, rhs)[:, :, 0]             # [F,M] = (WV)^-1 e_n
            denom = np.einsum("fm,fmk,fk->f", np.conj(w), V, w)
            w = w / np.sqrt(np.abs(denom)[:, None] + 1e-12)
            W[:, n, :] = np.conj(w)
        Y = np.einsum("fnm,fmt->fnt", W, Xp)                 # [F,N,T]
    # minimal-distortion principle: scale rows by diag(W^-1)
    for fi in range(F):
        Winv = np.linalg.pinv(W[fi])
        W[fi] = np.diag(np.diag(Winv)) @ W[fi]
    Y = np.einsum("fnm,fmt->fnt", W, Xp)                     # [F,N,T]
    return [_istft_one(Y[:, n, :]) for n in range(M)], Y


def ratio_mask_refine(Y_stft, target_idx: int):
    """Complex ratio mask: refine the target output by its share of total energy."""
    mag = np.abs(Y_stft) ** 2                                # [F,N,T]
    total = mag.sum(axis=1) + 1e-9                           # [F,T]
    mask = mag[:, target_idx, :] / total                     # [F,T] in [0,1]
    return _istft_one(Y_stft[:, target_idx, :] * mask)
