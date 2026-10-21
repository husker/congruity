"""FLAC / WavPack compression baselines via CLI subprocess.

We use real codec CLIs (flac, wavpack) because the Python wrappers are spotty
and we want canonical compressed-byte numbers.
"""

from __future__ import annotations

import subprocess
import tempfile
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import soundfile as sf


@contextmanager
def _temp_dir():
    with tempfile.TemporaryDirectory(prefix="stems_") as d:
        yield Path(d)


def _write_wav(path: Path, pcm: np.ndarray, rate: int = 44100) -> None:
    """Write a WAV file. pcm shape: [N, channels]. dtype: int16.

    Uses soundfile (libsndfile). For >2-channel data this produces a
    WAVE_FORMAT_EXTENSIBLE container that FLAC accepts; the stdlib `wave`
    module cannot.
    """
    if pcm.dtype != np.int16:
        raise ValueError(f"expected int16, got {pcm.dtype}")
    sf.write(str(path), pcm, rate, subtype="PCM_16")


def _flac_bytes(wav_path: Path, level: int = 8) -> int:
    """Encode wav -> flac at given level, return compressed bytes."""
    flac_path = wav_path.with_suffix(".flac")
    r = subprocess.run(
        ["flac", f"-{level}", "--silent", "--force", "-o", str(flac_path), str(wav_path)],
        capture_output=True, check=False,
    )
    if r.returncode != 0:
        raise RuntimeError(f"flac failed: {r.stderr.decode()[:200]}")
    return flac_path.stat().st_size


def _flac_bytes_raw(pcm: np.ndarray, rate: int = 44100, level: int = 8) -> int:
    """Encode raw int16 PCM (interleaved) directly via flac stdin.

    Avoids the soundfile WAV-header issue for >2 channel data: FLAC's --raw
    mode accepts any channel count via stdin.
    """
    if pcm.dtype != np.int16:
        raise ValueError(f"expected int16, got {pcm.dtype}")
    channels = pcm.shape[1] if pcm.ndim == 2 else 1
    args = [
        "flac", f"-{level}", "--silent", "--force",
        "--force-raw-format",
        "--endian=little", "--sign=signed", "--bps=16",
        f"--sample-rate={rate}", f"--channels={channels}",
        "-o", "-", "-",
    ]
    r = subprocess.run(args, input=pcm.tobytes(), capture_output=True, check=False)
    if r.returncode != 0:
        raise RuntimeError(f"flac raw failed: {r.stderr.decode()[:200]}")
    return len(r.stdout)


def _wavpack_bytes(wav_path: Path, level: int = 4) -> int:
    """Encode wav -> wv at given level (-h, -hh, -x4 etc.). Multichannel native.

    Levels: 1 (fast), 2 (default), 3 (high -h), 4 (very high -hh).
    """
    wv_path = wav_path.with_suffix(".wv")
    args = ["wavpack", "-y", "-q"]
    if level >= 4:
        args.append("-hh")
    elif level >= 3:
        args.append("-h")
    args.extend(["-o", str(wv_path), str(wav_path)])
    r = subprocess.run(args, capture_output=True, check=False)
    if r.returncode != 0:
        raise RuntimeError(f"wavpack failed: {r.stderr.decode()[:200]}")
    return wv_path.stat().st_size


def per_stem_flac_bytes(stems: dict[str, np.ndarray], rate: int = 44100) -> dict:
    """Encode each stem to FLAC separately. Returns total bytes + per-stem.

    'raw_total' = uncompressed PCM total (denominator for ratio).
    """
    out = {"per_stem": {}, "total_flac": 0, "raw_total": 0}
    with _temp_dir() as d:
        for name, arr in stems.items():
            wav = d / f"{name}.wav"
            _write_wav(wav, arr, rate=rate)
            raw = wav.stat().st_size  # includes 44-byte header
            comp = _flac_bytes(wav, level=8)
            out["per_stem"][name] = {"raw": raw, "flac": comp}
            out["total_flac"] += comp
            out["raw_total"] += raw
    return out


def joint_multichannel_flac_bytes(stems: dict[str, np.ndarray], rate: int = 44100) -> int:
    """Pack all stems as one 8-channel PCM stream, FLAC encode via stdin.

    FLAC supports up to 8 channels natively. This is a stronger baseline than
    per-stem FLAC because it can do limited cross-channel decorrelation
    within the stereo pair structure (FLAC's joint stereo only operates on
    stereo pairs; for 8ch it codes each channel independently — but we still
    include this as a baseline).
    """
    arrs = [stems[n] for n in ("vocals", "drums", "bass", "other")]
    n_samples = min(a.shape[0] for a in arrs)
    arrs = [a[:n_samples] for a in arrs]
    packed = np.concatenate(arrs, axis=1).astype(np.int16)  # [N, 8]
    return _flac_bytes_raw(packed, rate=rate, level=8)


def joint_multichannel_wavpack_bytes(stems: dict[str, np.ndarray], rate: int = 44100) -> int:
    """Pack all stems as one 8-channel WAV, WavPack encode.

    WavPack does multichannel decorrelation (MPEG-4 ALS-like) — the adversary's
    required baseline. Uses -hh (very high mode).
    """
    arrs = [stems[n] for n in ("vocals", "drums", "bass", "other")]
    n_samples = min(a.shape[0] for a in arrs)
    arrs = [a[:n_samples] for a in arrs]
    packed = np.concatenate(arrs, axis=1)
    with _temp_dir() as d:
        wav = d / "joint8ch.wav"
        _write_wav(wav, packed, rate=rate)
        return _wavpack_bytes(wav, level=4)


def encode_arr_with_flac(arr: np.ndarray, rate: int = 44100) -> int:
    """Encode an int16 [N, C] array with FLAC, return compressed bytes."""
    with _temp_dir() as d:
        wav = d / "x.wav"
        _write_wav(wav, arr, rate=rate)
        return _flac_bytes(wav, level=8)
