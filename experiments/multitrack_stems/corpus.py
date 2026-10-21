"""MUSDB18 sample dataset loader for the multitrack stem benchmark.

The `musdb` package auto-downloads a free 144-clip sample subset (each ~6.8s)
into the uv cache. Stems are AAC-coded inside the .stem.mp4 container, but
after PCM decoding they look like any modern multitrack production output.
For Phase 1 (cross-stem entropy reduction signal check), this is fine.

Each track gives us 4 stems × stereo @ 44.1 kHz. We quantize to 16-bit
signed PCM for FLAC/WavPack compatibility.
"""

from __future__ import annotations

import numpy as np


STEM_NAMES = ["vocals", "drums", "bass", "other"]


def _to_int16(audio: np.ndarray) -> np.ndarray:
    """Convert float64 audio in [-1,1] to int16 PCM with clipping.

    Use the same scale as standard WAV/FLAC: int16 sample = round(float * 32767).
    """
    clipped = np.clip(audio, -1.0, 1.0)
    return np.round(clipped * 32767.0).astype(np.int16)


def load_track(track) -> dict[str, np.ndarray]:
    """Return {stem_name: int16 array [N, 2]} for one musdb track."""
    out = {}
    for name in STEM_NAMES:
        src = track.sources[name]
        out[name] = _to_int16(src.audio)
    return out


def load_corpus(max_tracks: int | None = None):
    """Yield (track_name, dict_of_int16_stems) for the musdb sample subset."""
    import musdb
    mus = musdb.DB(download=True)
    tracks = list(mus)
    if max_tracks is not None:
        tracks = tracks[:max_tracks]
    for t in tracks:
        yield t.name, load_track(t)


def stem_stats(stems: dict[str, np.ndarray]) -> dict[str, dict]:
    """Per-stem stats: variance per channel, peak, sample count."""
    out = {}
    for name, arr in stems.items():
        out[name] = {
            "shape": arr.shape,
            "var": float(np.var(arr.astype(np.float64))),
            "peak": int(np.max(np.abs(arr))),
            "samples": int(arr.shape[0]),
        }
    return out


if __name__ == "__main__":
    for name, stems in load_corpus(max_tracks=3):
        print(f"\n{name}")
        for sn, st in stem_stats(stems).items():
            print(f"  {sn:8s} shape={st['shape']} var={st['var']:>12.0f} peak={st['peak']}")
