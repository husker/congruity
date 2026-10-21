"""C5 verification bench: blind AuxIVA (+mask) vs beamformers across conditions.

Decisive question: does C5's BLIND demixing prior match/beat DOA-based
beamformers on SI-SDR — including when the beamformer DOA is imperfect?

Run:
    uv run --with scipy python -m experiments.voice_isolation_c5.bench
"""

from __future__ import annotations

import numpy as np

from experiments.voice_isolation_c5.separation import (
    FS,
    auxiva,
    delay_and_sum,
    make_source,
    mix_anechoic,
    mvdr,
    ratio_mask_refine,
    si_sdr,
)


def one_trial(n_src: int, snr_db: float, doa_err: float, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    n = FS * 3
    srcs = [make_source(n, rng) for _ in range(n_src)]
    # well-separated random angles
    base = np.linspace(-50, 50, n_src)
    angles = [float(a + rng.uniform(-8, 8)) for a in base]
    M = n_src  # determined
    spacing = 0.05
    mics, images = mix_anechoic(srcs, angles, M, spacing, snr_db, rng)
    target = 0
    ref = images[target, 0]
    L = len(ref)

    def sd(x):
        return si_sdr(x[:L], ref[:L])

    out = {"input": sd(mics[0])}
    out["delaysum_oracle"] = sd(delay_and_sum(mics, M, spacing, angles[target]))
    out["mvdr_oracle"] = sd(mvdr(mics, M, spacing, angles[target]))
    out["mvdr_doaerr"] = sd(mvdr(mics, M, spacing, angles[target] + doa_err))
    ests, Y = auxiva(mics, n_iter=40)
    sis = [sd(e) for e in ests]
    ti = int(np.argmax(sis))
    out["auxiva_blind"] = sis[ti]
    out["c5_iva_mask"] = sd(ratio_mask_refine(Y, ti))
    return out


def run(conditions, seeds):
    methods = ["input", "delaysum_oracle", "mvdr_oracle", "mvdr_doaerr",
               "auxiva_blind", "c5_iva_mask"]
    for n_src, snr in conditions:
        agg = {m: [] for m in methods}
        for s in range(seeds):
            r = one_trial(n_src, snr, doa_err=10.0, seed=s)
            for m in methods:
                agg[m].append(r[m])
        inp = np.mean(agg["input"])
        print(f"\n=== {n_src} sources / {n_src} mics, SNR {snr} dB, "
              f"{seeds} seeds (DOA err 10 deg for mvdr_doaerr) ===")
        print(f"  {'method':18} {'SI-SDR dB':>10} {'SI-SDRi dB':>11}")
        for m in methods:
            mean = np.mean(agg[m])
            imp = mean - inp
            tag = "  (blind)" if m.startswith(("auxiva", "c5")) else ""
            print(f"  {m:18} {mean:10.2f} {imp:11.2f}{tag}")


if __name__ == "__main__":
    run(conditions=[(2, 20), (2, 10), (3, 20)], seeds=8)
