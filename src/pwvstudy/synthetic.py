from __future__ import annotations
import numpy as np
from typing import Dict

def _lognormal_pulse(t: np.ndarray, t0: float, amp: float, width: float, skew: float) -> np.ndarray:
    x = (t - t0) / max(width, 1e-6)
    x = np.clip(x, 1e-9, None)
    y = amp * np.exp(-0.5 * (np.log(x) / skew) ** 2) / (x * np.sqrt(2*np.pi) * abs(skew))
    y[t < t0] = 0.0
    return y

def synthesize_ppg_bundle(seconds: float = 30.0, fs: int = 1000, hr: float = 70.0, hrv: float = 0.02,
                          amp: float = 1.0, width: float = 0.15, skew: float = 0.35,
                          noise_snr_db: float = 20.0, mains_hz: float | None = 50.0,
                          baseline_drift_hz: float = 0.15, motion_steps: int = 0,
                          seed: int | None = 123) -> Dict[str, np.ndarray]:
    """Return dict with keys: t, ppg (noisy), clean, fd, sd.

    Generates a clean PPG-like signal via pulses on a mildly varying HR trajectory,
    then applies noise and simple artifacts.
    """
    rng = np.random.default_rng(seed)
    n = int(seconds * fs)
    t = np.arange(n) / fs

    # HRV as bounded random walk
    hr_series = hr * (1.0 + hrv * np.cumsum(rng.normal(0, 0.005, size=n)))
    hr_series = np.clip(hr_series, 35, 180)
    inst_period = 60.0 / hr_series

    beats = [0.3]
    while beats[-1] < seconds:
        dt = inst_period[int(min(len(inst_period)-1, beats[-1]*fs))]
        beats.append(beats[-1] + float(dt))
    beats = np.array(beats[1:])

    clean = np.zeros_like(t)
    for bt in beats:
        clean += _lognormal_pulse(t, bt, amp=amp, width=width, skew=skew)

    # normalize clean
    if clean.ptp() > 0:
        clean = (clean - clean.min()) / (clean.ptp())

    x = clean.copy()
    # Additive white noise with target SNR
    p_sig = np.mean(x**2) + 1e-12
    snr = 10**(noise_snr_db/10)
    p_noise = p_sig / snr
    noise = rng.normal(0, np.sqrt(p_noise), size=x.size)
    x = x + noise

    # mains hum
    if mains_hz is not None:
        x += 0.02 * np.sin(2*np.pi*mains_hz*t + rng.uniform(0, 2*np.pi))

    # baseline drift
    if baseline_drift_hz > 0:
        x += 0.1 * np.sin(2*np.pi*baseline_drift_hz*t + rng.uniform(0, 2*np.pi))

    # motion steps
    for _ in range(max(0, motion_steps)):
        i = rng.integers(low=int(0.1*n), high=int(0.9*n))
        x[i:] += rng.choice([-1,1]) * 0.2 * rng.uniform(0.4, 1.2)

    # derivatives
    fd = np.gradient(x) * fs
    sd = np.gradient(fd) * fs

    return {"t": t, "ppg": x, "clean": clean, "fd": fd, "sd": sd}
