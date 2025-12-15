from __future__ import annotations
import numpy as np
from typing import Tuple

def bootstrap_ci(values: np.ndarray, iters: int = 1000, alpha: float = 0.05, seed: int | None = 123) -> Tuple[float,float]:
    rng = np.random.default_rng(seed)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float('nan'), float('nan')
    dist = []
    for _ in range(iters):
        sample = rng.choice(values, size=values.size, replace=True)
        dist.append(np.median(sample))
    lo = float(np.percentile(dist, 100*alpha/2))
    hi = float(np.percentile(dist, 100*(1-alpha/2)))
    return lo, hi

def bland_altman(ref: np.ndarray, test: np.ndarray) -> Tuple[float, float, float]:
    """Return (mean_diff, loa_lo, loa_hi) with ±1.96 SD limits of agreement."""
    mask = np.isfinite(ref) & np.isfinite(test)
    d = test[mask] - ref[mask]
    mean_diff = float(np.mean(d))
    sd = float(np.std(d, ddof=1)) if d.size > 1 else float('nan')
    loa_lo = mean_diff - 1.96*sd if sd==sd else float('nan')
    loa_hi = mean_diff + 1.96*sd if sd==sd else float('nan')
    return mean_diff, loa_lo, loa_hi
