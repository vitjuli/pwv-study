from __future__ import annotations
import numpy as np
from typing import Tuple

def compute_transit_time(proximal_idx: np.ndarray, distal_idx: np.ndarray, fs: int) -> Tuple[np.ndarray, float]:
    """Pair nearest distal fiducials to proximal ones and compute TT in seconds.
    Returns (tt_array, median_tt).
    """
    if proximal_idx.size == 0 or distal_idx.size == 0:
        return np.array([]), np.nan
    tt = []
    j = 0
    for i in range(len(proximal_idx)):
        p = proximal_idx[i]
        while j < len(distal_idx) and distal_idx[j] < p:
            j += 1
        if j >= len(distal_idx):
            break
        tt.append((distal_idx[j] - p) / fs)
        j += 1
    tt = np.array(tt, dtype=float)
    med = float(np.median(tt)) if tt.size else float('nan')
    return tt, med

def estimate_pwv(distance_m: float, tt_seconds: float) -> float:
    """PWV = distance / TT (m/s)."""
    if not np.isfinite(tt_seconds) or tt_seconds <= 0:
        return float('nan')
    return distance_m / tt_seconds
