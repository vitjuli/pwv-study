from __future__ import annotations
import numpy as np
from scipy.signal import find_peaks

def detect_fd_peaks(fd: np.ndarray, fs: int, rel_height: float = 0.3, min_distance_ms: int = 300) -> np.ndarray:
    """Baseline first-derivative peaks with a relative height threshold and refractory window."""
    height = fd.min() + rel_height * (fd.max() - fd.min())
    distance = max(1, int(min_distance_ms * fs / 1000))
    peaks, _ = find_peaks(fd, height=height, distance=distance)
    return peaks

def detect_sd_valleys_gated(sd: np.ndarray, anchors: np.ndarray, fs: int, gate_ms: int = 180) -> np.ndarray:
    """Find SD local minima within a right-open gate after each anchor (FD peak).
    This addresses typical right-shifts and double-peak artefacts seen in practice.
    """
    valleys = []
    g = max(1, int(gate_ms * fs / 1000))
    for a in anchors:
        i0 = a
        i1 = min(sd.size, a + g)
        if i0 >= i1:
            continue
        local = np.argmin(sd[i0:i1])
        valleys.append(i0 + int(local))
    if not valleys:
        return np.array([], dtype=int)
    return np.array(valleys, dtype=int)
