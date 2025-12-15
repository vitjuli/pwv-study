from __future__ import annotations
import numpy as np
from scipy.signal import firwin, lfilter
from typing import Tuple

def lowpass_fir(x: np.ndarray, fs: int, cutoff: float = 10.0, taps: int = 129) -> np.ndarray:
    b = firwin(taps, cutoff=cutoff, fs=fs)
    return lfilter(b, [1.0], x)

def polynomial_detrend(x: np.ndarray, order: int = 3) -> np.ndarray:
    n = x.size
    t = np.arange(n)
    A = np.vander(t, N=order+1, increasing=True)
    coeffs, *_ = np.linalg.lstsq(A, x, rcond=None)
    trend = A @ coeffs
    return x - trend

def trim_seconds(t: np.ndarray, x: np.ndarray, start: float, end: float) -> Tuple[np.ndarray, np.ndarray]:
    assert t.shape == x.shape
    i0 = int(max(0.0, start) * (len(t) / (t[-1]-t[0]+1e-12)))
    i1 = int(min(t[-1], end) * (len(t) / (t[-1]-t[0]+1e-12)))
    return t[i0:i1], x[i0:i1]
