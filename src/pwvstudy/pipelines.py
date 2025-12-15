from __future__ import annotations
import numpy as np
from .preprocess import lowpass_fir, polynomial_detrend
from .fiducials import detect_fd_peaks, detect_sd_valleys_gated
from .metrics import compute_transit_time, estimate_pwv
from typing import Dict

class SegmentalProcessing:
    """RTP-like pipeline: filter -> (optional detrend) -> FD peaks -> SD valleys (gated) -> TT/PWV."""
    def __init__(self, fs: int, cutoff: float = 10.0):
        self.fs = fs
        self.cutoff = cutoff

    def run(self, proximal: np.ndarray, distal: np.ndarray, distance_m: float) -> Dict[str, float]:
        x_p = lowpass_fir(proximal, fs=self.fs, cutoff=self.cutoff)
        x_d = lowpass_fir(distal,   fs=self.fs, cutoff=self.cutoff)

        fd_p = np.gradient(x_p) * self.fs
        fd_d = np.gradient(x_d) * self.fs
        sd_p = np.gradient(fd_p) * self.fs
        sd_d = np.gradient(fd_d) * self.fs

        fp = detect_fd_peaks(fd_p, fs=self.fs, rel_height=0.35, min_distance_ms=300)
        fd = detect_fd_peaks(fd_d, fs=self.fs, rel_height=0.35, min_distance_ms=300)
        vp = detect_sd_valleys_gated(sd_p, anchors=fp, fs=self.fs, gate_ms=180)
        vd = detect_sd_valleys_gated(sd_d, anchors=fd, fs=self.fs, gate_ms=180)

        tt, tt_med = compute_transit_time(vp, vd, fs=self.fs)
        pwv = estimate_pwv(distance_m, tt_med)
        return {"tt_median": float(tt_med), "pwv": float(pwv), "n_pairs": int(tt.size)}

class TransitTimeMethod:
    """TT-like pipeline using peak-to-peak or valley-to-valley logic on filtered signals."""
    def __init__(self, fs: int, cutoff: float = 10.0):
        self.fs = fs
        self.cutoff = cutoff

    def run(self, proximal: np.ndarray, distal: np.ndarray, distance_m: float) -> Dict[str, float]:
        x_p = lowpass_fir(proximal, fs=self.fs, cutoff=self.cutoff)
        x_d = lowpass_fir(distal,   fs=self.fs, cutoff=self.cutoff)
        # Simple: use derivative peaks as fiducials
        fd_p = np.gradient(x_p) * self.fs
        fd_d = np.gradient(x_d) * self.fs
        fp = detect_fd_peaks(fd_p, fs=self.fs, rel_height=0.4, min_distance_ms=280)
        fd = detect_fd_peaks(fd_d, fs=self.fs, rel_height=0.4, min_distance_ms=280)
        tt, tt_med = compute_transit_time(fp, fd, fs=self.fs)
        pwv = estimate_pwv(distance_m, tt_med)
        return {"tt_median": float(tt_med), "pwv": float(pwv), "n_pairs": int(tt.size)}

class CompositePWV:
    """GP-like: combine multiple fiducial strategies and choose robust median."""
    def __init__(self, fs: int, cutoff: float = 10.0):
        self.fs = fs
        self.cutoff = cutoff

    def run(self, proximal: np.ndarray, distal: np.ndarray, distance_m: float) -> Dict[str, float]:
        x_p = lowpass_fir(proximal, fs=self.fs, cutoff=self.cutoff)
        x_d = lowpass_fir(distal,   fs=self.fs, cutoff=self.cutoff)

        fd_p = np.gradient(x_p) * self.fs
        fd_d = np.gradient(x_d) * self.fs
        sd_p = np.gradient(fd_p) * self.fs
        sd_d = np.gradient(fd_d) * self.fs

        # Strategy A: FD→SD gated
        fp = detect_fd_peaks(fd_p, fs=self.fs, rel_height=0.35, min_distance_ms=300)
        fd_ = detect_fd_peaks(fd_d, fs=self.fs, rel_height=0.35, min_distance_ms=300)
        vp = detect_sd_valleys_gated(sd_p, anchors=fp, fs=self.fs, gate_ms=180)
        vd = detect_sd_valleys_gated(sd_d, anchors=fd_, fs=self.fs, gate_ms=180)
        tt_a, tt_a_med = compute_transit_time(vp, vd, fs=self.fs)

        # Strategy B: FD→FD
        tt_b, tt_b_med = compute_transit_time(fp, fd_, fs=self.fs)

        # Combine: robust median across strategies
        medians = [m for m in (tt_a_med, tt_b_med) if np.isfinite(m)]
        tt_robust = float(np.median(medians)) if medians else float('nan')
        pwv = estimate_pwv(distance_m, tt_robust)
        return {"tt_median": tt_robust, "pwv": float(pwv), "n_pairs": int(max(len(tt_a), len(tt_b)))}
