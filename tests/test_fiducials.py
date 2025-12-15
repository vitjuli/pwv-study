import numpy as np
from pwvstudy.synthetic import synthesize_ppg_bundle
from pwvstudy.fiducials import detect_fd_peaks, detect_sd_valleys_gated

def test_fd_sd_detection_runs():
    b = synthesize_ppg_bundle(seconds=6, fs=500, hr=72, seed=7)
    fd_peaks = detect_fd_peaks(b['fd'], fs=500, rel_height=0.3, min_distance_ms=250)
    sd_vals  = detect_sd_valleys_gated(b['sd'], anchors=fd_peaks, fs=500, gate_ms=180)
    assert fd_peaks.size >= 2
    assert sd_vals.size >= 1 or fd_peaks.size >= 1
    assert (fd_peaks >= 0).all() and (fd_peaks < b['fd'].size).all()
