import numpy as np
from pwvstudy.metrics import compute_transit_time, estimate_pwv

def test_tt_and_pwv():
    fs = 1000
    p = np.array([100, 400, 700])
    d = np.array([180, 480, 780])
    tt, med = compute_transit_time(p, d, fs=fs)
    assert tt.size == 3
    assert abs(med - 0.08) < 1e-6  # 80 ms delay
    v = estimate_pwv(0.8, med)
    assert abs(v - 10.0) < 1e-6
