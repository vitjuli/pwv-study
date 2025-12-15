from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt

def plot_signals(t: np.ndarray, ppg: np.ndarray, fd: np.ndarray | None = None, sd: np.ndarray | None = None,
                 peaks_p: np.ndarray | None = None, peaks_d: np.ndarray | None = None, out: str | None = None) -> None:
    plt.figure(figsize=(10,4))
    plt.plot(t, ppg, label='PPG')
    if fd is not None:
        plt.plot(t, (fd - fd.min())/(fd.ptp()+1e-12) - 1.2, label='FD (scaled)')
    if sd is not None:
        plt.plot(t, (sd - sd.min())/(sd.ptp()+1e-12) - 2.4, label='SD (scaled)')
    if peaks_p is not None and peaks_p.size:
        plt.scatter(t[peaks_p], ppg[peaks_p], marker='o', label='prox fid')
    if peaks_d is not None and peaks_d.size:
        plt.scatter(t[peaks_d], ppg[peaks_d], marker='x', label='dist fid')
    plt.xlabel('Time (s)'); plt.ylabel('a.u.'); plt.legend(); plt.tight_layout()
    if out: plt.savefig(out, dpi=160)

def plot_bland_altman(ref: np.ndarray, test: np.ndarray, out: str | None = None) -> None:
    mean = (ref + test)/2
    diff = test - ref
    m = np.isfinite(mean) & np.isfinite(diff)
    mean, diff = mean[m], diff[m]
    plt.figure(figsize=(5,4))
    plt.scatter(mean, diff, s=12)
    md, lo, hi = np.mean(diff), np.mean(diff) - 1.96*np.std(diff, ddof=1), np.mean(diff) + 1.96*np.std(diff, ddof=1)
    plt.axhline(md, linestyle='--')
    plt.axhline(lo, linestyle=':'); plt.axhline(hi, linestyle=':')
    plt.xlabel('Mean of methods'); plt.ylabel('Difference (test - ref)'); plt.tight_layout()
    if out: plt.savefig(out, dpi=160)
