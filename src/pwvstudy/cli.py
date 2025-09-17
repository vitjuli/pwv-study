import os, numpy as np, pandas as pd, typer
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

from .synthetic import synthesize_ppg_bundle
from .pipelines import (
    WindowAlignedPipeline, FeatureLagPipeline, HybridMedianPipeline, ClusterGuidedPipeline
)
from .fiducials import (
    detect_slope_rise_points, detect_curvature_valleys, gaussian_weighted_valley
)
from .preprocess import lowpass_fir

app = typer.Typer(no_args_is_help=True)

def _run_all(prox, dist, distance_m, fs):
    out=[]
    for name, pipe in [
        ("WindowAligned", WindowAlignedPipeline(fs)),
        ("FeatureLag",   FeatureLagPipeline(fs)),
        ("HybridMedian", HybridMedianPipeline(fs)),
        ("ClusterGuided",ClusterGuidedPipeline(fs)),
    ]:
        s,_ = pipe.run(prox, dist, distance_m, export_beats=False)
        s.update({"pipeline": name})
        out.append(s)
    return pd.DataFrame(out)

@app.command()
def sweep(out_dir: str = "results/sweep", fs: int = 1000, distance_m: float = 0.8):
    """Synthetic sweep across SNR and motion artefacts + CSV export."""
    os.makedirs(out_dir, exist_ok=True)
    rows=[]
    for snr in [10,15,20]:
        for steps in [0,2,4]:
            b = synthesize_ppg_bundle(seconds=12, fs=fs, hr=70, seed=snr+steps,
                                      noise_snr_db=snr, motion_steps=steps)
            prox = b['ppg']; dist = np.roll(prox, int(0.08*fs))
            df = _run_all(prox, dist, distance_m, fs)
            df['snr'] = snr; df['steps'] = steps
            rows.append(df)
    pd.concat(rows, ignore_index=True).to_csv(os.path.join(out_dir, "sweep_metrics.csv"), index=False)
    print("Saved sweep")

@app.command("figures-advanced")
def figures_advanced(out_dir: str = "docs/figures", fs: int = 1000, distance_m: float = 0.8):
    """Generate advanced figures for README."""
    os.makedirs(out_dir, exist_ok=True)
    # Build sweep
    rows=[]
    for snr in [10,15,20]:
        for steps in [0,2,4]:
            b = synthesize_ppg_bundle(seconds=12, fs=fs, hr=70, seed=snr+steps,
                                      noise_snr_db=snr, motion_steps=steps)
            prox=b['ppg']; dist=np.roll(prox, int(0.08*fs))
            for name, pipe in [
                ("WindowAligned", WindowAlignedPipeline(fs)),
                ("FeatureLag",   FeatureLagPipeline(fs)),
                ("HybridMedian", HybridMedianPipeline(fs)),
                ("ClusterGuided",ClusterGuidedPipeline(fs)),
            ]:
                s,_=pipe.run(prox,dist,distance_m,export_beats=False)
                s.update({"pipeline":name,"snr":snr,"steps":steps})
                rows.append(s)
    df = pd.DataFrame(rows)

    # Bland–Altman
    ref = df[df["pipeline"]=="WindowAligned"]["velocity"].values
    test = df[df["pipeline"]=="ClusterGuided"]["velocity"].values
    n = min(ref.size, test.size); ref, test = ref[:n], test[:n]
    mean = (ref+test)/2; diff = test-ref
    md = float(np.mean(diff)); sd = float(np.std(diff, ddof=1)) if n>1 else float('nan')
    import numpy as np
    loa_lo = md - 1.96*sd if np.isfinite(sd) else float('nan')
    loa_hi = md + 1.96*sd if np.isfinite(sd) else float('nan')
    plt.figure(figsize=(6.2,5.2))
    plt.scatter(mean, diff, alpha=0.75)
    plt.axhline(md, linestyle='--', label=f"mean Δ = {md:.3f}")
    if np.isfinite(loa_lo): plt.axhline(loa_lo, linestyle=':', label=f"LoA = {loa_lo:.3f}")
    if np.isfinite(loa_hi): plt.axhline(loa_hi, linestyle=':', label=f"LoA = {loa_hi:.3f}")
    plt.xlabel("Mean velocity (m/s)"); plt.ylabel("Δ velocity (ClusterGuided − WindowAligned) (m/s)")
    plt.title("Bland–Altman agreement"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "bland_altman_cluster_vs_window.png"), dpi=160)

    # Stability curves
    for steps in [0,2,4]:
        sub = df[(df["pipeline"]=="ClusterGuided") & (df["steps"]==steps)]
        means = sub.groupby("snr")["velocity"].mean()
        stds  = sub.groupby("snr")["velocity"].std()
        plt.figure(figsize=(6.2,4.2))
        plt.plot(means.index.values, means.values, marker='o')
        if stds.notna().any():
            plt.fill_between(means.index.values, means.values-stds.values, means.values+stds.values, alpha=0.2)
        plt.xlabel("SNR (dB)"); plt.ylabel("Velocity (m/s)")
        plt.title(f"ClusterGuided stability vs SNR (motion steps={steps})"); plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"stability_vs_snr_steps{steps}.png"), dpi=160)

    # Per-beat lag scatter (single record)
    b = synthesize_ppg_bundle(seconds=12, fs=fs, hr=70, seed=123, noise_snr_db=15, motion_steps=2)
    prox = lowpass_fir(b["ppg"], fs=fs, cutoff=10.0)
    dist = lowpass_fir(np.roll(b["ppg"], int(0.08*fs)), fs=fs, cutoff=10.0)
    fd_p = np.gradient(prox)*fs; fd_d = np.gradient(dist)*fs
    sd_p = np.gradient(fd_p)*fs; sd_d = np.gradient(fd_d)*fs
    rp = detect_slope_rise_points(fd_p, fs=fs, rel_height=0.35, min_distance_ms=300)
    rd = detect_slope_rise_points(fd_d, fs=fs, rel_height=0.35, min_distance_ms=300)
    vp0 = detect_curvature_valleys(sd_p, rp, fs=fs, gate_ms=180)
    vd0 = detect_curvature_valleys(sd_d, rd, fs=fs, gate_ms=180)
    vp = np.array([gaussian_weighted_valley(sd_p, c, fs, sigma_ms=40) for c in vp0], int)
    vd = np.array([gaussian_weighted_valley(sd_d, c, fs, sigma_ms=40) for c in vd0], int)
    lags=[]; j=0
    for p in vp:
        while j<len(vd) and vd[j]<p: j+=1
        if j>=len(vd): break
        lags.append((vd[j]-p)/fs); j+=1
    lags=np.array(lags, float)
    plt.figure(figsize=(7.5,4.2))
    plt.scatter(np.arange(lags.size), lags, alpha=0.75)
    if lags.size>0:
        med=float(np.median(lags)); plt.axhline(med, linestyle='--', label=f"median = {med:.3f}s")
        if np.unique(lags).size>=3:
            xs=np.linspace(lags.min(), lags.max(), 200)
            try:
                kde=gaussian_kde(lags); mode=float(xs[np.argmax(kde(xs))])
                plt.axhline(mode, linestyle=':', label=f"KDE mode ≈ {mode:.3f}s")
            except Exception: pass
    plt.xlabel("Beat index (paired)"); plt.ylabel("Lag (s)")
    plt.title("Per-beat lag estimates with robust summaries")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "per_beat_lag_scatter_robust.png"), dpi=160)

    # Flow diagram
    plt.figure(figsize=(9,4.8)); ax=plt.gca(); ax.axis("off")
    def box(x,y,w,h,text):
        rect=plt.Rectangle((x,y),w,h,fill=False); ax.add_patch(rect)
        plt.text(x+w/2, y+h/2, text, ha='center', va='center')
    def arrow(x1,y1,x2,y2):
        plt.annotate("", xy=(x2,y2), xytext=(x1,y1), arrowprops=dict(arrowstyle="->"))
    box(0.02,0.55,0.18,0.2,"Low-pass\nfiltering")
    box(0.25,0.55,0.18,0.2,"FD/SD\n(derivatives)")
    box(0.48,0.55,0.18,0.2,"Anchors (FD)\n+ valleys (SD)\nGaussian refine")
    box(0.71,0.55,0.25,0.2,"Beat features:\n[amp span, max slope,\nmin curvature, len]")
    box(0.02,0.15,0.28,0.2,"K-means on beats\nSelect 'stable' beats")
    box(0.35,0.15,0.28,0.2,"Pair prox/dist\nwithin gate")
    box(0.68,0.15,0.28,0.2,"Robust stats:\nMAD filter, median,\nKDE mode → velocity")
    arrow(0.20,0.65,0.25,0.65); arrow(0.43,0.65,0.48,0.65); arrow(0.66,0.65,0.71,0.65)
    arrow(0.84,0.55,0.16,0.35); arrow(0.16,0.25,0.35,0.25); arrow(0.63,0.25,0.68,0.25)
    plt.title("ClusterGuided pipeline (overview)"); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "pipeline_flow_cluster_guided.png"), dpi=160)