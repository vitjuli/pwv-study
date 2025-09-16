import os, numpy as np, pandas as pd, typer
from .synthetic import synthesize_ppg_bundle
from .pipelines import WindowAlignedPipeline, FeatureLagPipeline, HybridMedianPipeline, ClusterGuidedPipeline
app = typer.Typer(no_args_is_help=True)
def _run_all(prox, dist, distance_m, fs):
    out=[]
    for name, pipe in [("WindowAligned",WindowAlignedPipeline(fs)),("FeatureLag",FeatureLagPipeline(fs)),("HybridMedian",HybridMedianPipeline(fs)),("ClusterGuided",ClusterGuidedPipeline(fs))]:
        s,_=pipe.run(prox,dist,distance_m,export_beats=False); s.update({"pipeline":name}); out.append(s)
    return pd.DataFrame(out)
@app.command()
def sweep(out_dir: str = "results/sweep", fs: int = 1000, distance_m: float = 0.8):
    os.makedirs(out_dir, exist_ok=True); rows=[]
    for snr in [10,15,20]:
        for steps in [0,2,4]:
            b=synthesize_ppg_bundle(seconds=12, fs=fs, hr=70, seed=snr+steps, noise_snr_db=snr, motion_steps=steps)
            prox=b['ppg']; dist=np.roll(prox,int(0.08*fs)); df=_run_all(prox,dist,distance_m,fs); df['snr']=snr; df['steps']=steps; rows.append(df)
    pd.concat(rows,ignore_index=True).to_csv(os.path.join(out_dir,"sweep_metrics.csv"), index=False); print("Saved sweep")
