from __future__ import annotations
import os
import numpy as np
import pandas as pd
import typer
from .synthetic import synthesize_ppg_bundle
from .pipelines import SegmentalProcessing, TransitTimeMethod, CompositePWV

app = typer.Typer(no_args_is_help=True)

@app.command()
def gen(out: str = typer.Argument(..., help="Output CSV/PNG prefix, e.g. results/demo"),
        seconds: float = 30.0, fs: int = 1000, hr: float = 70.0, seed: int = 123):
    os.makedirs(os.path.dirname(out), exist_ok=True)
    b = synthesize_ppg_bundle(seconds=seconds, fs=fs, hr=hr, seed=seed)
    df = pd.DataFrame({"t": b['t'], "ppg": b['ppg']})
    df.to_csv(out + ".csv", index=False)
    typer.echo(f"Saved {out}.csv")


@app.command()
def run(in_csv: str = typer.Option(..., help="Input CSV with columns t,ppg_prox,ppg_dist or t,ppg (single)"),
        out_dir: str = typer.Option("results/run", help="Output directory for metrics/figures"),
        distance_m: float = typer.Option(0.8, help="Distance between proximal and distal sites (m)"),
        fs: int = typer.Option(1000, help="Sampling rate (Hz)")
        ):
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(in_csv)
    if {'ppg_prox','ppg_dist','t'}.issubset(df.columns):
        prox, dist, t = df['ppg_prox'].to_numpy(), df['ppg_dist'].to_numpy(), df['t'].to_numpy()
    elif {'ppg','t'}.issubset(df.columns):
        # If only one signal, use a delayed copy to mimic distal site (purely synthetic demo)
        t = df['t'].to_numpy()
        prox = df['ppg'].to_numpy()
        delay = int(0.08 * fs)  # 80 ms synthetic delay
        dist = np.roll(prox, delay)
    else:
        raise ValueError("Input CSV must contain t,ppg or t,ppg_prox,ppg_dist.")

    seg = SegmentalProcessing(fs=fs)
    tt = TransitTimeMethod(fs=fs)
    comp = CompositePWV(fs=fs)

    r1 = seg.run(prox, dist, distance_m=distance_m)
    r2 = tt.run(prox, dist, distance_m=distance_m)
    r3 = comp.run(prox, dist, distance_m=distance_m)

    out = os.path.join(out_dir, "metrics.csv")
    pd.DataFrame([{"pipeline":"Segmental","tt_median":r1['tt_median'],"pwv":r1['pwv']},
                  {"pipeline":"TransitTime","tt_median":r2['tt_median'],"pwv":r2['pwv']},
                  {"pipeline":"Composite","tt_median":r3['tt_median'],"pwv":r3['pwv']}]).to_csv(out, index=False)
    typer.echo(f"Saved {out}")
