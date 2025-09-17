# PWV-Study — Lag-Based Vascular Velocity Pipelines

[![CI](https://github.com/vitjuli/pwv-study/actions/workflows/ci.yml/badge.svg)](https://github.com/vitjuli/pwv-study/actions/workflows/ci.yml)

## Motivation
Pulse Wave Velocity (PWV) is a key biomarker of vascular stiffness. Robust lag-based estimation from peripheral signals is challenged by noise, motion artefacts and fiducial ambiguity. This repo implements **four pipelines** with SQI and robust statistics to obtain stable velocities on synthetic and public CSV signals.

## Methods (pipelines)
1. **WindowAligned** – FD anchors → SD valleys within gates → robust median/mode.  
2. **FeatureLag** – lags from FD anchors directly.  
3. **HybridMedian** – consensus (robust median of multiple lag definitions).  
4. **ClusterGuided** – Gaussian-refined fiducials → beat features → **K-means** pick “stable” beats → robust stats.

Quality/robustness: SNR-proxy, spectral entropy, gate-hit ratio, **MAD-filter**, **KDE-mode**.

## Install & quick start
```bash
git clone https://github.com/vitjuli/pwv-study.git
cd pwv-study
python -m venv .venv && source .venv/bin/activate
pip install -e .
pytest -q
```

Synthetic sweep:
```bash
pwv-study sweep --out_dir results/sweep --fs 1000 --distance_m 0.8
```

Advanced figures for README:
```bash
pwv-study figures-advanced --out_dir docs/figures --fs 1000 --distance_m 0.8
```

Public CSV format:
- `t,ppg` (demo: distal = shift by 80 ms), or  
- `t,ppg_prox,ppg_dist` (two-channel).


## Repo layout
```
src/pwvstudy/
  pipelines.py      # 4 pipelines
  fiducials.py      # slope-rise, curvature valleys, Gaussian refine
  clustering.py     # beat features + K-means selection
  quality.py        # SQI, MAD, KDE-mode
  metrics.py        # lags → velocity
  preprocess.py     # filtering utils
  synthetic.py      # synthetic generator
tests/
docs/figures/
```

## Citation
See `CITATION.cff`.
