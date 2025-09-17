# src/pwvstudy/cli.py
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

# ... full code from previous answer ...
