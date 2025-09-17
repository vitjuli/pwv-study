# scripts/make_advanced_figures.py
# See CLI `figures-advanced` for identical logic. This script mirrors the CLI for IDE users.
from pwvstudy.cli import figures_advanced
if __name__ == "__main__":
    # defaults: out_dir="docs/figures", fs=1000, distance_m=0.8
    figures_advanced()