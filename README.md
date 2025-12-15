# pwv-study — Robust PWV Pipelines on Synthetic & Public Data

**Goal.** This repository provides a research-grade, NDA-safe implementation and comparison of
three pulse wave velocity processing pipelines on PPG-like signals:

- **SegmentalProcessing (RTP-like)** — windowed preprocessing and fiducial extraction.
- **TransitTimeMethod (TT-like)** — peak-to-peak / derivative-based transit time estimation.
- **CompositePWV (GP-like)** — integrated pipeline combining multiple fiducial strategies.

It is designed to demonstrate your ability to structure research code, ensure reproducibility,
and communicate results as a small paper. All experiments run on **synthetic data** by default.

> ⚠️ No confidential work data is included. Function and variable names are *original* to this repo.

---

## Quickstart

```bash
# Create env
python -m venv .venv && source .venv/bin/activate         # Windows: .venv\Scripts\activate
pip install -e .

# Run tests
pytest -q

# Generate synthetic dataset and run the study (CSV + figures in results/)
pwv-study gen --seconds 30 --fs 1000 --hr 70 --seed 42 --out results/demo
pwv-study run --in_csv results/demo.csv --out_dir results/study_demo
```

### What you get
- **Clean library** in `src/pwvstudy/` with docstrings and type hints.
- **CLI** (`pwv-study`) to generate synthetic signals and run pipelines end-to-end.
- **Notebook examples** in `notebooks/`.
- **Tests** (`pytest`) and **CI** (GitHub Actions) for reliability.
- **Paper-style documentation** in `docs/methodology.md` with figures.

---

## Repo layout

```
pwv-study/
  pyproject.toml
  src/pwvstudy/
    synthetic.py        # synthetic waveform generator (self-contained)
    preprocess.py       # filters, detrend, trimming
    fiducials.py        # derivative-based fiducial detection with gating
    metrics.py          # transit time & PWV estimation
    pipelines.py        # SegmentalProcessing, TransitTimeMethod, CompositePWV
    validation.py       # bootstrap CI, Bland–Altman
    plots.py            # publication figures
    cli.py              # Typer CLI
  tests/
    test_fiducials.py
    test_metrics.py
  docs/
    methodology.md
    figures/            # will be populated by running notebooks/CLI
  notebooks/
    01_quick_demo.ipynb
    02_compare_pipelines.ipynb
  .github/workflows/ci.yml
  LICENSE
  CITATION.cff
```

---

## How to cite
See `CITATION.cff`.

## License
MIT (see `LICENSE`).

## Notes
- This work uses synthetic signals by default. You **can** plug in public datasets
  (e.g., PhysioNet) via `--in_csv` using the expected column names: `t, ppg`.
- All function names and API are written *from scratch* for this open project.
