# Methodology (Paper-style)

## 1. Motivation
Pulse Wave Velocity (PWV) is a key vascular marker. Robust extraction from noisy PPG-like signals
is non-trivial: derivative fiducials can shift, distal traces may exhibit double peaks, and baseline
wander can bias naive detectors.

## 2. Pipelines
- **SegmentalProcessing (RTP-like)**: lowpass → FD peaks → SD valleys within gates → TT → PWV.
- **TransitTimeMethod (TT-like)**: lowpass → FD peaks (both sites) → TT → PWV.
- **CompositePWV (GP-like)**: multiple strategies; robust median over TT estimates.

## 3. Fiducials & Gating
We use a right-open gate after FD anchors to pick SD minima, countering systematic right-shifts
and double-peak artefacts. Refractory windows avoid spurious duplicates.

## 4. Experiments
- Synthetic signals with controlled SNR, baseline drift, motion steps.
- A single-signal setting can emulate distal site with a known delay (sanity check).

## 5. Metrics & Uncertainty
- TT median, PWV = distance / TT.
- Bootstrap confidence intervals for median TT.
- Bland–Altman analysis for method comparison (optional when a reference is available).

## 6. Reproducibility
- Fixed seeds, tests, CLI, clean API. No proprietary data or code.
