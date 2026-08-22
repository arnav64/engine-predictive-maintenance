# Predictive Maintenance Framework for Aircraft Engines

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Results Dashboard](https://img.shields.io/badge/Live%20Dashboard-View%20Results-orange)](https://arnav64.github.io/engine-predictive-maintenance/dashboard.html)

**An open-source framework for predicting Remaining Useful Life (RUL) of commercial turbofan engines, cross-checked against real U.S. operational data.**

---

Predicts RUL from NASA's N-CMAPSS turbofan degradation simulation data, then validates that model's practical impact against real FAA Service Difficulty Reports and BTS carrier cancellation records. The goal is to translate simulated engine health signals into actionable, cost-aware maintenance scheduling for U.S. air carriers — particularly smaller regional operators who lack access to enterprise predictive maintenance systems.

## Getting Started

```bash
# Install dependencies
pip install -r requirements.txt

# Download NASA N-CMAPSS dataset (~15 GB)
python src/download_data.py

# Visualise sensor degradation and RUL trajectories
python src/explore_ncmapss.py

# Engineer cycle-level features from raw sensor streams
python src/feature_engineering.py

# Train XGBoost RUL model with leave-one-unit-out cross-validation
python src/train_rul.py

# Download FAA SDR data (follow printed instructions, then re-run)
python src/download_faa_sdr.py --instructions
python src/download_faa_sdr.py   # after saving the CSV

# Analyse SDR engine failure patterns
python src/analyze_sdr.py

# Download BTS data and compute carrier cancellation baseline
python src/bts_groundings.py   # defaults to 2025; --year 2024 also works

# Integrate all three datasets into an impact estimate
python src/impact_analysis.py

# Sensitivity analysis on the assumptions behind the impact estimate
python src/sensitivity_analysis.py
```

## Visualization

```bash
# Build the interactive results dashboard (generates docs/dashboard.html)
python viz/build_dashboard.py
```

`viz/build_dashboard.py` reads the CSVs in `results/` and `data/processed/features_DS02.parquet`, builds a set of Plotly charts from that real pipeline output, and writes a self-contained HTML file to `docs/dashboard.html`.

To publish: enable GitHub Pages in repo Settings → Pages → Source: main branch, `/docs` folder.

## Architecture

### RUL Model (`src/feature_engineering.py` → `src/train_rul.py`)

**Input:** N-CMAPSS HDF5 files with millions of timesteps across engine units (DS01, DS02).

**Feature engineering:** Aggregates per-flight-cycle mean and standard deviation of 14 physical sensors (temperatures, pressures, shaft speeds, fuel flow) and 14 virtual sensors, plus flight operating conditions (altitude, Mach, throttle, inlet temperature). On top of that per-cycle snapshot, `add_temporal_features()` adds per-unit trend signal — a trailing rolling mean, delta vs. 5 cycles ago, and drift vs. each unit's first recorded cycle — since RUL is fundamentally about how sensors drift over a unit's own trajectory, not any single cycle in isolation. Elapsed cycle count and flight class are included as features (both are known in advance during real operation, so neither is a leak); health state (`hs`) is excluded since it directly encodes the degradation label.

**Model:** XGBoost regressor. RUL labels are piece-wise linear capped at 125 cycles (standard in the literature). Evaluated with leave-one-unit-out cross-validation so the model is never tested on engines it saw during training.

**Metrics:** RMSE (cycles) and NASA asymmetric s-score — the s-score penalises late predictions (false confidence) more than early ones.

### FAA SDR Analysis (`src/download_faa_sdr.py` → `src/analyze_sdr.py`)

Filters the public FAA Service Difficulty Reporting System for JASC code 7200 (engine — turbine/turboprop), 2020–2024. Tabulates event counts by ATA chapter, affected aircraft model, and annual trend. Used as real-world validation that engine failures are a substantial, persistent share of maintenance events — not as an additional discount factor on the literature engine-fraction baseline (see Impact Analysis below for why).

### BTS Groundings Baseline (`src/bts_groundings.py`)

Downloads BTS on-time performance records and filters for carrier-caused cancellations (`CancellationCode = 'A'`), the standard proxy for mechanical/maintenance groundings. Computes annual and monthly cancellation rates per carrier.

### Impact Analysis (`src/impact_analysis.py`)

Combines:
1. **RUL detection rate** — derived from CV RMSE: if mean error is small relative to the 30-cycle warning window, what fraction of failures would have been flagged in time?
2. **Engine fraction** — share of carrier cancellations attributable to engine/mechanical issues. Uses the literature baseline (~18%, BTS Air Carrier cause category / NAS delay-attribution studies) directly, unadjusted. An earlier version of this pipeline further multiplied this by an SDR-derived "unplanned fraction," but that number was a recurrence-timing proxy with no ground-truth label behind it, and the 18% baseline already represents mechanical/engine cancellations specifically — which are effectively unplanned by construction (a scheduled maintenance finding gets an aircraft swap, not a cancellation). Applying an extra "unplanned" discount was double-counting the same thing twice, so it was removed.
3. **BTS baseline** — total annual carrier cancellations per operator.

Outputs per-carrier estimates of avoidable cancellations and the projected system-wide reduction in unplanned groundings. `src/sensitivity_analysis.py` shows how this estimate moves across a defensible range of engine-fraction and warning-window assumptions. Current results are on the [live dashboard](https://arnav64.github.io/engine-predictive-maintenance/dashboard.html), not reproduced here, since they change as the pipeline is rerun on new data.

## Repository Structure

```
engine-predictive-maintenance/
├── data/
│   ├── raw/          # Downloaded source files (gitignored)
│   └── processed/    # Feature parquets and cleaned SDR (gitignored)
├── notebooks/        # Exploratory notebooks
├── src/              # Pipeline scripts
│   ├── download_data.py       # NASA N-CMAPSS downloader
│   ├── explore_ncmapss.py     # EDA and sensor visualisation
│   ├── feature_engineering.py # Cycle-level feature aggregation
│   ├── train_rul.py           # XGBoost RUL model + CV
│   ├── download_faa_sdr.py    # FAA SDR download helper
│   ├── analyze_sdr.py         # SDR failure pattern analysis
│   ├── bts_groundings.py      # BTS carrier cancellation baseline
│   ├── impact_analysis.py     # End-to-end impact estimate
│   └── sensitivity_analysis.py # Assumption sweep + methodology checks
├── viz/               # Interactive results dashboard builder
├── results/           # Plots and CSVs (charts committed, large data gitignored)
├── requirements.txt
└── README.md
```

## Key Domain Notes

- **N-CMAPSS** simulates a CFM56-class turbofan under real flight profiles. DS01 has one failure mode (HPT efficiency degradation); DS02 adds a second (LPT).
- **RUL cap at 125 cycles** is the standard piece-wise linear convention — early in engine life the exact RUL is not meaningful for scheduling, so we cap it.
- **Carrier cancellations as a maintenance proxy** is an upper bound; not all carrier cancellations are engine-related. The literature engine fraction (~18%) adjusts for this — see Impact Analysis above for why this is used unadjusted rather than further discounted by SDR recurrence timing.
- **Warning window of 30 cycles** corresponds to roughly 30 revenue flights — enough lead time to schedule planned maintenance at a maintenance station without disrupting operations.
