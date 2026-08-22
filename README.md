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

# Engineer cycle-level features and train the RUL model
python src/feature_engineering.py
python src/train_rul.py

# Download FAA SDR data (follow printed instructions, then re-run) and analyse it
python src/download_faa_sdr.py --instructions
python src/download_faa_sdr.py
python src/analyze_sdr.py

# Compute the BTS carrier cancellation baseline
python src/bts_groundings.py

# Integrate everything into an impact estimate + sensitivity analysis
python src/impact_analysis.py
python src/sensitivity_analysis.py
```

## Visualization

```bash
# Build the interactive results dashboard (generates docs/dashboard.html)
python viz/build_dashboard.py
```

Reads the CSVs in `results/` and `data/processed/features_DS02.parquet`, and writes a self-contained Plotly dashboard to `docs/dashboard.html`. To publish: enable GitHub Pages in repo Settings → Pages → Source: main branch, `/docs` folder.

## Architecture

### Pipeline (`feature_engineering.py` → `train_rul.py` → `impact_analysis.py`)

- **`feature_engineering.py`** — aggregates N-CMAPSS sensor streams to per-cycle features, plus per-unit trend signal (rolling mean, short-horizon delta, drift from first cycle).
- **`train_rul.py`** — XGBoost RUL regressor, leave-one-unit-out cross-validated, evaluated by RMSE and the NASA asymmetric s-score.
- **`download_faa_sdr.py` / `analyze_sdr.py`** — combines manually-pulled FAA Service Difficulty Reports (JASC 7200, 2020–2024) and tabulates failure patterns, used as real-world validation of engine-failure prevalence.
- **`bts_groundings.py`** — BTS carrier-caused cancellation rates by airline and month, the maintenance-grounding proxy.
- **`impact_analysis.py`** — combines RUL detection rate, a literature engine-fraction baseline (~18%), and the BTS baseline into a per-carrier avoidable-cancellation estimate. `sensitivity_analysis.py` shows how that estimate moves across the underlying assumptions.

Current results are on the [live dashboard](https://arnav64.github.io/engine-predictive-maintenance/dashboard.html), not reproduced here since they change as the pipeline is rerun.

### Key Domain Constants

| Constant | Value | Notes |
|---|---|---|
| RUL cap | 125 cycles | Standard piece-wise linear convention in the literature |
| Warning window | 30 cycles | ~30 revenue flights of lead time to schedule planned maintenance |
| Engine fraction | ~18% | Literature baseline (BTS Air Carrier cause category), used unadjusted |
| FAA SDR scope | JASC 7200, 2020–2024 | Engine — turbine/turboprop |

## Repository Structure

- `src/` – Pipeline scripts (data download, feature engineering, RUL model, SDR/BTS analysis, impact estimate).
- `viz/` – Results dashboard builder.
- `data/` – Raw and processed datasets (gitignored).
- `results/` – Plots and CSVs (charts committed, large data gitignored).
- `notebooks/` – Exploratory notebooks.

## Data Sources

| Source | Data | Access |
|---|---|---|
| NASA Prognostics Repository | N-CMAPSS DS01, DS02 — run-to-failure turbofan trajectories | `python src/download_data.py` |
| FAA Service Difficulty Reports | Engine SDR events (JASC 7200), 2020–2024 | Manual export — see `src/download_faa_sdr.py` |
| BTS On-Time Performance | Carrier cancellations by airline and month | `python src/bts_groundings.py` |
