# Predictive Maintenance Framework for Aircraft Engines

An open-source framework for predicting Remaining Useful Life (RUL) of commercial turbofan engines using NASA degradation simulation data, validated against real U.S. operational data from the FAA Service Difficulty Reporting System and BTS on-time performance records. The goal is to translate simulated engine health signals into actionable, cost-aware maintenance scheduling for U.S. air carriers — particularly smaller regional operators who lack access to enterprise predictive maintenance systems.

## Datasets

| Source | Data | Access |
|---|---|---|
| NASA Prognostics Repository | N-CMAPSS DS01, DS02 — run-to-failure turbofan trajectories | `python src/download_data.py` |
| FAA Service Difficulty Reports | Engine-related SDR events (ATA 71–80), 2020–2024 | Manual export — see `src/download_faa_sdr.py` |
| BTS On-Time Performance | Carrier cancellations by airline and month | `python src/bts_groundings.py` |

## Pipeline

Run scripts in order:

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download NASA N-CMAPSS dataset (~15 GB)
python src/download_data.py

# 3. Visualise sensor degradation and RUL trajectories
python src/explore_ncmapss.py

# 4. Engineer cycle-level features from raw sensor streams
python src/feature_engineering.py

# 5. Train XGBoost RUL model with leave-one-unit-out cross-validation
python src/train_rul.py

# 6. Download FAA SDR data (follow printed instructions, then re-run)
python src/download_faa_sdr.py --instructions
python src/download_faa_sdr.py   # after saving the CSV

# 7. Analyse SDR engine failure patterns
python src/analyze_sdr.py

# 8. Download BTS data and compute carrier cancellation baseline
python src/bts_groundings.py   # defaults to 2025; --year 2024 also works

# 9. Integrate all three datasets into an impact estimate
python src/impact_analysis.py
```

## Architecture

### RUL Model (`src/feature_engineering.py` → `src/train_rul.py`)

**Input:** N-CMAPSS HDF5 files with 6.5M timesteps across 9 engine units (DS02).

**Feature engineering:** Aggregates per-flight-cycle mean and standard deviation of 14 physical sensors (temperatures, pressures, shaft speeds, fuel flow) and 14 virtual sensors. Also includes flight operating conditions (altitude, Mach, throttle, inlet temperature) and flight class.

**Model:** XGBoost regressor. RUL labels are piece-wise linear capped at 125 cycles (standard in the literature). Evaluated with leave-one-unit-out cross-validation so the model is never tested on engines it saw during training.

**Metrics:** RMSE (cycles) and NASA asymmetric s-score — the s-score penalises late predictions (false confidence) more than early ones.

### FAA SDR Analysis (`src/download_faa_sdr.py` → `src/analyze_sdr.py`)

Filters the public FAA Service Difficulty Reporting System for ATA chapters 71–80 (engine systems). Tabulates failure modes, affected aircraft models, and annual event counts. Identifies the fraction of SDR events that appear to have been unplanned (no prior warning), which feeds into the impact estimate.

### BTS Groundings Baseline (`src/bts_groundings.py`)

Downloads BTS on-time performance records and filters for carrier-caused cancellations (`CancellationCode = 'A'`), the standard proxy for mechanical/maintenance groundings. Computes annual and monthly cancellation rates per carrier.

### Impact Analysis (`src/impact_analysis.py`)

Combines:
1. **RUL detection rate** — derived from CV RMSE: if mean error is small relative to the 30-cycle warning window, what fraction of failures would have been flagged in time?
2. **Engine fraction** — share of carrier cancellations attributable to engine/mechanical issues (from SDR data, or a literature baseline of ~18%).
3. **BTS baseline** — total annual carrier cancellations per operator.

Outputs per-carrier estimates of avoidable cancellations and the projected system-wide reduction in unplanned groundings.

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
│   └── impact_analysis.py     # End-to-end impact estimate
├── results/          # Plots and CSVs (charts committed, large data gitignored)
├── requirements.txt
└── README.md
```

## Key Domain Notes

- **N-CMAPSS** simulates a CFM56-class turbofan under real flight profiles. DS01 has one failure mode (HPT efficiency degradation); DS02 adds a second (LPT).
- **RUL cap at 125 cycles** is the standard piece-wise linear convention — early in engine life the exact RUL is not meaningful for scheduling, so we cap it.
- **Carrier cancellations as a maintenance proxy** is an upper bound; not all carrier cancellations are engine-related. The SDR-derived engine fraction (~18%) adjusts for this.
- **Warning window of 30 cycles** corresponds to roughly 30 revenue flights — enough lead time to schedule planned maintenance at a maintenance station without disrupting operations.
