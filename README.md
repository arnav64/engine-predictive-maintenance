# Predictive Maintenance Framework for Aircraft Engines

An open-source framework for predicting Remaining Useful Life (RUL) of commercial turbofan engines using NASA degradation simulation data, validated against real U.S. operational data from the FAA Service Difficulty Reporting System and BTS on-time performance records. The goal is to translate simulated engine health signals into actionable, cost-aware maintenance scheduling for U.S. air carriers — particularly smaller regional operators who lack access to enterprise predictive maintenance systems.

## Datasets

| Source | Data | Access |
|---|---|---|
| NASA Prognostics Repository | N-CMAPSS DS01, DS02 — run-to-failure turbofan trajectories | `python src/download_data.py` |
| FAA Service Difficulty Reports | Engine SDR events (JASC 7200 — turbine/turboprop), 2020–2024 | Manual export — see `src/download_faa_sdr.py` |
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

# 10. (Optional) Sensitivity analysis on the assumptions behind the headline number
python src/sensitivity_analysis.py
```

## Architecture

### RUL Model (`src/feature_engineering.py` → `src/train_rul.py`)

**Input:** N-CMAPSS HDF5 files with millions of timesteps across engine units (DS01, DS02).

**Feature engineering:** Aggregates per-flight-cycle mean and standard deviation of 14 physical sensors (temperatures, pressures, shaft speeds, fuel flow) and 14 virtual sensors, plus flight operating conditions (altitude, Mach, throttle, inlet temperature). On top of that per-cycle snapshot, `add_temporal_features()` adds per-unit trend signal — a trailing rolling mean, delta vs. 5 cycles ago, and drift vs. each unit's first recorded cycle — since RUL is fundamentally about how sensors drift over a unit's own trajectory, not any single cycle in isolation. Elapsed cycle count and flight class are included as features (both are known in advance during real operation, so neither is a leak); health state (`hs`) is excluded since it directly encodes the degradation label.

**Model:** XGBoost regressor. RUL labels are piece-wise linear capped at 125 cycles (standard in the literature). Evaluated with leave-one-unit-out cross-validation so the model is never tested on engines it saw during training.

**Metrics:** RMSE (cycles) and NASA asymmetric s-score — the s-score penalises late predictions (false confidence) more than early ones. Current CV RMSE: **7.78 cycles (DS01)**, **9.56 cycles (DS02)**.

### FAA SDR Analysis (`src/download_faa_sdr.py` → `src/analyze_sdr.py`)

Filters the public FAA Service Difficulty Reporting System for JASC code 7200 (engine — turbine/turboprop), 2020–2024 (1,533 cleaned events after dropping non-carrier and malformed records). Tabulates event counts by ATA chapter, affected aircraft model, and annual trend. This is used as real-world validation that engine failures are a substantial, persistent share of maintenance events — not as an additional discount factor on the literature engine-fraction baseline (see Impact Analysis below for why).

### BTS Groundings Baseline (`src/bts_groundings.py`)

Downloads BTS on-time performance records and filters for carrier-caused cancellations (`CancellationCode = 'A'`), the standard proxy for mechanical/maintenance groundings. Computes annual and monthly cancellation rates per carrier.

### Impact Analysis (`src/impact_analysis.py`)

Combines:
1. **RUL detection rate** — derived from CV RMSE: if mean error is small relative to the 30-cycle warning window, what fraction of failures would have been flagged in time?
2. **Engine fraction** — share of carrier cancellations attributable to engine/mechanical issues. Uses the literature baseline (~18%, BTS Air Carrier cause category / NAS delay-attribution studies) directly, unadjusted. An earlier version of this pipeline further multiplied this by an SDR-derived "unplanned fraction," but that number was a recurrence-timing proxy with no ground-truth label behind it, and the 18% baseline already represents mechanical/engine cancellations specifically — which are effectively unplanned by construction (a scheduled maintenance finding gets an aircraft swap, not a cancellation). Applying an extra "unplanned" discount was double-counting the same thing twice, so it was removed.
3. **BTS baseline** — total annual carrier cancellations per operator.

Outputs per-carrier estimates of avoidable cancellations and the projected system-wide reduction in unplanned groundings. Run `src/sensitivity_analysis.py` to see how this estimate moves across a defensible range of engine-fraction and warning-window assumptions.

## Results

Current end-to-end estimate (RUL model on N-CMAPSS DS02, FAA SDR 2020–2024, BTS carrier cancellations):

- **12.3% of carrier cancellations avoidable system-wide** — 2,586 of 21,073 annual carrier-caused cancellations, projected across 14 carriers.
- Driven by a 68.1% RUL detection rate (9.56-cycle CV RMSE against a 30-cycle warning window) and an 18% literature engine-fraction baseline.
- Delta has the largest absolute avoidable count (742/year); all major carriers cluster within a point or two of the 12.3% system-wide rate, since detection rate and engine fraction are applied uniformly.
- `src/sensitivity_analysis.py` shows this estimate is not especially sensitive to small assumption changes: reaching materially higher reduction rates (15%+) requires either an engine fraction above the top of the range found in a literature check (~26%, and only from a source with a different denominator than "carrier cancellations") or a RUL warning window well beyond what the model's current accuracy supports. See `results/sensitivity_grid.csv` and `results/sensitivity_heatmap.png`.

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
├── results/          # Plots and CSVs (charts committed, large data gitignored)
├── requirements.txt
└── README.md
```

## Key Domain Notes

- **N-CMAPSS** simulates a CFM56-class turbofan under real flight profiles. DS01 has one failure mode (HPT efficiency degradation); DS02 adds a second (LPT).
- **RUL cap at 125 cycles** is the standard piece-wise linear convention — early in engine life the exact RUL is not meaningful for scheduling, so we cap it.
- **Carrier cancellations as a maintenance proxy** is an upper bound; not all carrier cancellations are engine-related. The literature engine fraction (~18%) adjusts for this — see Impact Analysis above for why this is used unadjusted rather than further discounted by SDR recurrence timing.
- **Warning window of 30 cycles** corresponds to roughly 30 revenue flights — enough lead time to schedule planned maintenance at a maintenance station without disrupting operations.
