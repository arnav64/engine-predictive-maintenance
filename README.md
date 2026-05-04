# Predictive Maintenance Framework for Aircraft Engines Using Open Aviation Datasets

## Overview

RUL (Remaining Useful Life) prediction on N-CMAPSS turbofan simulation data serves as the foundation for a framework connecting simulated engine degradation to real-world operational costs. The goal is to move beyond benchmark accuracy metrics toward actionable maintenance decision support — translating degradation signals into cost-aware scheduling recommendations.

## Datasets

- **N-CMAPSS DS01** — NASA Commercial Modular Aero-Propulsion System Simulation, scenario 1
- **N-CMAPSS DS02** — NASA Commercial Modular Aero-Propulsion System Simulation, scenario 2

Both datasets are available from the [NASA Prognostics Data Repository](https://www.nasa.gov/content/prognostics-center-of-excellence-data-set-repository).

## Project Structure

```
engine-predictive-maintenance/
├── data/
│   ├── raw/          # Downloaded source files (gitignored)
│   └── processed/    # Engineered features and splits (gitignored)
├── notebooks/        # Exploratory analysis and visualisations
├── src/              # Pipeline scripts
├── results/          # Model outputs, plots, metrics (gitignored for png/csv)
├── requirements.txt
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
```

## Usage

Run the pipeline scripts in order:

1. **Download data**
   ```bash
   python src/download_data.py
   ```

2. **Explore N-CMAPSS**
   ```bash
   python src/explore_ncmapss.py
   ```

3. **Feature engineering**
   ```bash
   python src/feature_engineering.py
   ```

4. **Train model**
   ```bash
   python src/train_model.py
   ```
