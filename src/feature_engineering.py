"""
Engineer cycle-level features from N-CMAPSS sensor streams.

For each (unit, cycle) pair, aggregates raw timestep readings into:
  - Mean and std of each of the 14 physical sensors (X_s)
  - Mean and std of each of the 14 virtual sensors (X_v)
  - Mean flight operating conditions (W: alt, Mach, TRA, T2)
  - Flight class (Fc) and health state (hs) from auxiliary data
  - True RUL label (constant within a cycle; capped at MAX_RUL=125)

Outputs (parquet, saved to data/processed/):
  features_DS01.parquet
  features_DS02.parquet
"""

from pathlib import Path

import h5py
import numpy as np
import pandas as pd

MAX_RUL = 125  # piece-wise linear cap standard in RUL literature

DATA_DIR     = Path(__file__).resolve().parents[1] / "data" / "raw" / "data_set"
PROC_DIR     = Path(__file__).resolve().parents[1] / "data" / "processed"
PROC_DIR.mkdir(parents=True, exist_ok=True)

DS_FILES = {
    "DS01": DATA_DIR / "N-CMAPSS_DS01-005.h5",
    "DS02": DATA_DIR / "N-CMAPSS_DS02-006.h5",
}

W_VAR   = ["alt", "Mach", "TRA", "T2"]
X_S_VAR = ["T24", "T30", "T48", "T50", "P15", "P2", "P21",
           "P24", "Ps30", "P40", "P50", "Nf", "Nc", "Wf"]
X_V_VAR = ["T40", "P30", "P45", "W21", "W22", "W25", "W31",
           "W32", "W48", "W50", "SmFan", "SmLPC", "SmHPC", "phi"]
A_VAR   = ["unit", "cycle", "Fc", "hs"]


def load_split(f: h5py.File, split: str) -> tuple[np.ndarray, ...]:
    return (
        np.array(f[f"W_{split}"]),
        np.array(f[f"X_s_{split}"]),
        np.array(f[f"X_v_{split}"]),
        np.array(f[f"Y_{split}"]).flatten(),
        np.array(f[f"A_{split}"]),
    )


def build_cycle_features(W, X_s, X_v, Y, A) -> pd.DataFrame:
    """Aggregate timestep data to one row per (unit, cycle)."""
    df = pd.DataFrame(A, columns=A_VAR)
    df["RUL_raw"] = Y

    for i, col in enumerate(W_VAR):
        df[f"W_{col}_mean"] = W[:, i]

    for i, col in enumerate(X_S_VAR):
        df[f"Xs_{col}"] = X_s[:, i]

    for i, col in enumerate(X_V_VAR):
        df[f"Xv_{col}"] = X_v[:, i]

    # Aggregate sensor columns to cycle level
    sensor_cols = (
        [f"W_{c}_mean" for c in W_VAR]
        + [f"Xs_{c}" for c in X_S_VAR]
        + [f"Xv_{c}" for c in X_V_VAR]
    )

    agg_dict = {col: ["mean", "std"] for col in sensor_cols}
    agg_dict["RUL_raw"] = "first"
    agg_dict["Fc"]      = "first"
    agg_dict["hs"]      = "first"

    cyc = (
        df.groupby(["unit", "cycle"])
        .agg(agg_dict)
        .reset_index()
    )

    # Flatten multi-level columns
    cyc.columns = [
        "_".join(c).strip("_") if c[1] else c[0]
        for c in cyc.columns
    ]

    cyc = cyc.rename(columns={"RUL_raw_first": "RUL", "Fc_first": "Fc", "hs_first": "hs"})
    cyc["RUL"] = cyc["RUL"].clip(upper=MAX_RUL).astype(int)
    cyc["split"] = "unknown"
    return cyc


def process_dataset(path: Path, ds_name: str) -> pd.DataFrame:
    print(f"\nProcessing {ds_name} ...")
    frames = []
    with h5py.File(path, "r") as f:
        for split in ("dev", "test"):
            W, X_s, X_v, Y, A = load_split(f, split)
            frame = build_cycle_features(W, X_s, X_v, Y, A)
            frame["split"] = split
            frames.append(frame)
            print(f"  {split}: {len(A):,} timesteps → {len(frame):,} cycle rows")

    combined = pd.concat(frames, ignore_index=True)

    # Drop std columns that are all-NaN (single-timestep cycles)
    std_cols = [c for c in combined.columns if c.endswith("_std")]
    combined[std_cols] = combined[std_cols].fillna(0.0)

    out = PROC_DIR / f"features_{ds_name}.parquet"
    combined.to_parquet(out, index=False)
    print(f"  Saved {len(combined):,} rows → {out}")
    return combined


def main() -> None:
    for ds_name, path in DS_FILES.items():
        if not path.exists():
            print(f"Missing {path.name} — run src/download_data.py first.")
            continue
        process_dataset(path, ds_name)


if __name__ == "__main__":
    main()
