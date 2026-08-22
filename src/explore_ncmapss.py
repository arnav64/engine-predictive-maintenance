"""
Visualise sensor degradation trends in N-CMAPSS DS01 and DS02.

Outputs (saved to results/):
  sensor_trends_engine1.png   — per-cycle mean of each X_s sensor for one unit
  health_index_all_engines.png — health state (hs) trajectories for all units
  rul_all_engines.png          — true RUL trajectories for all units
"""

from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "raw" / "data_set"
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
RESULTS_DIR.mkdir(exist_ok=True)

DS_FILES = {
    "DS01": DATA_DIR / "N-CMAPSS_DS01-005.h5",
    "DS02": DATA_DIR / "N-CMAPSS_DS02-006.h5",
}

X_S_VAR = ["T24", "T30", "T48", "T50", "P15", "P2", "P21",
           "P24", "Ps30", "P40", "P50", "Nf", "Nc", "Wf"]
A_VAR   = ["unit", "cycle", "Fc", "hs"]


def load_dataset(path: Path) -> pd.DataFrame:
    """Load an N-CMAPSS HDF5 file and return a flat DataFrame (dev + test)."""
    arrays = {}
    with h5py.File(path, "r") as f:
        for split in ("dev", "test"):
            for key in ("W", "X_s", "X_v", "Y", "A"):
                arr = np.array(f[f"{key}_{split}"])
                arrays.setdefault(key, []).append(arr)

    W   = np.concatenate(arrays["W"],   axis=0)
    X_s = np.concatenate(arrays["X_s"], axis=0)
    Y   = np.concatenate(arrays["Y"],   axis=0).flatten()
    A   = np.concatenate(arrays["A"],   axis=0)

    df = pd.DataFrame(A, columns=A_VAR)
    df["RUL"] = Y
    for i, col in enumerate(X_S_VAR):
        df[col] = X_s[:, i]
    return df


def plot_sensor_trends(df: pd.DataFrame, unit_id: float, ds_name: str, out: Path) -> None:
    """Plot per-cycle mean of each physical sensor for a single engine unit."""
    unit_df = df[df["unit"] == unit_id].copy()
    cyc = unit_df.groupby("cycle")[X_S_VAR].mean().reset_index()

    n = len(X_S_VAR)
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(16, rows * 3))
    axes = axes.flatten()

    for i, col in enumerate(X_S_VAR):
        axes[i].plot(cyc["cycle"], cyc[col], linewidth=1.5)
        axes[i].set_title(col, fontsize=10)
        axes[i].set_xlabel("Flight Cycle")
        axes[i].set_ylabel("Mean Value")

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(f"{ds_name} — Sensor trends, Engine Unit {int(unit_id)}", fontsize=13)
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved {out.name}")


def plot_health_index(df: pd.DataFrame, ds_name: str, out: Path) -> None:
    """Plot health state (hs) per cycle for every engine unit."""
    fig, ax = plt.subplots(figsize=(10, 5))
    for uid in sorted(df["unit"].unique()):
        sub = df[df["unit"] == uid].drop_duplicates("cycle").sort_values("cycle")
        ax.plot(sub["cycle"], sub["hs"], label=f"Unit {int(uid)}", linewidth=1.5)

    ax.set_xlabel("Flight Cycle")
    ax.set_ylabel("Health State (hs)")
    ax.set_title(f"{ds_name} — Health state per engine unit")
    ax.legend(fontsize=8, ncol=3)
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved {out.name}")


def plot_rul_trajectories(df: pd.DataFrame, ds_name: str, out: Path) -> None:
    """Plot true RUL trajectory per cycle for every engine unit."""
    fig, ax = plt.subplots(figsize=(10, 5))
    for uid in sorted(df["unit"].unique()):
        sub = df[df["unit"] == uid].drop_duplicates("cycle").sort_values("cycle")
        ax.plot(sub["cycle"], sub["RUL"], label=f"Unit {int(uid)}", linewidth=1.5)

    ax.set_xlabel("Flight Cycle")
    ax.set_ylabel("Remaining Useful Life (cycles)")
    ax.set_title(f"{ds_name} — True RUL per engine unit")
    ax.legend(fontsize=8, ncol=3)
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved {out.name}")


def main() -> None:
    for ds_name, path in DS_FILES.items():
        if not path.exists():
            print(f"Missing {path} — run src/download_data.py first.")
            continue

        print(f"\nLoading {ds_name} ...")
        df = load_dataset(path)
        print(f"  {len(df):,} timesteps | {df['unit'].nunique()} engine units")

        first_unit = sorted(df["unit"].unique())[0]
        plot_sensor_trends(df, first_unit, ds_name,
                           RESULTS_DIR / f"sensor_trends_{ds_name.lower()}_engine{int(first_unit)}.png")
        plot_health_index(df, ds_name,
                          RESULTS_DIR / f"health_index_{ds_name.lower()}_all_engines.png")
        plot_rul_trajectories(df, ds_name,
                              RESULTS_DIR / f"rul_{ds_name.lower()}_all_engines.png")


if __name__ == "__main__":
    main()
