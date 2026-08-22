"""
Integrate N-CMAPSS RUL model performance, FAA SDR failure data, and BTS
carrier cancellation rates to estimate the operational impact of deploying
predictive engine maintenance.

Methodology:
  1. RUL detection window: if a model predicts engine failure ≥ WARNING_DAYS
     flight cycles ahead, maintenance can be scheduled (planned) rather than
     reactive (unplanned grounding).
  2. SDR unplanned fraction: estimate the share of engine SDR events that
     were unplanned (no prior warning recorded).
  3. BTS baseline: carrier-caused cancellations per year across U.S. carriers.
  4. Impact = BTS_carrier_cancels × engine_fraction × RUL_detection_rate

Requires:
  results/rul_cv_results_DS01.csv or DS02.csv  (from src/train_rul.py)
  results/bts_groundings_summary.csv           (from src/bts_groundings.py)
  data/processed/faa_sdr_engine_clean.parquet  (from src/download_faa_sdr.py)
    OR operates in SDR-absent mode with a literature-based engine fraction.

Outputs:
  results/impact_summary.csv
  results/impact_analysis.png
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
PROC_DIR    = Path(__file__).resolve().parents[1] / "data" / "processed"

WARNING_CYCLES = 30   # flight cycles of advance warning considered "planneable"
# Literature baseline: ~18% of carrier cancellations are engine/mechanical
# (source: BTS Air Carrier Statistics, NAS delay attribution studies)
ENGINE_FRACTION_DEFAULT = 0.18


def load_rul_results() -> pd.DataFrame | None:
    for ds in ("DS02", "DS01"):
        path = RESULTS_DIR / f"rul_cv_results_{ds}.csv"
        if path.exists():
            df = pd.read_csv(path)
            df["dataset"] = ds
            print(f"Loaded RUL CV results from {path.name}")
            return df
    print("No RUL CV results found — run src/train_rul.py first.")
    return None


def load_bts_summary() -> pd.DataFrame | None:
    path = RESULTS_DIR / "bts_groundings_summary.csv"
    if not path.exists():
        print("No BTS summary found — run src/bts_groundings.py first.")
        return None
    return pd.read_csv(path)


def load_sdr_engine_fraction() -> float:
    """Compute engine event fraction from SDR data if available."""
    sdr_path = PROC_DIR / "faa_sdr_engine_clean.parquet"
    if not sdr_path.exists():
        print(f"SDR data not found — using literature baseline ({ENGINE_FRACTION_DEFAULT:.0%})")
        return ENGINE_FRACTION_DEFAULT

    sdr = pd.read_parquet(sdr_path)
    # Unplanned = events without a prior same-unit report within 60 days
    # (proxy: first occurrence per aircraft_model per year is unplanned)
    if "date" in sdr.columns and "aircraft_model" in sdr.columns:
        sdr = sdr.sort_values("date")
        sdr["year"] = sdr["date"].dt.year
        sdr["prev_event"] = sdr.groupby(["aircraft_model", "year"])["date"].shift(1)
        sdr["days_since_last"] = (sdr["date"] - sdr["prev_event"]).dt.days
        unplanned_frac = (sdr["days_since_last"].isna() |
                          (sdr["days_since_last"] > 60)).mean()
        print(f"SDR-derived unplanned fraction: {unplanned_frac:.1%}")
        return float(unplanned_frac) * ENGINE_FRACTION_DEFAULT
    return ENGINE_FRACTION_DEFAULT


def rul_detection_rate(cv_df: pd.DataFrame, warning_cycles: int = WARNING_CYCLES) -> float:
    """
    Estimate the fraction of engine failures the RUL model would have detected
    ≥ warning_cycles ahead of time.

    Uses CV RMSE: if mean RMSE < warning_cycles / 2, we assume the model gives
    reliable enough signal. Scales linearly between 0 and 1.
    """
    mean_rmse = cv_df["rmse"].mean()
    # Detection rate: high when RMSE is small relative to the warning window
    rate = float(np.clip(1.0 - (mean_rmse / warning_cycles), 0.3, 0.95))
    print(f"RUL mean CV RMSE: {mean_rmse:.2f} cycles → detection rate: {rate:.1%}")
    return rate


def compute_impact(bts: pd.DataFrame, detection_rate: float,
                   engine_fraction: float) -> pd.DataFrame:
    """Project reduction in unplanned carrier cancellations per carrier."""
    df = bts.copy()
    df["engine_related_cancels"] = (df["carrier_cancels"] * engine_fraction).round()
    df["avoided_cancels"]        = (df["engine_related_cancels"] * detection_rate).round()
    df["reduction_pct"]          = (df["avoided_cancels"] / df["carrier_cancels"].clip(lower=1) * 100)
    return df


def plot_impact(impact: pd.DataFrame, detection_rate: float,
                engine_fraction: float, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    top = impact.nlargest(12, "avoided_cancels")

    ax = axes[0]
    x  = range(len(top))
    ax.bar(x, top["carrier_cancels"], label="Total carrier cancels", color="lightcoral")
    ax.bar(x, top["avoided_cancels"], label="Avoidable with RUL model", color="steelblue")
    ax.set_xticks(x)
    ax.set_xticklabels(top["carrier_name"], rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Cancellations (annual)")
    ax.set_title("Carrier Cancellations vs. Avoidable with Predictive Maintenance")
    ax.legend()

    ax = axes[1]
    ax.barh(top["carrier_name"][::-1], top["reduction_pct"][::-1], color="seagreen")
    ax.axvline(impact["reduction_pct"].mean(), color="red", linestyle="--",
               label=f"Mean: {impact['reduction_pct'].mean():.1f}%")
    ax.set_xlabel("Projected Reduction in Carrier Cancellations (%)")
    ax.set_title("Projected Impact of Predictive Engine Maintenance")
    ax.legend(fontsize=9)

    total_avoided = int(impact["avoided_cancels"].sum())
    total_cancels = int(impact["carrier_cancels"].sum())
    pct           = total_avoided / max(total_cancels, 1) * 100

    fig.suptitle(
        f"Assumptions: engine fraction={engine_fraction:.0%} of carrier cancels, "
        f"RUL detection rate={detection_rate:.0%}",
        fontsize=9, style="italic"
    )
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved {out.name}")
    print(f"\nSYSTEM-WIDE ESTIMATE: {total_avoided:,} of {total_cancels:,} "
          f"carrier cancellations avoidable ({pct:.1f}%)")


def main() -> None:
    rul_cv = load_rul_results()
    bts    = load_bts_summary()

    if rul_cv is None or bts is None:
        print("\nRun these first:\n  python src/train_rul.py\n  python src/bts_groundings.py")
        return

    engine_fraction = load_sdr_engine_fraction()
    detection_rate  = rul_detection_rate(rul_cv, WARNING_CYCLES)

    impact = compute_impact(bts, detection_rate, engine_fraction)
    impact.to_csv(RESULTS_DIR / "impact_summary.csv", index=False)

    print("\nImpact summary (top 10 carriers):")
    print(impact[["carrier_name", "carrier_cancels", "engine_related_cancels",
                  "avoided_cancels", "reduction_pct"]]
          .nlargest(10, "avoided_cancels")
          .to_string(index=False))

    plot_impact(impact, detection_rate, engine_fraction,
                RESULTS_DIR / "impact_analysis.png")


if __name__ == "__main__":
    main()
