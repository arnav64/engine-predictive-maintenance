"""
Sensitivity analysis + methodology critique for the impact estimate in
impact_analysis.py, requested after the headline 5.3% number came in far
below an expected 15-25% range.

Two questions this script answers:

  1. Is the "unplanned fraction" SDR proxy in impact_analysis.py actually
     measuring what it claims to measure, or is there a more defensible
     alternative given the columns we actually collected?

  2. Across a defensible range of assumptions (engine fraction from
     published mechanical/technical-cause estimates, and RUL warning-window
     length), what does the system-wide reduction estimate look like — and
     is 15-25% achievable without picking assumptions purely to hit that
     target?

Requires the same inputs as impact_analysis.py.
Outputs:
  results/sensitivity_grid.csv
  results/sensitivity_heatmap.png
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
PROC_DIR    = Path(__file__).resolve().parents[1] / "data" / "processed"

CURRENT_WARNING_CYCLES = 30
LITERATURE_BASELINE    = 0.18  # value used in impact_analysis.py


# ---------------------------------------------------------------------------
# 1. Methodology critique: is the "unplanned fraction" proxy meaningful?
# ---------------------------------------------------------------------------

def unplanned_fraction_by_model_year(sdr: pd.DataFrame, gap_days: int = 60) -> float:
    """
    CURRENT method (impact_analysis.py::load_sdr_engine_fraction).
    Groups by (aircraft_model, year) and flags an event "unplanned" if no
    same-model event occurred in the prior `gap_days`.

    Problem: aircraft_model pools every physical airframe of that model
    across every operator (e.g. every 737-823 in the dataset, not one
    aircraft). With hundreds of engine SDRs per top model, the gap between
    "some 737-823 somewhere" having an event is almost never large — this
    measures fleet-wide reporting *frequency* for a model, not whether any
    single failure was foreseeable. It is not a meaningful planned/unplanned
    signal.
    """
    df = sdr.sort_values("date").copy()
    df["prev_event"] = df.groupby(["aircraft_model", "year"])["date"].shift(1)
    df["days_since_last"] = (df["date"] - df["prev_event"]).dt.days
    return float((df["days_since_last"].isna() | (df["days_since_last"] > gap_days)).mean())


def unplanned_fraction_by_tail(sdr: pd.DataFrame, gap_days: int = 60) -> float:
    """
    REVISED method: group by (operator, n_number) — i.e. the actual physical
    tail number — instead of aircraft model. A repeat engine SDR on the SAME
    airframe within gap_days is a more physically grounded signal of a
    recurring/escalating issue (arguably something predictive maintenance
    could have flagged); a fresh SDR on a tail with no recent history is
    closer to a genuinely new, unplanned failure.

    Rows with no N-number (15 of 1,533) are dropped rather than assumed
    unplanned, since we have no basis for that assumption.
    """
    df = sdr.dropna(subset=["n_number"]).copy()
    df = df[df["n_number"].str.strip() != ""]
    df = df.sort_values("date")
    df["prev_event"] = df.groupby(["operator", "n_number"])["date"].shift(1)
    df["days_since_last"] = (df["date"] - df["prev_event"]).dt.days
    return float((df["days_since_last"].isna() | (df["days_since_last"] > gap_days)).mean())


def print_methodology_comparison(sdr: pd.DataFrame) -> None:
    current = unplanned_fraction_by_model_year(sdr)
    revised = unplanned_fraction_by_tail(sdr)

    print("=" * 72)
    print("METHODOLOGY CHECK: 'unplanned fraction' proxy")
    print("=" * 72)
    print(f"Current  (group by aircraft_model, year): {current:.1%}")
    print(f"Revised  (group by operator + tail number): {revised:.1%}")
    print()
    print("Neither of these is a real 'planned vs. unplanned' label — the SDR")
    print("query we ran only pulled control#, operator, date, N-number, make,")
    print("model, and JASC code, not a problem description or discovery method,")
    print("so there is no field in this dataset that actually distinguishes a")
    print("scheduled finding from a reactive one. Both numbers above are proxies")
    print("built on recurrence timing, not ground truth.")
    print()
    print("More importantly: the 18% literature baseline (BTS 'Air Carrier'")
    print("cause category, which bundles maintenance with crew/cleaning/baggage/")
    print("fueling per BTS's own methodology notes) plausibly ALREADY represents")
    print("mechanical cancellations that were unplanned — a cancellation doesn't")
    print("usually happen for maintenance that was scheduled in advance, since")
    print("the airline would swap the aircraft instead. Multiplying that baseline")
    print("by an extra 'unplanned fraction' on top of itself is likely double-")
    print("discounting, which is most of why the current pipeline's estimate")
    print("(~9.2% engine fraction) landed at roughly half of the 18% baseline")
    print("it started from.")
    print()


# ---------------------------------------------------------------------------
# 2. Sensitivity grid: engine fraction x RUL warning window
# ---------------------------------------------------------------------------

def load_rmse() -> float:
    for ds in ("DS02", "DS01"):
        path = RESULTS_DIR / f"rul_cv_results_{ds}.csv"
        if path.exists():
            return pd.read_csv(path)["rmse"].mean()
    raise FileNotFoundError("No RUL CV results found — run src/train_rul.py first.")


def detection_rate(rmse: float, warning_cycles: float) -> float:
    return float(np.clip(1.0 - (rmse / warning_cycles), 0.3, 0.95))


def build_grid(rmse: float) -> pd.DataFrame:
    # Engine-fraction candidates, each with an explicit source/caveat:
    engine_fractions = {
        0.10: "conservative low end",
        0.18: "literature baseline used in impact_analysis.py (BTS NAS delay-cause studies)",
        0.22: "midpoint stretch",
        0.267: "SkyRefund 2025 'technical issues' share of disruption claims "
               "(different denominator: all disruptions, not carrier-caused cancels only)",
        0.30: "upper stretch bound",
    }
    warning_cycles = [10, 15, 20, 25, CURRENT_WARNING_CYCLES, 40, 50, 60, 75, 90]

    rows = []
    for ef, note in engine_fractions.items():
        for wc in warning_cycles:
            dr = detection_rate(rmse, wc)
            rows.append({
                "engine_fraction": ef,
                "engine_fraction_note": note,
                "warning_cycles": wc,
                "detection_rate": dr,
                "system_wide_reduction_pct": ef * dr * 100,
            })
    return pd.DataFrame(rows)


def plot_heatmap(grid: pd.DataFrame, out: Path) -> None:
    pivot = grid.pivot(index="engine_fraction", columns="warning_cycles",
                        values="system_wide_reduction_pct")

    fig, ax = plt.subplots(figsize=(10, 5.5))
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn_r", vmin=0, vmax=30)

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"{v:.0%}" for v in pivot.index])
    ax.set_xlabel("RUL warning window assumed (flight cycles)")
    ax.set_ylabel("Engine fraction of carrier cancellations")
    ax.set_title("System-wide reduction (%) across assumption grid\n"
                  "(green band = falls in requested 15-25% range)")

    for i, ef in enumerate(pivot.index):
        for j, wc in enumerate(pivot.columns):
            val = pivot.values[i, j]
            in_band = 15 <= val <= 25
            ax.text(j, i, f"{val:.1f}", ha="center", va="center",
                     fontsize=8, fontweight="bold" if in_band else "normal",
                     color="black")

    ax.axvline(pivot.columns.get_loc(CURRENT_WARNING_CYCLES), color="blue",
               linestyle="--", linewidth=1, alpha=0.6)
    ax.text(pivot.columns.get_loc(CURRENT_WARNING_CYCLES), -0.7,
            "current\nRUL model", ha="center", fontsize=7.5, color="blue")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("System-wide reduction (%)")

    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved {out.name}")


def print_verdict(rmse: float) -> None:
    print("=" * 72)
    print("WHAT WOULD IT TAKE TO REACH 15-25%?")
    print("=" * 72)
    for target in (15, 25):
        # best case: detection_rate capped at 0.95 (very long warning window)
        ef_needed_best = target / 100 / 0.95
        # at the model's actual detection rate (WARNING_CYCLES=30)
        dr_current = detection_rate(rmse, CURRENT_WARNING_CYCLES)
        ef_needed_current = target / 100 / dr_current
        print(f"\nTo reach {target}% system-wide reduction:")
        print(f"  - Even with best-case detection (95%, requires a very long "
              f"warning window), engine fraction would need to be >= {ef_needed_best:.1%}")
        print(f"  - At the RUL model's real performance (RMSE={rmse:.1f} cycles, "
              f"WARNING_CYCLES={CURRENT_WARNING_CYCLES} -> {dr_current:.1%} detection), "
              f"engine fraction would need to be >= {ef_needed_current:.1%}")
    print()
    print("15% is reachable only near the top of the defensible engine-fraction")
    print("range (~26%, the SkyRefund reference point) AND assuming a much longer")
    print("usable warning window than the current model demonstrates. 25% would")
    print("require an engine fraction (~26-43%) that isn't supported by anything")
    print("found in a literature check — BTS doesn't publish a clean mechanical-only")
    print("percentage, and the closest third-party figure (26.6%, different")
    print("denominator) only just clears the 15% floor, not 25%.")
    print()
    print("Conclusion: 15-25% is not a defensible range for this analysis without")
    print("assumptions that aren't supported by the data or literature found so far.")


def main() -> None:
    sdr = pd.read_parquet(PROC_DIR / "faa_sdr_engine_clean.parquet")
    print_methodology_comparison(sdr)

    rmse = load_rmse()
    print(f"RUL model mean CV RMSE: {rmse:.2f} cycles\n")

    grid = build_grid(rmse)
    grid.to_csv(RESULTS_DIR / "sensitivity_grid.csv", index=False)
    print(f"Saved sensitivity_grid.csv ({len(grid)} combinations)\n")

    plot_heatmap(grid, RESULTS_DIR / "sensitivity_heatmap.png")
    print()
    print_verdict(rmse)


if __name__ == "__main__":
    main()
