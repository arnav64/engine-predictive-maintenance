"""
Analyse FAA Service Difficulty Report data for engine failure patterns.

Requires: data/processed/faa_sdr_engine_clean.parquet
  (produced by src/download_faa_sdr.py)

Outputs (saved to results/):
  sdr_events_by_ata.csv        — event counts by ATA chapter
  sdr_events_by_aircraft.csv   — event counts by aircraft model
  sdr_annual_trend.csv         — annual event counts
  sdr_failure_rates.png        — bar charts of the above
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PROC_DIR    = Path(__file__).resolve().parents[1] / "data" / "processed"
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
RESULTS_DIR.mkdir(exist_ok=True)

SDR_PATH = PROC_DIR / "faa_sdr_engine_clean.parquet"

ATA_LABELS = {
    71: "Power Plant — General",
    72: "Engine (Turbine/Turboprop)",
    73: "Engine Fuel & Control",
    74: "Ignition",
    75: "Air",
    76: "Engine Controls",
    77: "Engine Indicating",
    78: "Exhaust",
    79: "Oil",
    80: "Starting",
}


def load_sdr() -> pd.DataFrame:
    if not SDR_PATH.exists():
        raise FileNotFoundError(
            f"{SDR_PATH} not found. Run src/download_faa_sdr.py first."
        )
    return pd.read_parquet(SDR_PATH)


def events_by_ata(df: pd.DataFrame) -> pd.DataFrame:
    counts = (
        df["ata_chapter"]
        .value_counts()
        .sort_index()
        .reset_index()
        .rename(columns={"ata_chapter": "ata_chapter", "count": "events"})
    )
    counts["description"] = counts["ata_chapter"].map(ATA_LABELS)
    return counts


def events_by_aircraft(df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    if "aircraft_model" not in df.columns:
        return pd.DataFrame()
    return (
        df["aircraft_model"]
        .value_counts()
        .head(top_n)
        .reset_index()
        .rename(columns={"aircraft_model": "aircraft_model", "count": "events"})
    )


def annual_trend(df: pd.DataFrame) -> pd.DataFrame:
    if "date" not in df.columns:
        return pd.DataFrame()
    df = df.copy()
    df["year"] = df["date"].dt.year
    return df["year"].value_counts().sort_index().reset_index().rename(
        columns={"year": "year", "count": "events"}
    )


def plot_summary(by_ata: pd.DataFrame, by_aircraft: pd.DataFrame,
                 trend: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    if not by_ata.empty:
        ax = axes[0]
        labels = [f"ATA {int(r.ata_chapter)}\n{ATA_LABELS.get(int(r.ata_chapter), '')}"
                  for _, r in by_ata.iterrows()]
        ax.barh(labels, by_ata["events"], color="steelblue")
        ax.set_xlabel("SDR Events")
        ax.set_title("Events by ATA Chapter (Engine)")
        ax.invert_yaxis()

    if not by_aircraft.empty:
        ax = axes[1]
        ax.barh(by_aircraft["aircraft_model"], by_aircraft["events"], color="darkorange")
        ax.set_xlabel("SDR Events")
        ax.set_title("Top 20 Aircraft Models")
        ax.invert_yaxis()
        ax.tick_params(axis="y", labelsize=8)

    if not trend.empty:
        ax = axes[2]
        ax.bar(trend["year"], trend["events"], color="seagreen")
        ax.set_xlabel("Year")
        ax.set_ylabel("Events")
        ax.set_title("Annual SDR Engine Events")

    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved {out.name}")


def main() -> None:
    df = load_sdr()
    print(f"Loaded {len(df):,} SDR engine events")

    by_ata = events_by_ata(df)
    by_ac  = events_by_aircraft(df)
    trend  = annual_trend(df)

    by_ata.to_csv(RESULTS_DIR / "sdr_events_by_ata.csv", index=False)
    by_ac.to_csv(RESULTS_DIR / "sdr_events_by_aircraft.csv", index=False)
    trend.to_csv(RESULTS_DIR / "sdr_annual_trend.csv", index=False)

    print("\nEvents by ATA chapter:")
    print(by_ata.to_string(index=False))
    print(f"\nAnnual trend:\n{trend.to_string(index=False)}")

    plot_summary(by_ata, by_ac, trend, RESULTS_DIR / "sdr_failure_rates.png")


if __name__ == "__main__":
    main()
