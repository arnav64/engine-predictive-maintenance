"""
Download BTS on-time performance data and compute carrier-caused cancellation
rates as a proxy for unplanned maintenance groundings.

BTS Cancellation Codes:
  A = Carrier (includes mechanical/maintenance)
  B = Weather
  C = National Air System
  D = Security

Usage:
  python src/bts_groundings.py              # downloads and analyses 2025 full year
  python src/bts_groundings.py --year 2024  # use a different year

Outputs (saved to results/):
  bts_carrier_cancellations.csv  — cancellation rates by carrier and month
  bts_groundings_summary.csv     — annual summary per carrier
  bts_groundings.png             — visualisation
"""

import argparse
import zipfile
from io import BytesIO
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import requests
from tqdm import tqdm

RAW_DIR     = Path(__file__).resolve().parents[1] / "data" / "raw" / "bts"
PROC_DIR    = Path(__file__).resolve().parents[1] / "data" / "processed"
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"

for d in (RAW_DIR, PROC_DIR, RESULTS_DIR):
    d.mkdir(parents=True, exist_ok=True)

BTS_URL = (
    "https://transtats.bts.gov/PREZIP/"
    "On_Time_Reporting_Carrier_On_Time_Performance_1987_present_{year}_{month}.zip"
)

BTS_COLS = [
    "Year", "Month", "Reporting_Airline", "Tail_Number",
    "Origin", "Dest", "Cancelled", "CancellationCode",
    "CarrierDelay", "WeatherDelay", "NASDelay", "SecurityDelay",
    "LateAircraftDelay", "AirTime",
]

CARRIER_NAMES = {
    "AA": "American", "DL": "Delta", "UA": "United", "WN": "Southwest",
    "AS": "Alaska", "B6": "JetBlue", "NK": "Spirit", "F9": "Frontier",
    "G4": "Allegiant", "HA": "Hawaiian", "SY": "Sun Country",
}


def download_month(year: int, month: int) -> pd.DataFrame | None:
    url  = BTS_URL.format(year=year, month=month)
    dest = RAW_DIR / f"On_Time_{year}_{month}.zip"

    if not dest.exists():
        print(f"Downloading {year}-{month:02d} ...")
        try:
            r = requests.get(url, timeout=120, stream=True)
            r.raise_for_status()
        except requests.HTTPError:
            print(f"  Not available: {url}")
            return None
        total = int(r.headers.get("content-length", 0))
        buf = BytesIO()
        with tqdm(total=total, unit="B", unit_scale=True, desc=dest.name) as bar:
            for chunk in r.iter_content(8192):
                buf.write(chunk)
                bar.update(len(chunk))
        dest.write_bytes(buf.getvalue())

    try:
        with zipfile.ZipFile(dest) as zf:
            csv_name = next(n for n in zf.namelist() if n.endswith(".csv"))
            with zf.open(csv_name) as fh:
                usecols = [c for c in BTS_COLS if True]  # read all, filter after
                df = pd.read_csv(fh, encoding="latin-1", low_memory=False)
                available = [c for c in BTS_COLS if c in df.columns]
                return df[available]
    except Exception as exc:
        print(f"  Error reading {dest.name}: {exc}")
        return None


def compute_groundings(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate monthly cancellation rates by carrier."""
    df = df.copy()
    df["Cancelled"] = pd.to_numeric(df["Cancelled"], errors="coerce").fillna(0)
    df["is_carrier_cancel"] = (df["Cancelled"] == 1) & (df["CancellationCode"] == "A")

    monthly = (
        df.groupby(["Year", "Month", "Reporting_Airline"])
        .agg(
            total_flights=("Cancelled", "count"),
            carrier_cancels=("is_carrier_cancel", "sum"),
            total_cancels=("Cancelled", "sum"),
        )
        .reset_index()
    )
    monthly["carrier_cancel_rate"] = monthly["carrier_cancels"] / monthly["total_flights"]
    monthly["carrier_name"] = monthly["Reporting_Airline"].map(CARRIER_NAMES).fillna(
        monthly["Reporting_Airline"]
    )
    return monthly


def summarise(monthly: pd.DataFrame) -> pd.DataFrame:
    """Annual summary per carrier."""
    return (
        monthly.groupby("Reporting_Airline")
        .agg(
            carrier_name=("carrier_name", "first"),
            total_flights=("total_flights", "sum"),
            carrier_cancels=("carrier_cancels", "sum"),
        )
        .assign(carrier_cancel_rate=lambda d: d["carrier_cancels"] / d["total_flights"])
        .sort_values("carrier_cancel_rate", ascending=False)
        .reset_index()
    )


def plot_groundings(summary: pd.DataFrame, monthly: pd.DataFrame, year: int,
                    out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Bar: overall carrier cancellation rate
    top = summary.head(15)
    ax = axes[0]
    ax.barh(top["carrier_name"], top["carrier_cancel_rate"] * 100, color="firebrick")
    ax.set_xlabel("Carrier Cancellation Rate (%)")
    ax.set_title(f"{year} — Carrier-Caused Cancellation Rate\n(proxy for unplanned maintenance)")
    ax.invert_yaxis()

    # Line: monthly trend for top 5 carriers
    ax = axes[1]
    top5 = summary.head(5)["Reporting_Airline"].tolist()
    for carrier in top5:
        sub = monthly[monthly["Reporting_Airline"] == carrier].sort_values("Month")
        name = CARRIER_NAMES.get(carrier, carrier)
        ax.plot(sub["Month"], sub["carrier_cancel_rate"] * 100,
                marker="o", linewidth=1.5, markersize=4, label=name)
    ax.set_xlabel("Month")
    ax.set_ylabel("Carrier Cancellation Rate (%)")
    ax.set_title(f"{year} — Monthly trend (top 5 by cancel rate)")
    ax.legend(fontsize=8)
    ax.set_xticks(range(1, 13))

    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved {out.name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, default=2025)
    args = parser.parse_args()
    year = args.year

    frames = []
    for month in range(1, 13):
        df = download_month(year, month)
        if df is not None:
            frames.append(df)

    if not frames:
        print(f"No BTS data found for {year}. Try --year 2024.")
        return

    all_data = pd.concat(frames, ignore_index=True)
    print(f"\nLoaded {len(all_data):,} flights for {year}")

    monthly = compute_groundings(all_data)
    summary = summarise(monthly)

    monthly.to_csv(RESULTS_DIR / "bts_carrier_cancellations.csv", index=False)
    summary.to_csv(RESULTS_DIR / "bts_groundings_summary.csv", index=False)

    all_data.to_parquet(PROC_DIR / f"bts_ontime_{year}.parquet", index=False)

    print("\nTop 10 carriers by carrier cancellation rate:")
    print(summary[["carrier_name", "total_flights", "carrier_cancels",
                   "carrier_cancel_rate"]].head(10).to_string(index=False))

    plot_groundings(summary, monthly, year,
                    RESULTS_DIR / "bts_groundings.png")


if __name__ == "__main__":
    main()
