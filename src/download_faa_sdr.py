"""
Process FAA Service Difficulty Report (SDR) data for engine-related events.

HOW TO DOWNLOAD (one year at a time due to site row limits):
  1. Go to: https://sdrs.faa.gov → "Search All Processed Reports"
  2. Set filters:
       JASC Code: 7200   (Engine - Turbine/Turboprop)
       Difficulty Date From / To: 01/01/<year>  to  12/31/<year>
  3. Click Run Query. The site has no CSV export/download option — copy the
     results table as text and parse it into a CSV with these columns:
       control_number, operator, date, n_number, aircraft_make,
       aircraft_model, jasc_code
  4. Save each file as:
       data/raw/faa_sdr_engine_2020.csv
       data/raw/faa_sdr_engine_2021.csv
       data/raw/faa_sdr_engine_2022.csv
       data/raw/faa_sdr_engine_2023.csv
       data/raw/faa_sdr_engine_2024.csv

Once any year files are saved, run this script to combine and clean them:
  python src/download_faa_sdr.py
"""

import sys
from pathlib import Path

import pandas as pd

RAW_DIR  = Path(__file__).resolve().parents[1] / "data" / "raw"
PROC_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
PROC_DIR.mkdir(parents=True, exist_ok=True)

SDR_OUT = PROC_DIR / "faa_sdr_engine_clean.parquet"
YEARS   = range(2020, 2025)


def load_year_files() -> pd.DataFrame:
    frames = []
    for year in YEARS:
        path = RAW_DIR / f"faa_sdr_engine_{year}.csv"
        if path.exists():
            df = pd.read_csv(path, encoding="latin-1", low_memory=False)
            df["source_year"] = year
            frames.append(df)
            print(f"  Loaded {year}: {len(df):,} rows")
        else:
            print(f"  Missing {year} — skipping (save as {path.name} to include)")
    if not frames:
        sys.exit("No SDR year files found. See instructions at top of this script.")
    return pd.concat(frames, ignore_index=True)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    # Normalise column names
    df.columns = df.columns.str.strip().str.upper().str.replace(r"\s+", "_", regex=True)

    rename = {
        "DIFFICULTY_DATE":  "date",
        "DATE":             "date",
        "N-NUMBER":         "n_number",
        "N_NUMBER":         "n_number",
        "AIRCRAFT_MAKE":    "aircraft_make",
        "AIRCRAFT_MODEL":   "aircraft_model",
        "JASC_CODE":        "jasc_code",
        "OPERATOR\nDESIGNATOR": "operator",
        "OPERATOR_DESIGNATOR":  "operator",
        "OPERATOR":         "operator",
        "UNIQUE_CONTROL_#": "control_number",
        "UNIQUE_CONTROL_NUMBER": "control_number",
        "CONTROL_NUMBER":   "control_number",
        "SOURCE_YEAR":      "source_year",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"])
        df["year"] = df["date"].dt.year

    if "jasc_code" in df.columns:
        df["jasc_code"] = pd.to_numeric(df["jasc_code"], errors="coerce")
        df = df.dropna(subset=["jasc_code"])
        df["ata_chapter"] = (df["jasc_code"] // 100).astype(int)

    # Drop obviously non-air-carrier records (no operator designator)
    if "operator" in df.columns:
        df = df[df["operator"].notna() & (df["operator"].str.strip() != "")]

    return df.drop_duplicates()


def print_summary(df: pd.DataFrame) -> None:
    print(f"\nSDR Engine (ATA 72) Events: {len(df):,} records")
    if "year" in df.columns:
        print("\nEvents by year:")
        print(df["year"].value_counts().sort_index().to_string())
    if "aircraft_make" in df.columns:
        print("\nTop 10 aircraft makes:")
        print(df["aircraft_make"].value_counts().head(10).to_string())
    if "operator" in df.columns:
        print("\nTop 10 operators:")
        print(df["operator"].value_counts().head(10).to_string())


def main() -> None:
    print("Loading SDR year files...")
    raw = load_year_files()
    print(f"Combined: {len(raw):,} rows across {raw['source_year'].nunique()} year(s)")

    df = clean(raw)
    print(f"After cleaning: {len(df):,} rows")

    df.to_parquet(SDR_OUT, index=False)
    print(f"Saved → {SDR_OUT}")

    print_summary(df)


if __name__ == "__main__":
    main()
