"""
Download FAA Service Difficulty Report (SDR) data for engine-related events.

The FAA SDRS is publicly accessible — no account required.

HOW TO DOWNLOAD:
  1. Go to: https://av-info.faa.gov/sdrx/
  2. Set these filters:
       - ATA Chapter: 71, 72, 73, 74, 75, 76, 77, 78, 79, 80  (engine systems)
       - Date Range: 01/01/2020 to 12/31/2024
       - Aircraft Category: Air Carrier
  3. Click Search, then Export to CSV.
  4. Save the file as:  data/raw/faa_sdr_engine_2020_2024.csv

Once you have saved that file, run this script to clean and validate it:
  python src/download_faa_sdr.py

Alternatively, run with --instructions to print these steps again.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

RAW_DIR  = Path(__file__).resolve().parents[1] / "data" / "raw"
PROC_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
PROC_DIR.mkdir(parents=True, exist_ok=True)

SDR_RAW = RAW_DIR / "faa_sdr_engine_2020_2024.csv"
SDR_OUT = PROC_DIR / "faa_sdr_engine_clean.parquet"

ENGINE_ATA_CHAPTERS = {71, 72, 73, 74, 75, 76, 77, 78, 79, 80}

INSTRUCTIONS = """
FAA Service Difficulty Report — Download Instructions
=====================================================
1. Open: https://av-info.faa.gov/sdrx/

2. In the search form set:
     ATA Chapter(s): 71 72 73 74 75 76 77 78 79 80
     Occurrence Date: 01/01/2020 — 12/31/2024
     Aircraft Category: Air Carrier

3. Click "Search".

4. When results load, click "Export" (top right of results table)
   and choose CSV format.

5. Save the file to:
     data/raw/faa_sdr_engine_2020_2024.csv

6. Re-run:  python src/download_faa_sdr.py
"""

EXPECTED_COLS = {
    "ACFT_MAKE_MODEL": "aircraft_model",
    "ATA_CODE":        "ata_code",
    "DIFFICULTY_DATE": "date",
    "OPERATOR":        "operator",
    "PART_MFR_NAME":   "part_manufacturer",
    "PART_NAME":       "part_name",
    "REMARKS":         "remarks",
}


def clean_sdr(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path, encoding="latin-1", low_memory=False)
    print(f"Loaded {len(raw):,} rows, columns: {list(raw.columns)[:8]} ...")

    # Normalise column names (SDRS exports vary slightly)
    raw.columns = raw.columns.str.strip().str.upper().str.replace(r"\s+", "_", regex=True)

    rename = {k: v for k, v in EXPECTED_COLS.items() if k in raw.columns}
    raw = raw.rename(columns=rename)

    # Parse ATA chapter (first two digits of code)
    if "ata_code" in raw.columns:
        raw["ata_chapter"] = (
            raw["ata_code"].astype(str).str.extract(r"(\d{2})")[0].astype(float)
        )
        engine_mask = raw["ata_chapter"].isin(ENGINE_ATA_CHAPTERS)
        raw = raw[engine_mask].copy()
        print(f"After ATA 71–80 filter: {len(raw):,} rows")

    if "date" in raw.columns:
        raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
        raw = raw.dropna(subset=["date"])

    raw.to_parquet(SDR_OUT, index=False)
    print(f"Saved cleaned SDR data → {SDR_OUT}")
    return raw


def print_summary(df: pd.DataFrame) -> None:
    print(f"\nSDR Engine Events: {len(df):,} records")
    if "date" in df.columns:
        print(f"Date range: {df['date'].min().date()} — {df['date'].max().date()}")
    if "ata_chapter" in df.columns:
        print("\nEvents by ATA chapter:")
        print(df["ata_chapter"].value_counts().sort_index().to_string())
    if "aircraft_model" in df.columns:
        print("\nTop 15 aircraft models:")
        print(df["aircraft_model"].value_counts().head(15).to_string())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--instructions", action="store_true",
                        help="Print download instructions and exit")
    args = parser.parse_args()

    if args.instructions:
        print(INSTRUCTIONS)
        return

    if not SDR_RAW.exists():
        print(INSTRUCTIONS)
        sys.exit(
            f"\nERROR: {SDR_RAW} not found.\n"
            "Follow the instructions above, then re-run this script."
        )

    df = clean_sdr(SDR_RAW)
    print_summary(df)


if __name__ == "__main__":
    main()
