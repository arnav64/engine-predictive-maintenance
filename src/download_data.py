"""Download and verify the N-CMAPSS turbofan engine degradation dataset from NASA/PHM."""

import sys
import zipfile
from pathlib import Path

import h5py
import requests
from tqdm import tqdm

DATASET_URL = (
    "https://phm-datasets.s3.amazonaws.com/NASA/"
    "17.+Turbofan+Engine+Degradation+Simulation+Data+Set+2.zip"
)
RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"


def download_file(url: str, dest_dir: Path) -> Path:
    """Download a file from *url* into *dest_dir*, showing a progress bar.

    Returns the path to the downloaded file.
    Raises SystemExit with a clear message on any network or HTTP error.
    Skips the download if the file already exists.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    filename = dest_dir / url.split("/")[-1]

    if filename.exists():
        print(f"Already downloaded: {filename.name} — skipping.")
        return filename

    print(f"Downloading {url}")
    try:
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()
    except requests.exceptions.ConnectionError as exc:
        sys.exit(f"ERROR: Could not connect to {url}\n  {exc}")
    except requests.exceptions.Timeout:
        sys.exit(f"ERROR: Request timed out for {url}")
    except requests.exceptions.HTTPError as exc:
        sys.exit(f"ERROR: HTTP {exc.response.status_code} when fetching {url}")

    total = int(response.headers.get("content-length", 0))
    with open(filename, "wb") as fh, tqdm(
        total=total, unit="B", unit_scale=True, unit_divisor=1024, desc=filename.name
    ) as bar:
        for chunk in response.iter_content(chunk_size=8192):
            fh.write(chunk)
            bar.update(len(chunk))

    print(f"Saved to {filename}")
    return filename


def unzip_file(zip_path: Path, extract_to: Path) -> list[Path]:
    """Extract *zip_path* into *extract_to* and return a list of extracted paths.

    Any nested .zip files discovered during extraction are also unpacked
    into *extract_to*, one level deep.
    """
    print(f"\nExtracting {zip_path.name} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_to)
        extracted = [extract_to / name for name in zf.namelist()]
    print(f"Extracted {len(extracted)} item(s) from outer archive.")

    nested_zips = [p for p in extracted if p.suffix == ".zip" and p.is_file()]
    for nz in nested_zips:
        print(f"Found nested archive: {nz.name} — extracting ...")
        with zipfile.ZipFile(nz, "r") as zf:
            zf.extractall(extract_to)
            inner = [extract_to / name for name in zf.namelist()]
        print(f"Extracted {len(inner)} item(s) from {nz.name}.")
        extracted.extend(inner)

    return extracted


def report_files(paths: list[Path]) -> None:
    """Print the name and size of every regular file in *paths*."""
    files = sorted(p for p in paths if p.is_file())
    print(f"\nExtracted files ({len(files)} total):")
    for p in files:
        size_mb = p.stat().st_size / 1_048_576
        print(f"  {p.name:<60}  {size_mb:>8.2f} MB")


def inspect_h5_files(search_root: Path) -> None:
    """Recursively find every .h5 file under *search_root*, open it with h5py,
    and print its top-level keys to confirm the archive loaded correctly.
    """
    h5_files = sorted(search_root.rglob("*.h5"))
    if not h5_files:
        print("\nNo .h5 files found.")
        return

    print(f"\nTop-level HDF5 keys ({len(h5_files)} file(s)):")
    for h5_path in h5_files:
        try:
            with h5py.File(h5_path, "r") as f:
                keys = list(f.keys())
            print(f"  {h5_path.name}: {keys}")
        except OSError as exc:
            print(f"  {h5_path.name}: WARNING — could not open ({exc})")


def main() -> None:
    """Orchestrate download, extraction, and verification of the N-CMAPSS dataset."""
    zip_path = download_file(DATASET_URL, RAW_DIR)
    extracted = unzip_file(zip_path, RAW_DIR)
    report_files(extracted)
    inspect_h5_files(RAW_DIR)
    print("\nDone.")


if __name__ == "__main__":
    main()
