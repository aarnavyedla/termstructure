"""Parse the Federal Reserve GSW (2006) Svensson curve dataset."""

import io
import requests
import pandas as pd
from pathlib import Path

URL = "https://www.federalreserve.gov/data/yield-curve-tables/feds200628.csv"

OUTPUT_PATH = Path(__file__).resolve().parents[3] / "data/processed/treasury_bonds.parquet"

SVENSSON_PARAMS = ["BETA0", "BETA1", "BETA2", "BETA3", "TAU1", "TAU2"]
ZERO_YIELD_COLS = [f"SVENY{i:02d}" for i in range(1, 31)]


def load_fed_curves() -> pd.DataFrame:
    """Download and parse the Fed GSW Svensson curve dataset.

    Returns a long-format DataFrame with columns:
        [date, maturity_years, zero_yield]
    plus a wide parquet that also includes the six Svensson parameters.

    The parquet at data/processed/treasury_bonds.parquet stores:
        date, beta0-beta3, tau1, tau2, sveny01-sveny30
    """
    print("Downloading feds200628.csv (~16MB)...", flush=True)
    with requests.get(URL, timeout=120, stream=True) as resp:
        resp.raise_for_status()
        chunks = []
        downloaded = 0
        for chunk in resp.iter_content(chunk_size=65536):
            chunks.append(chunk)
            downloaded += len(chunk)
            print(f"  {downloaded / 1_000_000:.1f} MB", end="\r", flush=True)
        raw_bytes = b"".join(chunks)
    print(f"\nDownloaded {len(raw_bytes) / 1_000_000:.1f} MB. Parsing...")

    lines = raw_bytes.decode("utf-8").splitlines()
    header_idx = next(i for i, line in enumerate(lines) if line.startswith("Date,"))
    csv_text = "\n".join(lines[header_idx:])

    df = pd.read_csv(
        io.StringIO(csv_text),
        index_col=0,
        parse_dates=True,
        na_values=["NA", "-999.99"],
    )
    df.index.name = "date"
    df.columns = df.columns.str.lower()

    params = df[["beta0", "beta1", "beta2", "beta3", "tau1", "tau2"]].copy()
    zero_cols = [f"sveny{i:02d}" for i in range(1, 31)]
    zeros = df[zero_cols].copy()

    wide = pd.concat([params, zeros], axis=1).dropna(how="all")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    wide.reset_index().to_parquet(OUTPUT_PATH, index=False)
    print(f"Saved {len(wide):,} rows to {OUTPUT_PATH}")

    return wide
