"""Fetch US Treasury constant-maturity yields from FRED."""

import io
import requests
import pandas as pd
from pathlib import Path

SERIES = {
    "DGS3MO": 0.25,
    "DGS1":   1.0,
    "DGS2":   2.0,
    "DGS5":   5.0,
    "DGS10":  10.0,
    "DGS30":  30.0,
}

OUTPUT_PATH = Path(__file__).resolve().parents[3] / "data/processed/cmt_yields.parquet"


def _fetch_one(series_id: str) -> pd.Series:
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    print(f"  Fetching {series_id}...", end=" ", flush=True)
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    df = pd.read_csv(io.StringIO(resp.text), index_col=0, parse_dates=True)
    s = pd.to_numeric(df.iloc[:, 0], errors="coerce")
    s.name = series_id
    print(f"done ({len(s):,} rows)")
    return s


def fetch_fred_yields(start: str, end: str) -> pd.DataFrame:
    """Pull CMT yields from FRED and return a long-format DataFrame.

    Args:
        start: start date string, e.g. "2000-01-01"
        end:   end date string,   e.g. "2024-12-31"

    Returns:
        DataFrame with columns [date, maturity_years, yield_pct],
        one row per (date, maturity) pair, NaNs dropped.
    """
    frames = [_fetch_one(sid).loc[start:end] for sid in SERIES]
    wide = pd.concat(frames, axis=1)
    wide.index.name = "date"

    long = wide.reset_index().melt(
        id_vars="date",
        var_name="series",
        value_name="yield_pct",
    )

    long["maturity_years"] = long["series"].map(SERIES)
    long = long.drop(columns="series")
    long = long.dropna(subset=["yield_pct"])
    long = long.sort_values(["date", "maturity_years"]).reset_index(drop=True)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    long.to_parquet(OUTPUT_PATH, index=False)
    print(f"Saved {len(long):,} rows to {OUTPUT_PATH}")

    return long
