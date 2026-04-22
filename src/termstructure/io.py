"""Query layer for processed Treasury data."""

import duckdb
import pandas as pd
from pathlib import Path

_PROCESSED = Path(__file__).resolve().parents[2] / "data/processed"
_BONDS_PATH = _PROCESSED / "treasury_bonds.parquet"


def load_bonds_for_date(date: str) -> pd.DataFrame:
    """Return Fed GSW zero yields and Svensson parameters for a single date.

    Args:
        date: date string in 'YYYY-MM-DD' format, e.g. '2020-01-15'

    Returns:
        DataFrame with columns [date, beta0, beta1, beta2, beta3, tau1, tau2,
        sveny01 ... sveny30]. Empty DataFrame if the date has no data (holiday,
        weekend, or before 1961-06-14).
    """
    result = duckdb.sql(
        f"SELECT * FROM read_parquet('{_BONDS_PATH.as_posix()}') WHERE date = '{date}'"
    ).df()
    return result
