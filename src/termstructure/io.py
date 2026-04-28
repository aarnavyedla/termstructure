"""Query layer for processed Treasury data."""

import pandas as pd
from pathlib import Path

_PROCESSED = Path(__file__).resolve().parents[2] / "data/processed"
_BONDS_PATH = _PROCESSED / "treasury_bonds.parquet"

# loaded once per Python session; all date lookups filter this in-memory
_BONDS_DF: pd.DataFrame | None = None


def _get_bonds_df() -> pd.DataFrame:
    global _BONDS_DF
    if _BONDS_DF is None:
        _BONDS_DF = pd.read_parquet(_BONDS_PATH)
        _BONDS_DF["date"] = pd.to_datetime(_BONDS_DF["date"])
    return _BONDS_DF


def load_bonds_for_date(date: str) -> pd.DataFrame:
    """Return Fed GSW zero yields and Svensson parameters for a single date.

    Args:
        date: date string in 'YYYY-MM-DD' format, e.g. '2020-01-15'

    Returns:
        DataFrame with columns [date, beta0, beta1, beta2, beta3, tau1, tau2,
        sveny01 ... sveny30]. Empty DataFrame if the date has no data (holiday,
        weekend, or before 1961-06-14).
    """
    df = _get_bonds_df()
    mask = df["date"] == pd.Timestamp(date)
    return df[mask].reset_index(drop=True)
