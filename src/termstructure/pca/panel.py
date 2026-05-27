import numpy as np
import pandas as pd

from termstructure.curves.svensson import svensson_zero_rate

DEFAULT_MATURITIES = [0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30]


def _col(mat: float) -> str:
    """'1' for whole numbers, '0.25' for fractions."""
    return str(int(mat)) if mat == int(mat) else str(mat)


def build_zero_panel(
    maturities: list[float] | None = None,
) -> pd.DataFrame:
    """
    Build a panel of Svensson zero rates and their daily changes.

    Loads svensson_params.parquet, evaluates the fitted curve at each maturity
    for every date, then first-differences to get daily bp changes.

    Args:
        maturities: maturities in years. Defaults to
                    [0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30].

    Returns:
        DataFrame with columns:
            date
            y_{mat}  — zero rate in decimal (level), one per maturity
            dy_{mat} — daily change in bp (NaN for first date), one per maturity
        Saved to data/processed/zero_panel.parquet.
    """
    from pathlib import Path

    if maturities is None:
        maturities = DEFAULT_MATURITIES

    params_path = Path(__file__).resolve().parents[3] / "data/processed/svensson_params.parquet"
    out_path    = Path(__file__).resolve().parents[3] / "data/processed/zero_panel.parquet"

    params_df = pd.read_parquet(params_path)
    params_df["date"] = pd.to_datetime(params_df["date"])
    params_df = params_df.sort_values("date").reset_index(drop=True)

    param_cols = ["beta0", "beta1", "beta2", "beta3", "lambda1", "lambda2"]
    mats_arr   = np.array(maturities, dtype=float)
    params_arr = params_df[param_cols].to_numpy()

    # shape (n_dates, n_maturities) — evaluate all maturities at once per date
    rates = np.array([svensson_zero_rate(mats_arr, *row) for row in params_arr])

    # first differences in bp; NaN on first row
    changes = np.vstack([
        np.full((1, len(maturities)), np.nan),
        np.diff(rates, axis=0) * 10_000,
    ])

    result = pd.DataFrame({"date": params_df["date"]})
    for i, mat in enumerate(maturities):
        label = _col(mat)
        result[f"y_{label}"]  = rates[:, i]
        result[f"dy_{label}"] = changes[:, i]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(out_path, index=False)
    return result
