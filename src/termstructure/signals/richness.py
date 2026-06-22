import numpy as np
import pandas as pd
from pathlib import Path

from termstructure.curves.svensson import svensson_zero_rate
from termstructure.pca.panel import DEFAULT_MATURITIES

_ROOT = Path(__file__).resolve().parents[3]


def compute_residuals(
    maturities: list[int] | None = None,
) -> pd.DataFrame:
    """
    Compute richness/cheapness residuals at each maturity grid point.

    residual_bps = (observed_rate - fitted_rate) * 10_000

    Positive → cheap (observed yield above our fitted Svensson curve)
    Negative → rich  (observed yield below our fitted Svensson curve)

    Args:
        maturities: integer maturities in years to evaluate. Defaults to
                    the same grid used in the zero panel: [1, 2, 3, 5, 7, 10, 20, 30].

    Returns:
        Long-format DataFrame with columns:
            date, maturity, observed_rate, fitted_rate, residual_bps
        Saved to data/processed/bond_residuals.parquet.
    """
    if maturities is None:
        maturities = DEFAULT_MATURITIES

    params = pd.read_parquet(_ROOT / "data/processed/svensson_params.parquet")
    params["date"] = pd.to_datetime(params["date"])
    params = params.set_index("date").sort_index()

    bonds = pd.read_parquet(_ROOT / "data/processed/treasury_bonds.parquet")
    bonds["date"] = pd.to_datetime(bonds["date"])
    bonds = bonds.set_index("date").sort_index()

    common_dates = params.index.intersection(bonds.index)

    param_cols = ["beta0", "beta1", "beta2", "beta3", "lambda1", "lambda2"]
    param_arr = params.loc[common_dates, param_cols].to_numpy()

    records: list[dict] = []
    for mat in maturities:
        sveny_col = f"sveny{int(mat):02d}"
        obs_pct = bonds.loc[common_dates, sveny_col].to_numpy(dtype=float)
        obs_decimal = obs_pct / 100.0

        fitted = np.array([
            svensson_zero_rate(float(mat), *row)
            for row in param_arr
        ])

        residual_bps = (obs_decimal - fitted) * 10_000

        for i, date in enumerate(common_dates):
            if np.isnan(obs_decimal[i]):
                continue
            records.append({
                "date":          date,
                "maturity":      mat,
                "observed_rate": obs_decimal[i],
                "fitted_rate":   fitted[i],
                "residual_bps":  residual_bps[i],
            })

    result = pd.DataFrame(records)
    result = result.sort_values(["date", "maturity"]).reset_index(drop=True)

    # Drop data-error outliers before demeaning so they don't bias the mean.
    n_before = len(result)
    result = result[result["residual_bps"].abs() <= 30].reset_index(drop=True)
    print(f"Outlier filter: dropped {n_before - len(result):,} rows with |residual| > 30 bp")

    # Per-maturity demeaning: remove structural Svensson fitting bias at each
    # maturity (e.g. the 30Y systematically sits below our fitted curve).
    # Store constants so Day 3 can apply the same shift to new residuals.
    means = (
        result.groupby("maturity")["residual_bps"]
        .mean()
        .rename("mean_residual_bps")
        .reset_index()
    )
    means_path = _ROOT / "data/processed/residual_means.parquet"
    means.to_parquet(means_path, index=False)
    print(f"Saved demeaning constants -> {means_path}")

    # Diagnostic: verify the subtraction is exact before applying it.
    _chk = result.head(5)[["date", "maturity", "residual_bps"]].copy()
    _chk = _chk.merge(means, on="maturity", how="left")
    _chk["after"] = _chk["residual_bps"] - _chk["mean_residual_bps"]
    _chk["diff"]  = _chk["residual_bps"] - _chk["after"]
    print("\nDemeaning check (5 rows):")
    print(_chk[["maturity", "residual_bps", "mean_residual_bps", "after", "diff"]].to_string(index=False))
    print(f"diff == mean_residual_bps: {np.allclose(_chk['diff'], _chk['mean_residual_bps'])}\n")

    # Apply: join mean constants then subtract in one explicit step.
    result = result.merge(means, on="maturity", how="left")
    result["residual_bps"] = result["residual_bps"] - result["mean_residual_bps"]
    result = result.drop(columns=["mean_residual_bps"])

    out_path = _ROOT / "data/processed/bond_residuals.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(out_path, index=False)
    print(f"Saved {len(result):,} rows -> {out_path}")
    return result
