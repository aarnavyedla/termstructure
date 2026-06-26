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


def compute_zscore_signal(
    window: int = 60,
    min_periods: int = 30,
) -> pd.DataFrame:
    """
    Compute rolling z-score richness/cheapness signal from demeaned residuals.

    z_t = (residual_t - rolling_mean) / rolling_std

    where rolling_mean and rolling_std are computed over the trailing `window`
    trading days. Requires at least `min_periods` observations before emitting
    a score (earlier rows are NaN).

    Signal interpretation:
        z > +2.0  →  cheap  (buy)
        z < -2.0  →  rich   (sell)
        |z| < 2.0 →  no signal
        z → 0     →  exit

    Reads:  data/processed/bond_residuals.parquet  (already demeaned, outliers dropped)
    Writes: data/processed/richness_signal.parquet

    Returns:
        Long-format DataFrame with columns:
            date, maturity, residual_bps, rolling_mean, rolling_std, z_score
    """
    residuals = pd.read_parquet(_ROOT / "data/processed/bond_residuals.parquet")
    residuals["date"] = pd.to_datetime(residuals["date"])

    pivot = (
        residuals
        .pivot(index="date", columns="maturity", values="residual_bps")
        .sort_index()
    )

    roll_mean = pivot.rolling(window, min_periods=min_periods).mean()
    roll_std  = pivot.rolling(window, min_periods=min_periods).std()
    z         = (pivot - roll_mean) / roll_std

    records: list[pd.DataFrame] = []
    for mat in pivot.columns:
        df = pd.DataFrame({
            "date":         pivot.index,
            "maturity":     mat,
            "residual_bps": pivot[mat].values,
            "rolling_mean": roll_mean[mat].values,
            "rolling_std":  roll_std[mat].values,
            "z_score":      z[mat].values,
        })
        records.append(df)

    result = pd.concat(records, ignore_index=True)
    result = result.dropna(subset=["z_score"])
    result = result.sort_values(["date", "maturity"]).reset_index(drop=True)

    out_path = _ROOT / "data/processed/richness_signal.parquet"
    result.to_parquet(out_path, index=False)

    n_long  = (result["z_score"] >  2.0).sum()
    n_short = (result["z_score"] < -2.0).sum()
    pct_signal = (n_long + n_short) / len(result)
    print(f"Saved {len(result):,} rows -> {out_path}")
    print(f"Signal at |z| > 2:  {n_long:,} long  {n_short:,} short  ({pct_signal:.1%} of all obs)")
    return result


def build_signal_panel(entry_z: float = 2.0) -> pd.DataFrame:
    """
    Build the final backtest-ready signal panel from the z-score signal.

    Adds a `direction` column encoding the trade direction at the given entry
    threshold, then saves the full panel (all dates, all maturities, including
    direction=0 rows) so Week 6 can query any date without re-implementing
    threshold logic.

    direction encoding:
        +1  z >  +entry_z   cheap → buy
        -1  z <  -entry_z   rich  → sell
         0  |z| <= entry_z  no signal

    Reads:  data/processed/richness_signal.parquet
    Writes: data/processed/signal_panel.parquet

    Returns:
        Long-format DataFrame with columns:
            date, maturity, residual_bps, rolling_mean, rolling_std,
            z_score, direction
    """
    sig = pd.read_parquet(_ROOT / "data/processed/richness_signal.parquet")
    sig["date"] = pd.to_datetime(sig["date"])

    sig["direction"] = 0
    sig.loc[sig["z_score"] >  entry_z, "direction"] = 1
    sig.loc[sig["z_score"] < -entry_z, "direction"] = -1

    sig = sig.sort_values(["date", "maturity"]).reset_index(drop=True)

    out_path = _ROOT / "data/processed/signal_panel.parquet"
    sig.to_parquet(out_path, index=False)

    n_long  = (sig["direction"] ==  1).sum()
    n_short = (sig["direction"] == -1).sum()
    n_dates = sig["date"].nunique()
    print(f"Saved {len(sig):,} rows across {n_dates:,} dates -> {out_path}")
    print(f"Entry threshold : |z| > {entry_z}")
    print(f"Long signals    : {n_long:,}  ({n_long/len(sig):.1%})")
    print(f"Short signals   : {n_short:,}  ({n_short/len(sig):.1%})")
    return sig
