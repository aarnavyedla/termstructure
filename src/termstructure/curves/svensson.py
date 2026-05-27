import numpy as np
import pandas as pd
from scipy.optimize import least_squares


def svensson_zero_rate(
    tau: float | np.ndarray,
    beta0: float,
    beta1: float,
    beta2: float,
    beta3: float,
    lambda1: float,
    lambda2: float,
) -> float | np.ndarray:
    """
    Svensson (1994) zero-coupon yield at maturity tau (in years).

    Asymptotes:
      tau -> 0:   beta0 + beta1   (short rate)
      tau -> inf: beta0            (long rate)

    Parameters are in decimal form (e.g. 0.04 for 4%).
    """
    tau = np.asarray(tau, dtype=float)
    scalar = tau.ndim == 0
    tau = np.atleast_1d(tau)

    # Loading factors — (1 - exp(-t/λ)) / (t/λ)
    # At tau=0 the limit is 1.0; use np.where to avoid division by zero.
    def loading(t, lam):
        x = t / lam
        return np.where(x == 0.0, 1.0, (1.0 - np.exp(-x)) / x)

    L1 = loading(tau, lambda1)
    L2 = loading(tau, lambda2)

    result = (
        beta0
        + beta1 * L1
        + beta2 * (L1 - np.exp(-tau / lambda1))
        + beta3 * (L2 - np.exp(-tau / lambda2))
    )

    return float(result[0]) if scalar else result


def fit_svensson(
    maturities: np.ndarray,
    yields: np.ndarray,
    x0: np.ndarray | None = None,
) -> np.ndarray:
    """
    Fit Svensson parameters to observed zero yields via nonlinear least squares.

    Args:
        maturities: array of maturities in years, shape (N,)
        yields:     observed zero yields, same units as desired output, shape (N,)
        x0:         optional starting guess [b0,b1,b2,b3,l1,l2]; if None, a
                    data-driven guess is used. Pass previous day's params for
                    a warm start when running a daily loop.

    Returns:
        params: array [beta0, beta1, beta2, beta3, lambda1, lambda2]
    """
    maturities = np.asarray(maturities, dtype=float)
    yields = np.asarray(yields, dtype=float)
    weights = 1.0 / maturities  # inverse-duration proxy; down-weights noisy long end

    def residuals(p):
        return weights * (svensson_zero_rate(maturities, *p) - yields)

    if x0 is None:
        x0 = [yields[-1], yields[0] - yields[-1], 0.0, 0.0, 2.0, 5.0]
    bounds = (
        [0.00, -0.15, -0.15, -0.15, 0.1, 0.1],
        [0.20,  0.15,  0.15,  0.15, 10., 10.],
    )

    result = least_squares(residuals, x0, method='trf', bounds=bounds)
    return result.x


def filter_bonds_for_fitting(bonds_df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply GSW exclusion rules to a bond universe before Svensson fitting.

    Drops:
    - TIPS: real yields, incompatible with a nominal curve
    - Bills: maturity_years < 1, different pricing convention
    - Callable bonds: embedded option inflates observed yield
    - Bonds with < 90 days to maturity: trade as near-cash
    - On-the-run bonds: liquidity premium suppresses their yield
    - First off-the-run bonds: residual richness bias

    Expected columns in bonds_df:
        type               str  'bill', 'note', 'bond', or 'tips'
        maturity_years     float
        days_to_maturity   int
        callable           bool
        on_the_run         bool
        first_off_the_run  bool

    Returns a copy of the filtered rows.
    """
    mask = (
        (bonds_df["type"] != "tips")
        & (bonds_df["maturity_years"] >= 1.0)
        & (~bonds_df["callable"])
        & (bonds_df["days_to_maturity"] >= 90)
        & (~bonds_df["on_the_run"])
        & (~bonds_df["first_off_the_run"])
    )
    return bonds_df[mask].copy()


def fit_svensson_daily(
    start: str,
    end: str,
    min_maturities: int = 10,
) -> pd.DataFrame:
    """
    Fit Svensson parameters for every date in [start, end].

    Reads Fed zero yields from data/processed/treasury_bonds.parquet.
    For each date, fits our Svensson to the available 1-30Y zero yields
    using a warm start (previous day's solution as x0) for speed.

    Args:
        start:          first date, e.g. '2000-01-01'
        end:            last date,  e.g. '2024-12-31'
        min_maturities: skip dates with fewer non-NaN yield points

    Returns:
        DataFrame with columns:
            date, beta0, beta1, beta2, beta3, lambda1, lambda2,
            rmse_bps, n_bonds
        Also saved to data/processed/svensson_params.parquet.
    """
    from pathlib import Path

    parquet_in  = Path(__file__).resolve().parents[3] / "data/processed/treasury_bonds.parquet"
    parquet_out = Path(__file__).resolve().parents[3] / "data/processed/svensson_params.parquet"

    df = pd.read_parquet(parquet_in)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    df = df.loc[start:end]

    zero_cols    = [f"sveny{i:02d}" for i in range(1, 31)]
    all_mats     = np.arange(1, 31, dtype=float)

    records: list[dict] = []
    prev_params: np.ndarray | None = None
    n = len(df)

    import time as _time
    t0 = _time.perf_counter()
    print(f"Fitting {n} dates from {start} to {end}...")
    for i, (date, row) in enumerate(df.iterrows()):
        if i % 250 == 0:
            elapsed = _time.perf_counter() - t0
            if i > 0:
                eta_min = elapsed / i * (n - i) / 60
                print(f"  {i:>5}/{n} ({i/n*100:.0f}%)  {date.date()}  ETA {eta_min:.1f} min", flush=True)
            else:
                print(f"  {i:>5}/{n} ( 0%)  {date.date()}", flush=True)

        yields_pct = row[zero_cols].to_numpy(dtype=float)
        valid = ~np.isnan(yields_pct)
        if valid.sum() < min_maturities:
            prev_params = None  # gap in data; reset warm start
            continue

        mats = all_mats[valid]
        ylds = yields_pct[valid] / 100.0  # percent → decimal

        try:
            params = fit_svensson(mats, ylds, x0=prev_params)
        except Exception:
            prev_params = None
            continue

        fitted   = svensson_zero_rate(mats, *params)
        rmse_bps = float(np.sqrt(np.mean((fitted - ylds) ** 2)) * 10_000)

        records.append({
            "date":    date,
            "beta0":   params[0],
            "beta1":   params[1],
            "beta2":   params[2],
            "beta3":   params[3],
            "lambda1": params[4],
            "lambda2": params[5],
            "rmse_bps": rmse_bps,
            "n_bonds":  int(valid.sum()),
        })
        prev_params = params

    result = pd.DataFrame(records)
    parquet_out.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(parquet_out, index=False)
    print(f"Done. Saved {len(result):,} rows → {parquet_out}")
    return result


def validate_svensson_daily(
    maturities: list[int] | None = None,
) -> pd.DataFrame:
    """
    Compare our fitted Svensson zero rates against the Fed's published GSW rates.

    Loads svensson_params.parquet (our fits) and treasury_bonds.parquet
    (Fed's sveny01–sveny30 zero yields), then for each common date and each
    maturity computes:
        our_rate = svensson_zero_rate(mat, *our_params)  [decimal]
        fed_rate = svenyXX / 100                         [decimal]
        diff_bps = (our_rate - fed_rate) * 10_000

    Args:
        maturities: integer years 1–30. Defaults to [2, 5, 10, 30].

    Returns:
        DataFrame with columns: date, maturity, our_rate, fed_rate, diff_bps.
        Saved to data/processed/svensson_validation.parquet.
    """
    from pathlib import Path

    if maturities is None:
        maturities = [2, 5, 10, 30]

    params_path = Path(__file__).resolve().parents[3] / "data/processed/svensson_params.parquet"
    bonds_path  = Path(__file__).resolve().parents[3] / "data/processed/treasury_bonds.parquet"
    out_path    = Path(__file__).resolve().parents[3] / "data/processed/svensson_validation.parquet"

    our = pd.read_parquet(params_path)
    our["date"] = pd.to_datetime(our["date"])
    our = our.set_index("date")

    fed = pd.read_parquet(bonds_path)
    fed["date"] = pd.to_datetime(fed["date"])
    fed = fed.set_index("date").sort_index()

    common = our.index.intersection(fed.index)
    param_cols = ["beta0", "beta1", "beta2", "beta3", "lambda1", "lambda2"]

    records: list[dict] = []
    for mat in maturities:
        col = f"sveny{mat:02d}"
        valid = ~fed.loc[common, col].isna()
        dates = common[valid]
        if len(dates) == 0:
            continue

        fed_rates = fed.loc[dates, col].to_numpy() / 100.0
        our_sub   = our.loc[dates, param_cols].to_numpy()
        our_rates = np.array([svensson_zero_rate(float(mat), *row) for row in our_sub])
        diff_bps  = (our_rates - fed_rates) * 10_000

        for date, our_r, fed_r, diff in zip(dates, our_rates, fed_rates, diff_bps):
            records.append({
                "date":     date,
                "maturity": mat,
                "our_rate": our_r,
                "fed_rate": fed_r,
                "diff_bps": diff,
            })

    result = pd.DataFrame(records)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(out_path, index=False)
    return result
