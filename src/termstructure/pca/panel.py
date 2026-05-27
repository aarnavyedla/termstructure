import numpy as np
import pandas as pd

# Only integer maturities covered by the Fed's sveny columns.
# Sub-1Y maturities (0.25, 0.5) are Svensson extrapolations with no bond
# observations to anchor them — their day-to-day noise from parameter
# instability distorts the PCA covariance structure.
DEFAULT_MATURITIES = [1, 2, 3, 5, 7, 10, 20, 30]


def _col(mat: float) -> str:
    """'1' for whole numbers, '0.25' for fractions."""
    return str(int(mat)) if mat == int(mat) else str(mat)


def build_zero_panel(
    maturities: list[int] | None = None,
) -> pd.DataFrame:
    """
    Build a panel of zero rates and their daily changes.

    Reads the Fed's published sveny columns from treasury_bonds.parquet
    directly — not from our Svensson fit. This avoids parameter-instability
    noise that corrupts the PCA covariance structure.

    Args:
        maturities: integer maturities in years (1–30). Defaults to
                    [1, 2, 3, 5, 7, 10, 20, 30].

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

    bonds_path = Path(__file__).resolve().parents[3] / "data/processed/treasury_bonds.parquet"
    out_path   = Path(__file__).resolve().parents[3] / "data/processed/zero_panel.parquet"

    df = pd.read_parquet(bonds_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    sveny_cols = [f"sveny{int(m):02d}" for m in maturities]
    rates = df[sveny_cols].to_numpy(dtype=float) / 100.0  # percent → decimal

    # Drop dates where any maturity is missing (sparse early history)
    valid = ~np.isnan(rates).any(axis=1)
    rates = rates[valid]
    dates = df["date"].to_numpy()[valid]

    # First differences in bp; NaN on first row
    changes = np.vstack([
        np.full((1, len(maturities)), np.nan),
        np.diff(rates, axis=0) * 10_000,
    ])

    result = pd.DataFrame({"date": dates})
    for i, mat in enumerate(maturities):
        label = _col(float(mat))
        result[f"y_{label}"]  = rates[:, i]
        result[f"dy_{label}"] = changes[:, i]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(out_path, index=False)
    return result
