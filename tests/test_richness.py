"""Integration tests for richness/cheapness signal pipeline."""

from pathlib import Path

import pytest

from termstructure.signals.richness import (
    build_signal_panel,
    compute_residuals,
    compute_zscore_signal,
)

_RESIDUALS = Path("data/processed/bond_residuals.parquet")
_SIGNAL    = Path("data/processed/richness_signal.parquet")
_SVENSSON  = Path("data/processed/svensson_params.parquet")

_requires_residuals = pytest.mark.skipif(
    not _RESIDUALS.exists(), reason="bond_residuals.parquet not on disk"
)
_requires_svensson = pytest.mark.skipif(
    not _SVENSSON.exists(), reason="svensson_params.parquet not on disk"
)
_requires_signal = pytest.mark.skipif(
    not _SIGNAL.exists(), reason="richness_signal.parquet not on disk"
)


# ─── compute_residuals ────────────────────────────────────────────────────────

@_requires_svensson
def test_compute_residuals_schema() -> None:
    """compute_residuals returns expected columns."""
    df = compute_residuals()
    required = {"date", "maturity", "observed_rate", "fitted_rate", "residual_bps"}
    assert required.issubset(set(df.columns))


@_requires_svensson
def test_compute_residuals_nonempty() -> None:
    """compute_residuals returns at least 10,000 rows."""
    df = compute_residuals()
    assert len(df) > 10_000


@_requires_svensson
def test_compute_residuals_no_outliers() -> None:
    """All residuals are within the ±30bp filter applied inside the function."""
    df = compute_residuals()
    assert (df["residual_bps"].abs() <= 30).all()


@_requires_svensson
def test_compute_residuals_demeaned() -> None:
    """After demeaning, per-maturity mean should be near zero."""
    df = compute_residuals()
    means = df.groupby("maturity")["residual_bps"].mean()
    assert (means.abs() < 1.0).all()


# ─── compute_zscore_signal ────────────────────────────────────────────────────

@_requires_residuals
def test_compute_zscore_signal_schema() -> None:
    """compute_zscore_signal returns the expected columns."""
    df = compute_zscore_signal()
    required = {"date", "maturity", "residual_bps", "rolling_mean", "rolling_std", "z_score"}
    assert required.issubset(set(df.columns))


@_requires_residuals
def test_compute_zscore_signal_no_nans() -> None:
    """z_score column has no NaN values (NaN rows are dropped before return)."""
    df = compute_zscore_signal()
    assert df["z_score"].isna().sum() == 0


@_requires_residuals
def test_compute_zscore_signal_maturities() -> None:
    """Signal covers the expected maturity grid."""
    df = compute_zscore_signal()
    assert set(df["maturity"].unique()) >= {1, 2, 3, 5, 7, 10, 20, 30}


@_requires_residuals
def test_compute_zscore_signal_nonempty() -> None:
    """Signal has at least 10,000 rows."""
    df = compute_zscore_signal()
    assert len(df) > 10_000


# ─── build_signal_panel ───────────────────────────────────────────────────────

@_requires_signal
def test_build_signal_panel_schema() -> None:
    """build_signal_panel returns expected columns including direction."""
    df = build_signal_panel()
    required = {"date", "maturity", "residual_bps", "rolling_mean", "rolling_std",
                "z_score", "direction"}
    assert required.issubset(set(df.columns))


@_requires_signal
def test_build_signal_panel_direction_values() -> None:
    """direction column contains only -1, 0, +1."""
    df = build_signal_panel()
    assert set(df["direction"].unique()).issubset({-1, 0, 1})


@_requires_signal
def test_build_signal_panel_threshold_consistency() -> None:
    """direction=+1 rows all have z_score > 2.0, direction=-1 all have z_score < -2.0."""
    df = build_signal_panel(entry_z=2.0)
    longs  = df[df["direction"] ==  1]
    shorts = df[df["direction"] == -1]
    assert (longs["z_score"] > 2.0).all()
    assert (shorts["z_score"] < -2.0).all()


@_requires_signal
def test_build_signal_panel_wider_threshold_fewer_signals() -> None:
    """Higher entry_z → fewer signal rows."""
    df_2 = build_signal_panel(entry_z=2.0)
    df_3 = build_signal_panel(entry_z=3.0)
    n_signal_2 = (df_2["direction"] != 0).sum()
    n_signal_3 = (df_3["direction"] != 0).sum()
    assert n_signal_3 < n_signal_2
    # Restore production threshold so downstream data is not corrupted
    build_signal_panel(entry_z=2.0)
