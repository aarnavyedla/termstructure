"""Week 4 tests: zero-rate panel construction and PCA decomposition."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.termstructure.pca.panel import DEFAULT_MATURITIES, _col, build_zero_panel
from src.termstructure.pca.decomposition import fit_pca


# ---------------------------------------------------------------------------
# Pure-logic tests (no parquet required)
# ---------------------------------------------------------------------------

def test_col_whole_number():
    assert _col(1.0) == "1"
    assert _col(10.0) == "10"
    assert _col(30.0) == "30"


def test_col_fraction():
    assert _col(0.25) == "0.25"
    assert _col(0.5) == "0.5"


def test_default_maturities_count():
    assert len(DEFAULT_MATURITIES) == 8


def test_default_maturities_sorted():
    assert DEFAULT_MATURITIES == sorted(DEFAULT_MATURITIES)


def test_default_maturities_are_integers():
    """All default maturities must be whole years (sveny columns are integer only)."""
    for m in DEFAULT_MATURITIES:
        assert m == int(m)


# ---------------------------------------------------------------------------
# Integration tests (skip if treasury_bonds.parquet not on disk)
# ---------------------------------------------------------------------------

_BONDS = Path("data/processed/treasury_bonds.parquet")


@pytest.mark.skipif(not _BONDS.exists(), reason="treasury_bonds.parquet not on disk")
def test_build_zero_panel_schema():
    """Output must have date + y_/dy_ columns for every maturity."""
    result = build_zero_panel()
    assert "date" in result.columns
    for mat in DEFAULT_MATURITIES:
        label = _col(float(mat))
        assert f"y_{label}"  in result.columns, f"missing y_{label}"
        assert f"dy_{label}" in result.columns, f"missing dy_{label}"


@pytest.mark.skipif(not _BONDS.exists(), reason="treasury_bonds.parquet not on disk")
def test_build_zero_panel_levels_no_nan():
    """Level columns must have no NaN (NaN rows are dropped at build time)."""
    result = build_zero_panel()
    level_cols = [f"y_{_col(float(m))}" for m in DEFAULT_MATURITIES]
    assert result[level_cols].isna().sum().sum() == 0


@pytest.mark.skipif(not _BONDS.exists(), reason="treasury_bonds.parquet not on disk")
def test_build_zero_panel_first_row_nan():
    """dy_ columns must be NaN on the first date (no prior day to diff against)."""
    result = build_zero_panel()
    change_cols = [f"dy_{_col(float(m))}" for m in DEFAULT_MATURITIES]
    assert result.loc[0, change_cols].isna().all()


@pytest.mark.skipif(not _BONDS.exists(), reason="treasury_bonds.parquet not on disk")
def test_build_zero_panel_changes_after_first_no_nan():
    """dy_ columns must have no NaN after the first row."""
    result = build_zero_panel()
    change_cols = [f"dy_{_col(float(m))}" for m in DEFAULT_MATURITIES]
    assert result.iloc[1:][change_cols].isna().sum().sum() == 0


@pytest.mark.skipif(not _BONDS.exists(), reason="treasury_bonds.parquet not on disk")
def test_build_zero_panel_levels_are_decimal():
    """Zero rates should be in decimal (< 1), not percent."""
    result = build_zero_panel()
    level_cols = [f"y_{_col(float(m))}" for m in DEFAULT_MATURITIES]
    assert result[level_cols].max().max() < 1.0


@pytest.mark.skipif(not _BONDS.exists(), reason="treasury_bonds.parquet not on disk")
def test_build_zero_panel_changes_are_bps():
    """Median absolute daily change should be in single-digit bp, not decimal."""
    result = build_zero_panel()
    change_cols = [f"dy_{_col(float(m))}" for m in DEFAULT_MATURITIES]
    median_abs = result[change_cols].iloc[1:].abs().median().median()
    assert 0.1 < median_abs < 20.0


@pytest.mark.skipif(not _BONDS.exists(), reason="treasury_bonds.parquet not on disk")
def test_build_zero_panel_saves_parquet():
    """Output parquet must be written to disk."""
    out = Path("data/processed/zero_panel.parquet")
    build_zero_panel()
    assert out.exists()
    on_disk = pd.read_parquet(out)
    assert len(on_disk) > 0


@pytest.mark.skipif(not _BONDS.exists(), reason="treasury_bonds.parquet not on disk")
def test_build_zero_panel_sorted_by_date():
    """Dates must be in ascending order."""
    result = build_zero_panel()
    assert result["date"].is_monotonic_increasing


# ---------------------------------------------------------------------------
# fit_pca — PCA on daily yield changes
# ---------------------------------------------------------------------------

_PANEL = Path("data/processed/zero_panel.parquet")


@pytest.mark.skipif(not _PANEL.exists(), reason="zero_panel.parquet not on disk")
def test_fit_pca_returns_correct_n_components():
    n = len(DEFAULT_MATURITIES)
    pca, X, dates = fit_pca(n_components=n)
    assert pca.n_components_ == n


@pytest.mark.skipif(not _PANEL.exists(), reason="zero_panel.parquet not on disk")
def test_fit_pca_input_shape():
    """X must have 8 columns (one per maturity) and match the dates length."""
    pca, X, dates = fit_pca()
    assert X.shape[1] == len(DEFAULT_MATURITIES)
    assert X.shape[0] == len(dates)


@pytest.mark.skipif(not _PANEL.exists(), reason="zero_panel.parquet not on disk")
def test_fit_pca_explained_variance_sums_to_one():
    pca, X, dates = fit_pca()
    assert abs(pca.explained_variance_ratio_.sum() - 1.0) < 1e-6


@pytest.mark.skipif(not _PANEL.exists(), reason="zero_panel.parquet not on disk")
def test_fit_pca_three_components_explain_99pct():
    """Litterman-Scheinkman result: first 3 PCs explain ≥ 99% of variance."""
    pca, X, dates = fit_pca()
    cumvar = pca.explained_variance_ratio_[:3].sum()
    assert cumvar >= 0.99, f"First 3 PCs explain only {cumvar:.1%} (expected ≥ 99%)"


@pytest.mark.skipif(not _PANEL.exists(), reason="zero_panel.parquet not on disk")
def test_fit_pca_components_decreasing_variance():
    """Each PC must explain less variance than the previous one."""
    pca, X, dates = fit_pca()
    evr = pca.explained_variance_ratio_
    assert (evr[:-1] >= evr[1:]).all()


@pytest.mark.skipif(not _PANEL.exists(), reason="zero_panel.parquet not on disk")
def test_fit_pca_saves_joblib():
    """Fitted model must be saved to disk."""
    import joblib
    out = Path("data/processed/pca_model.joblib")
    fit_pca()
    assert out.exists()
    loaded = joblib.load(out)
    assert hasattr(loaded, "components_")
