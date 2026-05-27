"""Week 4 tests: zero-rate panel construction."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.termstructure.pca.panel import DEFAULT_MATURITIES, _col, build_zero_panel


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
    assert len(DEFAULT_MATURITIES) == 10


def test_default_maturities_sorted():
    assert DEFAULT_MATURITIES == sorted(DEFAULT_MATURITIES)


# ---------------------------------------------------------------------------
# Integration tests (skip if svensson_params.parquet not on disk)
# ---------------------------------------------------------------------------

_PARAMS = Path("data/processed/svensson_params.parquet")


@pytest.mark.skipif(not _PARAMS.exists(), reason="svensson_params.parquet not on disk")
def test_build_zero_panel_schema():
    """Output must have date + y_/dy_ columns for every maturity."""
    result = build_zero_panel()
    assert "date" in result.columns
    for mat in DEFAULT_MATURITIES:
        label = _col(mat)
        assert f"y_{label}"  in result.columns, f"missing y_{label}"
        assert f"dy_{label}" in result.columns, f"missing dy_{label}"


@pytest.mark.skipif(not _PARAMS.exists(), reason="svensson_params.parquet not on disk")
def test_build_zero_panel_row_count():
    """Row count must match svensson_params.parquet."""
    params = pd.read_parquet(_PARAMS)
    result = build_zero_panel()
    assert len(result) == len(params)


@pytest.mark.skipif(not _PARAMS.exists(), reason="svensson_params.parquet not on disk")
def test_build_zero_panel_levels_no_nan():
    """Level columns must have no NaN (all dates have valid params)."""
    result = build_zero_panel()
    level_cols = [f"y_{_col(m)}" for m in DEFAULT_MATURITIES]
    assert result[level_cols].isna().sum().sum() == 0


@pytest.mark.skipif(not _PARAMS.exists(), reason="svensson_params.parquet not on disk")
def test_build_zero_panel_first_row_nan():
    """dy_ columns must be NaN on the first date (no prior day to diff against)."""
    result = build_zero_panel()
    change_cols = [f"dy_{_col(m)}" for m in DEFAULT_MATURITIES]
    assert result.loc[0, change_cols].isna().all()


@pytest.mark.skipif(not _PARAMS.exists(), reason="svensson_params.parquet not on disk")
def test_build_zero_panel_changes_after_first_no_nan():
    """dy_ columns must have no NaN after the first row."""
    result = build_zero_panel()
    change_cols = [f"dy_{_col(m)}" for m in DEFAULT_MATURITIES]
    assert result.iloc[1:][change_cols].isna().sum().sum() == 0


@pytest.mark.skipif(not _PARAMS.exists(), reason="svensson_params.parquet not on disk")
def test_build_zero_panel_levels_are_decimal():
    """Zero rates should be in decimal (< 1), not percent."""
    result = build_zero_panel()
    level_cols = [f"y_{_col(m)}" for m in DEFAULT_MATURITIES]
    assert result[level_cols].max().max() < 1.0


@pytest.mark.skipif(not _PARAMS.exists(), reason="svensson_params.parquet not on disk")
def test_build_zero_panel_changes_are_bps():
    """Typical daily changes should be in the range of single-digit bp, not decimal."""
    result = build_zero_panel()
    change_cols = [f"dy_{_col(m)}" for m in DEFAULT_MATURITIES]
    median_abs = result[change_cols].iloc[1:].abs().median().median()
    # Median absolute daily change: typically 1–5 bp, never 0.0001–0.0005
    assert 0.1 < median_abs < 20.0


@pytest.mark.skipif(not _PARAMS.exists(), reason="svensson_params.parquet not on disk")
def test_build_zero_panel_saves_parquet():
    """Output parquet must be written to disk."""
    out = Path("data/processed/zero_panel.parquet")
    build_zero_panel()
    assert out.exists()
    on_disk = pd.read_parquet(out)
    assert len(on_disk) > 0


@pytest.mark.skipif(not _PARAMS.exists(), reason="svensson_params.parquet not on disk")
def test_build_zero_panel_sorted_by_date():
    """Dates must be in ascending order."""
    result = build_zero_panel()
    assert result["date"].is_monotonic_increasing
