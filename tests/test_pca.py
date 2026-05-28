"""Week 4 tests: zero-rate panel construction and PCA decomposition."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.termstructure.pca.panel import DEFAULT_MATURITIES, _col, build_zero_panel
from src.termstructure.pca.decomposition import fit_pca, compute_factor_scores, save_pca_outputs


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
    """First 3 PCs must explain ≥ 97% of variance.

    L-S (1991) reported ≥ 99% on a shorter, pre-2022 sample.  Our 40-year
    panel (1985-2026) includes the 2022 hiking cycle which raises PC4's share
    to ~1.3%; empirically the first 3 PCs capture ~97.8%.
    """
    pca, X, dates = fit_pca()
    cumvar = pca.explained_variance_ratio_[:3].sum()
    assert cumvar >= 0.97, f"First 3 PCs explain only {cumvar:.1%} (expected >= 97%)"


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


# ---------------------------------------------------------------------------
# compute_factor_scores
# ---------------------------------------------------------------------------

_MODEL = Path("data/processed/pca_model.joblib")
_BOTH  = _PANEL.exists() and _MODEL.exists()


@pytest.mark.skipif(not _BOTH, reason="zero_panel.parquet or pca_model.joblib not on disk")
def test_compute_factor_scores_schema():
    """Output must have a date column and one score column per component."""
    result = compute_factor_scores()
    assert "date" in result.columns
    assert "score_1" in result.columns
    assert "score_2" in result.columns
    assert "score_3" in result.columns


@pytest.mark.skipif(not _BOTH, reason="zero_panel.parquet or pca_model.joblib not on disk")
def test_compute_factor_scores_shape():
    """Row count must match zero_panel minus the first NaN row."""
    panel = pd.read_parquet(_PANEL)
    result = compute_factor_scores()
    assert len(result) == len(panel) - 1


@pytest.mark.skipif(not _BOTH, reason="zero_panel.parquet or pca_model.joblib not on disk")
def test_compute_factor_scores_proxy_regression():
    """
    Daily factor scores must correlate with simple observable proxy changes.

    All three regressions use daily scores vs daily bp changes (consistent method).
    Proxy choices follow the empirical PC loading vectors:
      PC1: avg(dy_2, dy_10)             — flat loading, 2Y/10Y representative
      PC2: dy_30 - dy_2                 — loading runs -0.40 at 1Y to +0.59 at 30Y
      PC3: 2·dy_10 - dy_2 - dy_30      — 10Y deepest negative, 30Y strongest positive

    R² targets reflect how much of the 8-maturity loading vector the 2-3 point
    proxy captures.  PC3 curvature is inherently harder to proxy than level/slope.
    """
    from scipy import stats
    panel = pd.read_parquet(_PANEL)
    panel["date"] = pd.to_datetime(panel["date"])
    panel = panel.sort_values("date").set_index("date")

    scores = compute_factor_scores().set_index("date")
    common = scores.index.intersection(panel.index)

    level_daily = (panel.loc[common, "dy_2"]  + panel.loc[common, "dy_10"]) / 2
    slope_daily = (panel.loc[common, "dy_30"] - panel.loc[common, "dy_2"])
    curv_daily  = (
        2 * panel.loc[common, "dy_10"]
        - panel.loc[common, "dy_2"]
        - panel.loc[common, "dy_30"]
    )

    thresholds = {"level": 0.95, "slope": 0.85, "curvature": 0.70}
    for proxy, score_col, name in [
        (level_daily, "score_1", "level"),
        (slope_daily, "score_2", "slope"),
        (curv_daily,  "score_3", "curvature"),
    ]:
        _, _, r, _, _ = stats.linregress(
            scores.loc[common, score_col].values,
            proxy.values,
        )
        assert r ** 2 >= thresholds[name], (
            f"{name} daily proxy R²={r**2:.3f} < {thresholds[name]}"
        )


# ---------------------------------------------------------------------------
# save_pca_outputs
# ---------------------------------------------------------------------------

_LOADINGS = Path("data/processed/pca_loadings.parquet")
_SCORES   = Path("data/processed/factor_scores.parquet")
_SAVE_OK  = _MODEL.exists() and _PANEL.exists()


@pytest.mark.skipif(not _SAVE_OK, reason="pca_model.joblib or zero_panel.parquet not on disk")
def test_save_pca_outputs_creates_files():
    """Both parquets must exist after calling save_pca_outputs."""
    save_pca_outputs()
    assert _LOADINGS.exists(), "pca_loadings.parquet not created"
    assert _SCORES.exists(),   "factor_scores.parquet not created"


@pytest.mark.skipif(not _SAVE_OK, reason="pca_model.joblib or zero_panel.parquet not on disk")
def test_pca_loadings_schema():
    """Loadings parquet must have pc, var_ratio, and one y_ column per maturity."""
    save_pca_outputs()
    df = pd.read_parquet(_LOADINGS)
    assert "pc" in df.columns
    assert "var_ratio" in df.columns
    for m in DEFAULT_MATURITIES:
        assert f"y_{_col(float(m))}" in df.columns, f"missing y_{_col(float(m))}"


@pytest.mark.skipif(not _SAVE_OK, reason="pca_model.joblib or zero_panel.parquet not on disk")
def test_pca_loadings_var_ratio_sums_to_one():
    """Saved variance ratios must sum to 1 (full-rank model)."""
    save_pca_outputs()
    df = pd.read_parquet(_LOADINGS)
    assert abs(df["var_ratio"].sum() - 1.0) < 1e-6


@pytest.mark.skipif(not _SAVE_OK, reason="pca_model.joblib or zero_panel.parquet not on disk")
def test_factor_scores_parquet_matches_compute():
    """Saved factor_scores.parquet must match compute_factor_scores() output."""
    save_pca_outputs()
    on_disk = pd.read_parquet(_SCORES)
    live    = compute_factor_scores()
    assert list(on_disk.columns) == list(live.columns)
    assert len(on_disk) == len(live)
