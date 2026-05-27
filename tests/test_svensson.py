"""Week 3 tests: GSW exclusion rules, Svensson fit quality, daily loop."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.termstructure.curves.svensson import (
    filter_bonds_for_fitting,
    fit_svensson,
    fit_svensson_daily,
    svensson_zero_rate,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bond(**overrides) -> dict:
    """Return a valid base bond; override any field via kwargs."""
    base = {
        "type": "note",
        "maturity_years": 5.0,
        "days_to_maturity": 5 * 365,
        "callable": False,
        "on_the_run": False,
        "first_off_the_run": False,
        "yield_pct": 0.04,
    }
    return {**base, **overrides}


# ---------------------------------------------------------------------------
# filter_bonds_for_fitting — one test per exclusion rule
# ---------------------------------------------------------------------------

def test_valid_bond_survives():
    df = pd.DataFrame([_bond()])
    assert len(filter_bonds_for_fitting(df)) == 1


def test_tips_excluded():
    df = pd.DataFrame([_bond(type="tips")])
    assert len(filter_bonds_for_fitting(df)) == 0


def test_bill_excluded():
    df = pd.DataFrame([_bond(type="bill", maturity_years=0.5)])
    assert len(filter_bonds_for_fitting(df)) == 0


def test_callable_excluded():
    df = pd.DataFrame([_bond(callable=True)])
    assert len(filter_bonds_for_fitting(df)) == 0


def test_short_maturity_excluded():
    """60 days remaining is below the 90-day threshold."""
    df = pd.DataFrame([_bond(days_to_maturity=60)])
    assert len(filter_bonds_for_fitting(df)) == 0


def test_on_the_run_excluded():
    df = pd.DataFrame([_bond(on_the_run=True)])
    assert len(filter_bonds_for_fitting(df)) == 0


def test_first_off_the_run_excluded():
    df = pd.DataFrame([_bond(first_off_the_run=True)])
    assert len(filter_bonds_for_fitting(df)) == 0


# ---------------------------------------------------------------------------
# Boundary conditions
# ---------------------------------------------------------------------------

def test_exactly_90_days_included():
    """90 days is the boundary — must pass (>= 90)."""
    df = pd.DataFrame([_bond(days_to_maturity=90)])
    assert len(filter_bonds_for_fitting(df)) == 1


def test_exactly_1_year_maturity_included():
    """1.0-year maturity is the bill boundary — must pass (>= 1.0)."""
    df = pd.DataFrame([_bond(maturity_years=1.0)])
    assert len(filter_bonds_for_fitting(df)) == 1


def test_89_days_excluded():
    """89 days is strictly below 90 — must fail."""
    df = pd.DataFrame([_bond(days_to_maturity=89)])
    assert len(filter_bonds_for_fitting(df)) == 0


# ---------------------------------------------------------------------------
# Mixed universe
# ---------------------------------------------------------------------------

def test_mixed_universe_keeps_only_valid():
    """6 excluded bonds + 2 valid ones → only 2 survive."""
    bonds = [
        _bond(),                                      # valid
        _bond(maturity_years=10.0),                  # valid
        _bond(type="tips"),                           # excluded: TIPS
        _bond(type="bill", maturity_years=0.25),      # excluded: bill
        _bond(callable=True),                         # excluded: callable
        _bond(days_to_maturity=45),                   # excluded: < 90 days
        _bond(on_the_run=True),                       # excluded: on-the-run
        _bond(first_off_the_run=True),                # excluded: 1st off-the-run
    ]
    result = filter_bonds_for_fitting(pd.DataFrame(bonds))
    assert len(result) == 2


def test_filter_returns_copy():
    """Mutating the filtered result must not affect the original."""
    df = pd.DataFrame([_bond()])
    result = filter_bonds_for_fitting(df)
    result.iloc[0, result.columns.get_loc("yield_pct")] = 999.0
    assert df.iloc[0]["yield_pct"] == pytest.approx(0.04)


# ---------------------------------------------------------------------------
# fit_svensson quality — single-date fit
# ---------------------------------------------------------------------------

def test_fit_recovers_known_curve_within_1bp():
    """
    Fit to noiseless synthetic yields generated from known parameters.
    RMSE should be < 1 bp when there's no noise in the input.
    """
    params_true = [0.04, -0.01, 0.02, 0.01, 2.0, 5.0]
    maturities = np.array([1, 2, 3, 5, 7, 10, 15, 20, 30], dtype=float)
    yields = svensson_zero_rate(maturities, *params_true)

    fitted_params = fit_svensson(maturities, yields)
    fitted_yields = svensson_zero_rate(maturities, *fitted_params)

    rmse_bps = np.sqrt(np.mean((fitted_yields - yields) ** 2)) * 10_000
    assert rmse_bps < 1.0


def test_fit_upward_curve_has_positive_slope():
    """
    For an upward-sloping input, the fitted curve must also slope upward:
    long-end yield > short-end yield.
    """
    params_true = [0.04, -0.015, 0.01, 0.01, 2.0, 5.0]
    maturities = np.array([1, 2, 3, 5, 7, 10, 20, 30], dtype=float)
    yields = svensson_zero_rate(maturities, *params_true)

    fitted_params = fit_svensson(maturities, yields)
    short_rate = svensson_zero_rate(0.5, *fitted_params)
    long_rate = svensson_zero_rate(30.0, *fitted_params)

    assert long_rate > short_rate


def test_fit_svensson_warm_start_matches_cold():
    """
    Warm start (x0 = true params) must converge to the same curve as a cold
    start. Parameters may differ (Svensson is ill-conditioned), but RMSE must
    be equally small.
    """
    params_true = [0.04, -0.01, 0.02, 0.01, 2.0, 5.0]
    maturities = np.array([1, 2, 3, 5, 7, 10, 15, 20, 30], dtype=float)
    yields = svensson_zero_rate(maturities, *params_true)

    cold = fit_svensson(maturities, yields)
    warm = fit_svensson(maturities, yields, x0=params_true)

    rmse_cold = np.sqrt(np.mean((svensson_zero_rate(maturities, *cold) - yields) ** 2)) * 10_000
    rmse_warm = np.sqrt(np.mean((svensson_zero_rate(maturities, *warm) - yields) ** 2)) * 10_000

    assert rmse_cold < 1.0
    assert rmse_warm < 1.0


# ---------------------------------------------------------------------------
# fit_svensson_daily — integration tests (skip if parquet not on disk)
# ---------------------------------------------------------------------------

_PARQUET = Path("data/processed/treasury_bonds.parquet")


@pytest.mark.skipif(not _PARQUET.exists(), reason="treasury_bonds.parquet not on disk")
def test_fit_svensson_daily_schema():
    """Output must have all required columns and correct dtypes."""
    result = fit_svensson_daily("2020-01-02", "2020-01-15")

    required = {"date", "beta0", "beta1", "beta2", "beta3",
                "lambda1", "lambda2", "rmse_bps", "n_bonds"}
    assert required.issubset(set(result.columns))
    assert len(result) >= 9  # ~9 business days in the window
    assert result["n_bonds"].dtype == int or result["n_bonds"].dtype == np.int64


@pytest.mark.skipif(not _PARQUET.exists(), reason="treasury_bonds.parquet not on disk")
def test_fit_svensson_daily_rmse_plausible():
    """All RMSE values in a recent window should be under 5 bp."""
    result = fit_svensson_daily("2023-01-02", "2023-01-31")
    assert result["rmse_bps"].max() < 5.0


@pytest.mark.skipif(not _PARQUET.exists(), reason="treasury_bonds.parquet not on disk")
def test_fit_svensson_daily_lambda_bounds():
    """λ1 and λ2 must stay within the optimizer bounds [0.1, 10]."""
    result = fit_svensson_daily("2023-01-02", "2023-01-31")
    assert (result["lambda1"] >= 0.1).all() and (result["lambda1"] <= 10.0).all()
    assert (result["lambda2"] >= 0.1).all() and (result["lambda2"] <= 10.0).all()


@pytest.mark.skipif(not _PARQUET.exists(), reason="treasury_bonds.parquet not on disk")
def test_fit_svensson_daily_saves_parquet(tmp_path, monkeypatch):
    """Running the loop must create svensson_params.parquet on disk."""
    out = Path("data/processed/svensson_params.parquet")
    result = fit_svensson_daily("2024-01-02", "2024-01-05")
    assert out.exists()
    on_disk = pd.read_parquet(out)
    assert len(on_disk) >= 1
