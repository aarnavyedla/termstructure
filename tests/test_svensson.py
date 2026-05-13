"""Week 3 Day 3 tests: GSW exclusion rules and Svensson fit quality."""

import numpy as np
import pandas as pd
import pytest

from src.termstructure.curves.svensson import (
    filter_bonds_for_fitting,
    fit_svensson,
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
