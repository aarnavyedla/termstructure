"""Week 2 tests: pricing, duration, key-rate duration, and conversions."""

import numpy as np
import pytest

from src.termstructure.bonds.pricing import (
    price_from_ytm,
    ytm_from_price,
    modified_duration,
    dv01,
    price_from_curve,
    key_rate_durations,
)
from src.termstructure.bonds.conversions import (
    zero_to_discount,
    discount_to_zero,
    zeros_to_par,
    par_to_zeros,
)


# ---------------------------------------------------------------------------
# price_from_ytm
# ---------------------------------------------------------------------------

def test_par_bond_prices_at_face():
    """A bond whose coupon equals its YTM must price at face value."""
    price = price_from_ytm(coupon=0.05, maturity_years=10, ytm=0.05)
    assert abs(price - 100.0) < 1e-6


def test_premium_bond():
    """Coupon > YTM means the bond is worth more than face."""
    price = price_from_ytm(coupon=0.06, maturity_years=10, ytm=0.04)
    assert price > 100.0


def test_discount_bond():
    """Coupon < YTM means the bond trades below face."""
    price = price_from_ytm(coupon=0.03, maturity_years=10, ytm=0.05)
    assert price < 100.0


def test_zero_coupon_bond():
    """A zero-coupon bond is just the discounted face value."""
    price = price_from_ytm(coupon=0.0, maturity_years=5, ytm=0.04)
    expected = 100 / (1 + 0.04 / 2) ** 10
    assert abs(price - expected) < 1e-10


def test_price_increases_as_ytm_falls():
    """Price and yield move in opposite directions."""
    p_low  = price_from_ytm(0.05, 10, ytm=0.03)
    p_high = price_from_ytm(0.05, 10, ytm=0.07)
    assert p_low > p_high


# ---------------------------------------------------------------------------
# ytm_from_price
# ---------------------------------------------------------------------------

def test_ytm_round_trip():
    """price -> ytm -> price must recover the original price to 6 decimal places."""
    original_price = 95.0
    ytm = ytm_from_price(original_price, coupon=0.04, maturity_years=7)
    recovered = price_from_ytm(coupon=0.04, maturity_years=7, ytm=ytm)
    assert abs(recovered - original_price) < 1e-6


def test_par_bond_yields_coupon_rate():
    """A bond priced at par yields exactly its coupon rate."""
    ytm = ytm_from_price(price=100.0, coupon=0.05, maturity_years=10)
    assert abs(ytm - 0.05) < 1e-6


# ---------------------------------------------------------------------------
# modified_duration and dv01
# ---------------------------------------------------------------------------

def test_longer_bond_has_higher_duration():
    """Duration increases with maturity, all else equal."""
    md_5y  = modified_duration(0.05, 5,  ytm=0.05)
    md_10y = modified_duration(0.05, 10, ytm=0.05)
    md_30y = modified_duration(0.05, 30, ytm=0.05)
    assert md_5y < md_10y < md_30y


def test_dv01_equals_duration_times_price():
    """DV01 = modified_duration * price * 1bp, to within numerical tolerance."""
    coupon, mat, ytm = 0.05, 10, 0.05
    md = modified_duration(coupon, mat, ytm)
    p  = price_from_ytm(coupon, mat, ytm)
    computed_dv01 = dv01(coupon, mat, ytm)
    assert abs(computed_dv01 - md * p * 1e-4) < 1e-8


def test_dv01_positive():
    """DV01 must always be positive (price falls when yield rises)."""
    assert dv01(0.04, 10, ytm=0.05) > 0


# ---------------------------------------------------------------------------
# price_from_curve and key_rate_durations
# ---------------------------------------------------------------------------

def test_price_from_flat_curve_matches_ytm():
    """Pricing off a flat spot curve at rate r equals price_from_ytm at ytm=r."""
    coupon, mat, r = 0.05, 10, 0.04
    curve_mats  = np.array([1, 2, 5, 10, 30], dtype=float)
    curve_rates = np.full(5, r)
    p_curve = price_from_curve(coupon, mat, curve_mats, curve_rates)
    p_ytm   = price_from_ytm(coupon, mat, ytm=r)
    assert abs(p_curve - p_ytm) < 1e-6


def test_krd_sum_equals_modified_duration():
    """The sum of KRDs at all buckets equals the bond's modified duration.

    The tent functions tile the maturity axis, so a parallel shift is exactly
    decomposed by the four buckets.
    """
    coupon, mat, ytm = 0.05, 10, 0.05
    bucket_points = [2, 5, 10, 30]
    curve_mats  = np.array(bucket_points, dtype=float)
    curve_rates = np.full(len(bucket_points), ytm)

    krds = key_rate_durations(coupon, mat, curve_mats, curve_rates,
                              bucket_points=bucket_points)
    krd_sum = sum(krds.values())
    md = modified_duration(coupon, mat, ytm)
    assert abs(krd_sum - md) < 1e-4


def test_krd_beyond_maturity_is_zero():
    """A 5Y bond has no sensitivity to the 30Y bucket."""
    coupon, mat = 0.05, 5.0
    ytm = 0.05
    bucket_points = [2, 5, 10, 30]
    curve_mats  = np.array(bucket_points, dtype=float)
    curve_rates = np.full(len(bucket_points), ytm)

    krds = key_rate_durations(coupon, mat, curve_mats, curve_rates,
                              bucket_points=bucket_points)
    assert abs(krds[30]) < 1e-6


# ---------------------------------------------------------------------------
# conversions
# ---------------------------------------------------------------------------

def test_zero_discount_round_trip():
    """zero -> discount factor -> zero must recover the original rate."""
    z = 0.045
    df = zero_to_discount(z, maturity=7)
    assert abs(discount_to_zero(df, maturity=7) - z) < 1e-15


def test_discount_factor_less_than_one():
    """A positive rate always gives a discount factor in (0, 1)."""
    df = zero_to_discount(0.04, maturity=10)
    assert 0 < df < 1


def test_par_yields_equal_zeros_on_flat_curve():
    """On a flat zero curve, par yields must equal the zero rate at every point."""
    mats = np.arange(0.5, 10.5, 0.5)
    zeros = np.full(len(mats), 0.04)
    pars = zeros_to_par(mats, zeros)
    assert np.max(np.abs(pars - 0.04)) < 1e-12


def test_zeros_to_par_round_trip_upward():
    """zeros -> par -> zeros round-trip on an upward-sloping curve."""
    mats = np.arange(0.5, 10.5, 0.5)
    zeros = 0.02 + 0.003 * mats
    pars = zeros_to_par(mats, zeros)
    recovered = par_to_zeros(mats, pars)
    assert np.max(np.abs(recovered - zeros)) < 1e-12


def test_zeros_to_par_round_trip_inverted():
    """zeros -> par -> zeros round-trip on an inverted curve."""
    mats = np.arange(0.5, 10.5, 0.5)
    zeros = 0.055 - 0.002 * mats
    pars = zeros_to_par(mats, zeros)
    recovered = par_to_zeros(mats, pars)
    assert np.max(np.abs(recovered - zeros)) < 1e-12


def test_par_below_zero_on_upward_curve():
    """Par yields lie below zero rates when the curve slopes upward.

    The first maturity is excluded: with only one cash flow, par yield == zero rate exactly.
    """
    mats = np.arange(0.5, 10.5, 0.5)
    zeros = 0.02 + 0.003 * mats
    pars = zeros_to_par(mats, zeros)
    assert abs(pars[0] - zeros[0]) < 1e-12   # equality at first period
    assert np.all(pars[1:] < zeros[1:])


def test_par_above_zero_on_inverted_curve():
    """Par yields lie above zero rates when the curve is inverted.

    The first maturity is excluded: with only one cash flow, par yield == zero rate exactly.
    """
    mats = np.arange(0.5, 10.5, 0.5)
    zeros = 0.055 - 0.002 * mats
    pars = zeros_to_par(mats, zeros)
    assert abs(pars[0] - zeros[0]) < 1e-12   # equality at first period
    assert np.all(pars[1:] > zeros[1:])
