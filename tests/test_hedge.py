"""Unit tests for factor-neutral hedge math."""

import numpy as np
import pytest

from termstructure.risk.hedge import (
    bond_factor_exposures,
    build_hedge,
    compute_hedge_weights,
)

# Flat yield curve at 4% over 10-point maturity grid matching PCA panel
_CURVE_MATS = np.array([0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0], dtype=float)
_CURVE_RATES = np.full(10, 0.04)
_KRD_BUCKETS = list(_CURVE_MATS)

# Realistic level/slope/curvature loadings (rows = factors, cols = maturity buckets)
# Chosen so that the 3×3 hedge matrix for 2Y/5Y/10Y instruments is well-conditioned.
_LOADINGS = np.array([
    [0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10],   # level
    [-0.35, -0.25, -0.15, -0.05, 0.10, 0.20, 0.30, 0.35, 0.40, 0.45],  # slope
    [0.30,  0.35,  0.25,  0.10, -0.10, -0.25, -0.30, -0.20, -0.10, 0.0],  # curvature
])


def test_compute_hedge_weights_identity():
    """3×3 identity matrix: weights equal negative exposures."""
    target = np.array([1.0, 2.0, 3.0])
    hedge_exp = np.eye(3)
    w = compute_hedge_weights(target, hedge_exp)
    np.testing.assert_allclose(w, -target)


def test_compute_hedge_weights_2x_diagonal():
    """2× diagonal: weights are half the negative exposure."""
    target = np.array([4.0, 6.0, 8.0])
    hedge_exp = 2.0 * np.eye(3)
    w = compute_hedge_weights(target, hedge_exp)
    np.testing.assert_allclose(w, -target / 2.0, rtol=1e-10)


def test_compute_hedge_weights_ill_conditioned_raises():
    """Ill-conditioned matrix raises ValueError."""
    H = np.array([[1.0, 1.0, 0.0],
                  [1.0, 1.0 + 1e-15, 0.0],
                  [0.0, 0.0, 1.0]])
    with pytest.raises(ValueError, match="ill-conditioned"):
        compute_hedge_weights(np.array([1.0, 0.0, 0.0]), H)


def test_bond_factor_exposures_shape():
    """Returns a (n_factors,) array."""
    exp = bond_factor_exposures(
        coupon=0.04,
        maturity_years=5.0,
        curve_mats=_CURVE_MATS,
        curve_rates=_CURVE_RATES,
        loadings=_LOADINGS,
        krd_buckets=_KRD_BUCKETS,
    )
    assert exp.shape == (3,)


def test_bond_factor_exposures_longer_bond_higher_level():
    """Longer maturity bond has larger level exposure (higher total duration)."""
    exp_5 = bond_factor_exposures(0.04, 5.0, _CURVE_MATS, _CURVE_RATES, _LOADINGS, _KRD_BUCKETS)
    exp_10 = bond_factor_exposures(0.04, 10.0, _CURVE_MATS, _CURVE_RATES, _LOADINGS, _KRD_BUCKETS)
    assert exp_10[0] > exp_5[0]


def test_build_hedge_wrong_n_instruments():
    """Passing 2 hedge instruments when 3 factors are needed raises ValueError."""
    with pytest.raises(ValueError, match="Need exactly 3"):
        build_hedge(
            target_coupon=0.04,
            target_maturity=7.0,
            hedge_coupons=[0.04, 0.04],
            hedge_maturities=[2.0, 10.0],
            curve_mats=_CURVE_MATS,
            curve_rates=_CURVE_RATES,
            loadings=_LOADINGS,
            krd_buckets=_KRD_BUCKETS,
        )


def test_build_hedge_returns_expected_keys():
    """build_hedge returns dict with all expected keys."""
    result = build_hedge(
        target_coupon=0.04,
        target_maturity=7.0,
        hedge_coupons=[0.04, 0.04, 0.04],
        hedge_maturities=[2.0, 5.0, 10.0],
        curve_mats=_CURVE_MATS,
        curve_rates=_CURVE_RATES,
        loadings=_LOADINGS,
        krd_buckets=_KRD_BUCKETS,
    )
    assert set(result.keys()) == {
        "target_exposures", "hedge_exposures", "hedge_weights", "condition_number"
    }


def test_build_hedge_weights_shape():
    """Hedge weights vector has n_factors elements and condition_number is a float."""
    result = build_hedge(
        target_coupon=0.04,
        target_maturity=7.0,
        hedge_coupons=[0.04, 0.04, 0.04],
        hedge_maturities=[2.0, 5.0, 10.0],
        curve_mats=_CURVE_MATS,
        curve_rates=_CURVE_RATES,
        loadings=_LOADINGS,
        krd_buckets=_KRD_BUCKETS,
    )
    assert result["hedge_weights"].shape == (3,)
    assert isinstance(result["condition_number"], float)


def test_build_hedge_neutralizes_exposures():
    """target + hedge_exposures @ hedge_weights ≈ 0 (perfect factor neutrality)."""
    result = build_hedge(
        target_coupon=0.04,
        target_maturity=7.0,
        hedge_coupons=[0.04, 0.04, 0.04],
        hedge_maturities=[2.0, 5.0, 10.0],
        curve_mats=_CURVE_MATS,
        curve_rates=_CURVE_RATES,
        loadings=_LOADINGS,
        krd_buckets=_KRD_BUCKETS,
    )
    residual = result["target_exposures"] + result["hedge_exposures"] @ result["hedge_weights"]
    np.testing.assert_allclose(residual, 0.0, atol=1e-8)
