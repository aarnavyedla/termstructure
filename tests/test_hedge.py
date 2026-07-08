"""Unit tests for factor-neutral hedge math."""

from pathlib import Path

import numpy as np
import pandas as pd
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


# ─── Regression: stale PCA loadings in hedge construction (Week 7 Day 3 bug) ──

_PCA_LOADINGS  = Path("data/processed/pca_loadings.parquet")
_POSITIONS     = Path("data/processed/portfolio_positions.parquet")
_SV_PARAMS     = Path("data/processed/svensson_params.parquet")
_SIGNAL_PANEL  = Path("data/processed/signal_panel.parquet")

_requires_portfolio_data = pytest.mark.skipif(
    not (_PCA_LOADINGS.exists() and _POSITIONS.exists()
         and _SV_PARAMS.exists() and _SIGNAL_PANEL.exists()),
    reason="portfolio data files not on disk",
)


@_requires_portfolio_data
def test_hedge_weights_match_current_loadings() -> None:
    """portfolio_positions.parquet was built with the current pca_loadings.parquet.

    Week 7 Day 3 bug: the PCA was re-run after the portfolio was first built.
    portfolio_positions.parquet still reflected old loadings that embedded a
    2Y/3Y slope bet (concentrated hedge weights, ~96% into the 2Y instrument
    for some positions). The fix was to rebuild portfolio_positions.parquet.

    This test picks a specific date (2011-04-28, 7Y long) where both
    portfolio_positions.parquet and signal_panel.parquet have data, re-runs
    construct_portfolio with the CURRENT pca_loadings.parquet, and asserts
    the hedge notionals match to 0.1%.

    If portfolio_positions.parquet were rebuilt with different (stale) loadings,
    the hedge weights would differ — e.g. the 2Y notional sign would flip and
    the relative magnitudes across 2Y/5Y/10Y would change materially.
    """
    from termstructure.backtest.portfolio import (
        _load_pca_loadings,
        construct_portfolio,
    )

    # ── Load all inputs ──────────────────────────────────────────────────────
    positions = pd.read_parquet(_POSITIONS)
    positions["date"] = pd.to_datetime(positions["date"])

    params = pd.read_parquet(_SV_PARAMS)
    params["date"] = pd.to_datetime(params["date"])
    params = params.set_index("date")

    signal = pd.read_parquet(_SIGNAL_PANEL)
    signal["date"] = pd.to_datetime(signal["date"])

    loadings = _load_pca_loadings()

    # ── Sample date: 2011-04-28, 7Y long ────────────────────────────────────
    # Chosen because it exists in both portfolio_positions.parquet and
    # signal_panel.parquet (signal panel starts 2010-02-16; positions go back
    # to 1971 from an earlier build, so only the overlap is re-computable).
    # Balanced stored weights (2Y -29%, 5Y -38%, 10Y -49%) make sign/magnitude
    # mismatches unambiguous — stale loadings produced a sign flip on 2Y.
    target_date = pd.Timestamp("2011-04-28")
    stored = positions[
        (positions["date"] == target_date)
        & (positions["signal_maturity"] == 7)
        & (positions["direction"] == 1)
    ]
    if stored.empty:
        pytest.skip(f"No 7Y long position on {target_date}; check data files")

    # ── Re-compute the portfolio for that date with current loadings ─────────
    sv = params.loc[
        target_date, ["beta0", "beta1", "beta2", "beta3", "lambda1", "lambda2"]
    ].to_numpy(float)
    today_sig = signal[signal["date"] == target_date]
    recomputed = construct_portfolio(target_date, sv, today_sig, loadings)

    recomputed_hedge = recomputed[
        (recomputed["signal_maturity"] == 7) & (recomputed["leg_type"] == "hedge")
    ].sort_values("leg_maturity").reset_index(drop=True)

    stored_hedge = stored[stored["leg_type"] == "hedge"].sort_values(
        "leg_maturity"
    ).reset_index(drop=True)

    assert not recomputed_hedge.empty, (
        f"construct_portfolio returned no 7Y hedge legs for {target_date}"
    )

    # ── Core assertion: hedge notionals must match to 0.1% ──────────────────
    # Hedge weights are solved to machine precision from the loading matrix,
    # so any mismatch here means a different loadings matrix was used during
    # the portfolio build.  On the pre-rebuild file the 2Y notional had the
    # WRONG SIGN for this position — that alone causes a >100% relative error.
    for _, row in recomputed_hedge.iterrows():
        mat = row["leg_maturity"]
        stored_notional = stored_hedge.loc[
            stored_hedge["leg_maturity"] == mat, "notional"
        ].iloc[0]
        assert row["notional"] == pytest.approx(stored_notional, rel=1e-3), (
            f"Hedge notional mismatch at {mat}Y on {target_date}: "
            f"re-computed={row['notional']:.0f} vs stored={stored_notional:.0f}. "
            f"portfolio_positions.parquet was built with stale PCA loadings — "
            f"rebuild with build_portfolio_history()."
        )

    # ── Shape checks on pca_loadings.parquet itself ──────────────────────────
    # Catch a corrupt or mis-ordered pca_loadings.parquet (e.g. eigenvectors
    # in wrong order or with wrong sign convention).

    # PC1 (level): must be single-signed — a parallel shift touches all maturities
    # with the same sign (Litterman-Scheinkman 1991).
    pc1 = loadings[0]
    assert np.all(pc1 > 0) or np.all(pc1 < 0), (
        f"PC1 (level) must be single-signed; "
        f"got signs {np.sign(pc1).astype(int).tolist()}"
    )

    # PC2 (slope): must be ≥80% monotonic.  The stale-loadings bug manifested as
    # a local sign reversal at the 2Y/3Y buckets of the slope eigenvector.
    pc2 = loadings[1]
    diffs = np.diff(pc2)
    frac_same_sign = float(max((diffs > 0).mean(), (diffs < 0).mean()))
    assert frac_same_sign >= 0.80, (
        f"PC2 (slope) should be >=80% monotonic; got {frac_same_sign:.0%} — "
        "a local 2Y/3Y reversal would indicate the stale-loadings bug is back"
    )

    # PC3 (curvature): interior must have the opposite sign to the endpoints.
    pc3 = loadings[2]
    n = len(pc3)
    endpoint_sign = np.sign((pc3[0] + pc3[-1]) / 2)
    midpoint_sign = np.sign(pc3[n // 2])
    assert endpoint_sign != midpoint_sign, (
        "PC3 (curvature) endpoints and midpoint must have opposite signs"
    )
