import numpy as np
from scipy.optimize import brentq
from typing import List


def price_from_ytm(
    coupon: float,
    maturity_years: float,
    ytm: float,
    freq: int = 2,
    face: float = 100.0,
) -> float:
    """Present value of a bond's cash flows discounted at ytm.

    coupon: annual coupon rate, e.g. 0.05 for 5%
    ytm:    annual yield-to-maturity, e.g. 0.05 for 5%
    """
    n = int(round(maturity_years * freq))
    c = coupon * face / freq          # cash flow each period
    r = ytm / freq                    # discount rate per period
    periods = np.arange(1, n + 1)
    pv_coupons = np.sum(c / (1 + r) ** periods)
    pv_face = face / (1 + r) ** n
    return pv_coupons + pv_face


def ytm_from_price(
    price: float,
    coupon: float,
    maturity_years: float,
    freq: int = 2,
    face: float = 100.0,
) -> float:
    """Yield-to-maturity implied by an observed price, solved via bisection."""
    objective = lambda y: price_from_ytm(coupon, maturity_years, y, freq, face) - price
    return brentq(objective, 1e-4, 0.30)


def modified_duration(
    coupon: float,
    maturity_years: float,
    ytm: float,
    freq: int = 2,
    face: float = 100.0,
    bump: float = 1e-4,
) -> float:
    """Modified duration via symmetric 1bp numerical bump."""
    p_up   = price_from_ytm(coupon, maturity_years, ytm + bump, freq, face)
    p_down = price_from_ytm(coupon, maturity_years, ytm - bump, freq, face)
    p      = price_from_ytm(coupon, maturity_years, ytm,        freq, face)
    return (p_down - p_up) / (2 * p * bump)


def dv01(
    coupon: float,
    maturity_years: float,
    ytm: float,
    freq: int = 2,
    face: float = 100.0,
) -> float:
    """Dollar value of a basis point: price change for a 1bp yield move."""
    md = modified_duration(coupon, maturity_years, ytm, freq, face)
    p  = price_from_ytm(coupon, maturity_years, ytm, freq, face)
    return md * p * 1e-4


def price_from_curve(
    coupon: float,
    maturity_years: float,
    curve_mats: np.ndarray,
    curve_rates: np.ndarray,
    freq: int = 2,
    face: float = 100.0,
) -> float:
    """Price a bond by discounting each cash flow at the interpolated spot rate.

    curve_mats:  array of maturity grid points in years, e.g. [1, 2, 5, 10, 30]
    curve_rates: array of annual zero rates (decimal), same length as curve_mats
    """
    n = int(round(maturity_years * freq))
    cf_times = np.arange(1, n + 1) / freq
    cash_flows = np.full(n, coupon * face / freq)
    cash_flows[-1] += face

    spot_rates = np.interp(cf_times, curve_mats, curve_rates)
    discount_factors = (1 + spot_rates / freq) ** -(cf_times * freq)
    return float(np.dot(cash_flows, discount_factors))


def _tent_bump(
    curve_mats: np.ndarray,
    bucket_points: List[float],
    k: int,
    bump: float,
) -> np.ndarray:
    """Tent function: bump of `bump` at bucket_points[k], linearly decays to 0 at neighbors."""
    values = np.zeros(len(bucket_points))
    values[k] = 1.0
    left_val  = values[0]   # flat extension left of first bucket
    right_val = values[-1]  # flat extension right of last bucket
    return bump * np.interp(curve_mats, bucket_points, values,
                            left=left_val, right=right_val)


def key_rate_durations(
    coupon: float,
    maturity_years: float,
    curve_mats: np.ndarray,
    curve_rates: np.ndarray,
    bucket_points: List[float] = [2, 5, 10, 30],
    freq: int = 2,
    face: float = 100.0,
    bump: float = 1e-4,
) -> dict:
    """Sensitivity of price (in duration units) to a 1bp tent-function bump at each bucket.

    Returns a dict mapping each bucket maturity to its key-rate duration.
    """
    curve_mats  = np.asarray(curve_mats,  dtype=float)
    curve_rates = np.asarray(curve_rates, dtype=float)
    p = price_from_curve(coupon, maturity_years, curve_mats, curve_rates, freq, face)

    krds = {}
    for k, bp in enumerate(bucket_points):
        bump_vec = _tent_bump(curve_mats, bucket_points, k, bump)
        p_up   = price_from_curve(coupon, maturity_years, curve_mats, curve_rates + bump_vec, freq, face)
        p_down = price_from_curve(coupon, maturity_years, curve_mats, curve_rates - bump_vec, freq, face)
        krds[bp] = (p_down - p_up) / (2 * p * bump)

    return krds
