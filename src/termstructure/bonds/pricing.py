import numpy as np
from scipy.optimize import brentq


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
