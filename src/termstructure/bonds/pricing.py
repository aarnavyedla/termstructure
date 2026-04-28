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
