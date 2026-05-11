import numpy as np
from numpy.typing import NDArray


def zero_to_discount(rate: float, maturity: float, freq: int = 2) -> float:
    """Convert a zero (spot) rate to a discount factor.

    rate:     annual zero rate, e.g. 0.05 for 5%
    maturity: time in years
    freq:     compounding frequency per year (2 = semi-annual)
    """
    return (1 + rate / freq) ** (-maturity * freq)


def discount_to_zero(df: float, maturity: float, freq: int = 2) -> float:
    """Convert a discount factor to an annual zero rate."""
    return freq * (df ** (-1.0 / (maturity * freq)) - 1)


def zeros_to_par(
    maturities: NDArray,
    zero_rates: NDArray,
    freq: int = 2,
) -> NDArray:
    """Compute par yields from a zero curve sampled at coupon-period maturities.

    maturities: 1-D array of times in years at each coupon date, e.g.
                [0.5, 1.0, 1.5, ..., 10.0] for a semi-annual 10Y grid
    zero_rates: corresponding annual zero rates (decimal)

    Returns an array of annual par yields, one per maturity.
    Each par yield uses all discount factors up to that maturity.
    """
    maturities = np.asarray(maturities, dtype=float)
    zero_rates = np.asarray(zero_rates, dtype=float)
    n = len(maturities)

    dfs = zero_to_discount(zero_rates, maturities, freq)
    par_yields = np.empty(n)

    for i in range(n):
        df_sum = dfs[: i + 1].sum()
        par_yields[i] = freq * (1 - dfs[i]) / df_sum

    return par_yields


def par_to_zeros(
    maturities: NDArray,
    par_yields: NDArray,
    freq: int = 2,
) -> NDArray:
    """Bootstrap zero rates from par yields (inverse of zeros_to_par).

    maturities: 1-D array of times in years at each coupon date
    par_yields: annual par yields (decimal), same length

    Returns an array of annual zero rates bootstrapped from the par curve.
    """
    maturities = np.asarray(maturities, dtype=float)
    par_yields = np.asarray(par_yields, dtype=float)
    n = len(maturities)

    dfs = np.empty(n)
    zero_rates = np.empty(n)

    for i in range(n):
        c = par_yields[i]
        # sum of discount factors for all prior coupon dates
        prior_df_sum = dfs[:i].sum()
        # algebraic bootstrap: solve 1 = (c/freq)*(prior_sum + df_i) + df_i
        dfs[i] = (1 - (c / freq) * prior_df_sum) / (1 + c / freq)
        zero_rates[i] = discount_to_zero(dfs[i], maturities[i], freq)

    return zero_rates
