import numpy as np
from scipy.optimize import least_squares


def svensson_zero_rate(
    tau: float | np.ndarray,
    beta0: float,
    beta1: float,
    beta2: float,
    beta3: float,
    lambda1: float,
    lambda2: float,
) -> float | np.ndarray:
    """
    Svensson (1994) zero-coupon yield at maturity tau (in years).

    Asymptotes:
      tau -> 0:   beta0 + beta1   (short rate)
      tau -> inf: beta0            (long rate)

    Parameters are in decimal form (e.g. 0.04 for 4%).
    """
    tau = np.asarray(tau, dtype=float)
    scalar = tau.ndim == 0
    tau = np.atleast_1d(tau)

    # Loading factors — (1 - exp(-t/λ)) / (t/λ)
    # At tau=0 the limit is 1.0; use np.where to avoid division by zero.
    def loading(t, lam):
        x = t / lam
        return np.where(x == 0.0, 1.0, (1.0 - np.exp(-x)) / x)

    L1 = loading(tau, lambda1)
    L2 = loading(tau, lambda2)

    result = (
        beta0
        + beta1 * L1
        + beta2 * (L1 - np.exp(-tau / lambda1))
        + beta3 * (L2 - np.exp(-tau / lambda2))
    )

    return float(result[0]) if scalar else result


def fit_svensson(
    maturities: np.ndarray,
    yields: np.ndarray,
) -> np.ndarray:
    """
    Fit Svensson parameters to observed zero yields via nonlinear least squares.

    Args:
        maturities: array of maturities in years, shape (N,)
        yields:     observed zero yields, same units as desired output, shape (N,)

    Returns:
        params: array [beta0, beta1, beta2, beta3, lambda1, lambda2]
    """
    maturities = np.asarray(maturities, dtype=float)
    yields = np.asarray(yields, dtype=float)
    weights = 1.0 / maturities  # inverse-duration proxy; down-weights noisy long end

    def residuals(p):
        return weights * (svensson_zero_rate(maturities, *p) - yields)

    # Data-driven starting guess: long-end yield ≈ beta0, slope ≈ beta1
    x0 = [yields[-1], yields[0] - yields[-1], 0.0, 0.0, 2.0, 5.0]
    bounds = (
        [-np.inf, -np.inf, -np.inf, -np.inf, 0.1, 0.1],
        [ np.inf,  np.inf,  np.inf,  np.inf, 10., 10.],
    )

    result = least_squares(residuals, x0, method='trf', bounds=bounds)
    return result.x
