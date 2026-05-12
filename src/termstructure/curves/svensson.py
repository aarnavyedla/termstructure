import numpy as np


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
