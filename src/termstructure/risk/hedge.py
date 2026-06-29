import numpy as np

from termstructure.bonds.pricing import key_rate_durations
from termstructure.pca.panel import DEFAULT_MATURITIES

# KRD buckets must match the PCA loading columns exactly so the dot product is valid.
KRD_BUCKETS: list[float] = [float(m) for m in DEFAULT_MATURITIES]


def bond_factor_exposures(
    coupon: float,
    maturity_years: float,
    curve_mats: np.ndarray,
    curve_rates: np.ndarray,
    loadings: np.ndarray,
    krd_buckets: list[float] = KRD_BUCKETS,
) -> np.ndarray:
    """Factor exposure vector for one bond.

    exposure[k] = KRD · loadings[k]  (dot product over maturity buckets)

    Args:
        coupon:         annual coupon rate (decimal, e.g. 0.04)
        maturity_years: years to maturity
        curve_mats:     zero-rate curve maturity grid (years)
        curve_rates:    zero rates at curve_mats (decimal)
        loadings:       (n_factors, n_buckets) PCA loading matrix
        krd_buckets:    maturity buckets; must align with loadings columns

    Returns:
        (n_factors,) array of factor exposures in duration units
    """
    krd_dict = key_rate_durations(
        coupon, maturity_years, curve_mats, curve_rates,
        bucket_points=krd_buckets,
    )
    krd_vec = np.array([krd_dict[b] for b in krd_buckets])
    return loadings @ krd_vec


def compute_hedge_weights(
    target_exposures: np.ndarray,
    hedge_exposures: np.ndarray,
) -> np.ndarray:
    """Solve for hedge weights that neutralize all factor exposures.

    Finds w such that:  hedge_exposures @ w = -target_exposures

    Args:
        target_exposures: (n_factors,) factor exposure of the target bond
        hedge_exposures:  (n_factors, n_hedge) exposure matrix; must be square

    Returns:
        (n_hedge,) weights — multiply by a notional to get dollar hedge sizes

    Raises:
        ValueError if the matrix is ill-conditioned (condition number > 1e8)
        np.linalg.LinAlgError if the matrix is singular
    """
    cond = np.linalg.cond(hedge_exposures)
    if cond > 1e8:
        raise ValueError(
            f"Hedge exposure matrix is ill-conditioned (cond={cond:.2e}). "
            "Choose hedge instruments with more distinct factor profiles."
        )
    return np.linalg.solve(hedge_exposures, -target_exposures)


def build_hedge(
    target_coupon: float,
    target_maturity: float,
    hedge_coupons: list[float],
    hedge_maturities: list[float],
    curve_mats: np.ndarray,
    curve_rates: np.ndarray,
    loadings: np.ndarray,
    krd_buckets: list[float] = KRD_BUCKETS,
) -> dict:
    """Compute a 3-factor hedge for one target bond.

    Args:
        target_coupon:    annual coupon of the target bond (decimal)
        target_maturity:  maturity in years
        hedge_coupons:    coupons of the hedge instruments (one per factor)
        hedge_maturities: maturities of the hedge instruments (one per factor)
        curve_mats:       zero-rate curve grid maturities (years)
        curve_rates:      zero rates at curve_mats (decimal)
        loadings:         (n_factors, n_buckets) PCA loading matrix, sign-corrected
        krd_buckets:      maturity buckets; must align with loadings columns

    Returns:
        dict with keys:
            target_exposures  — (n_factors,) array
            hedge_exposures   — (n_factors, n_hedge) matrix
            hedge_weights     — (n_hedge,) weights; short these to hedge the target
            condition_number  — of the hedge_exposures matrix (lower is better)
    """
    n_factors = loadings.shape[0]
    if len(hedge_coupons) != n_factors or len(hedge_maturities) != n_factors:
        raise ValueError(
            f"Need exactly {n_factors} hedge instruments (one per factor); "
            f"got {len(hedge_maturities)}."
        )

    target_exp = bond_factor_exposures(
        target_coupon, target_maturity, curve_mats, curve_rates,
        loadings, krd_buckets,
    )

    hedge_exp = np.stack(
        [
            bond_factor_exposures(c, m, curve_mats, curve_rates, loadings, krd_buckets)
            for c, m in zip(hedge_coupons, hedge_maturities)
        ],
        axis=1,  # shape: (n_factors, n_hedge)
    )

    weights = compute_hedge_weights(target_exp, hedge_exp)

    return {
        "target_exposures": target_exp,
        "hedge_exposures":  hedge_exp,
        "hedge_weights":    weights,
        "condition_number": float(np.linalg.cond(hedge_exp)),
    }
