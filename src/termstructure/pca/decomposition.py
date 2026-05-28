import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from termstructure.pca.panel import DEFAULT_MATURITIES, _col


def fit_pca(n_components: int | None = None) -> tuple:
    """
    Fit PCA to the daily yield-change panel.

    Uses the covariance matrix (sklearn centers but does not scale by default),
    which preserves the natural volatility differences across maturities.

    Args:
        n_components: components to retain. Defaults to None = full rank
                      (one per maturity in DEFAULT_MATURITIES).

    Returns:
        (pca, X, dates) where:
            pca   — fitted sklearn PCA object
            X     — (n_dates, 10) change matrix in bp used for fitting
            dates — DatetimeIndex aligned to X rows
        Saves fitted model to data/processed/pca_model.joblib.
    """
    from pathlib import Path
    import joblib

    panel_path = Path(__file__).resolve().parents[3] / "data/processed/zero_panel.parquet"
    model_path = Path(__file__).resolve().parents[3] / "data/processed/pca_model.joblib"

    panel = pd.read_parquet(panel_path)
    panel["date"] = pd.to_datetime(panel["date"])
    panel = panel.sort_values("date").reset_index(drop=True)

    change_cols = [f"dy_{_col(m)}" for m in DEFAULT_MATURITIES]
    sub   = panel.iloc[1:]  # drop first NaN row
    X     = sub[change_cols].to_numpy(dtype=float)
    dates = pd.DatetimeIndex(sub["date"])

    if n_components is None:
        n_components = X.shape[1]
    pca = PCA(n_components=n_components)
    pca.fit(X)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pca, model_path)

    return pca, X, dates


def compute_factor_scores() -> pd.DataFrame:
    """
    Project daily yield changes onto the fitted PCA components.

    Loads pca_model.joblib and zero_panel.parquet, returns one score per PC
    per date. Each score is in bp — it is how much that factor moved that day.

    Returns:
        DataFrame with columns: date, score_1, ..., score_N
    """
    from pathlib import Path
    import joblib

    model_path = Path(__file__).resolve().parents[3] / "data/processed/pca_model.joblib"
    panel_path = Path(__file__).resolve().parents[3] / "data/processed/zero_panel.parquet"

    pca = joblib.load(model_path)

    panel = pd.read_parquet(panel_path)
    panel["date"] = pd.to_datetime(panel["date"])
    panel = panel.sort_values("date").reset_index(drop=True)

    change_cols = [f"dy_{_col(m)}" for m in DEFAULT_MATURITIES]
    sub   = panel.iloc[1:]  # drop first NaN row
    X     = sub[change_cols].to_numpy(dtype=float)
    dates = pd.DatetimeIndex(sub["date"])

    # Enforce sign convention so cumulative scores move with their proxies:
    #   PC1: all loadings positive  (parallel shift up  → positive level score)
    #   PC2: 30Y loading positive   (steepening         → positive slope score)
    #   PC3: 1Y  loading positive   (curvature up       → positive curvature score)
    # sklearn eigenvectors are sign-arbitrary; flip as needed.
    components = pca.components_.copy()
    if len(components) > 0 and components[0].mean() < 0:
        components[0] *= -1
    if len(components) > 1 and components[1, -1] < 0:
        components[1] *= -1
    if len(components) > 2 and components[2, 0] < 0:
        components[2] *= -1

    scores = X @ components.T  # uncentered: cumsum tracks yield levels

    result = pd.DataFrame({"date": dates})
    for i in range(scores.shape[1]):
        result[f"score_{i + 1}"] = scores[:, i]

    return result
