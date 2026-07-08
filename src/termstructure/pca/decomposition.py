from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from termstructure.pca.panel import DEFAULT_MATURITIES, _col


def _sign_corrected_components(raw_components: np.ndarray) -> np.ndarray:
    """
    Return a copy of raw_components with a canonical sign convention:
      PC1: all loadings positive  (parallel shift up  → positive level score)
      PC2: 30Y loading positive   (steepening         → positive slope score)
      PC3: 1Y  loading positive   (curvature up       → positive curvature score)

    sklearn eigenvectors are sign-arbitrary; this makes cumulative scores move
    in the same direction as their standard observable proxies.
    """
    components: np.ndarray = np.array(raw_components)
    if len(components) > 0 and components[0].mean() < 0:
        components[0] *= -1
    if len(components) > 1 and components[1, -1] < 0:
        components[1] *= -1
    if len(components) > 2 and components[2, 0] < 0:
        components[2] *= -1
    return components


def fit_pca(n_components: int | None = None) -> tuple[Any, np.ndarray, pd.DatetimeIndex]:
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
            X     — (n_dates, 8) change matrix in bp used for fitting
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

    Uses an uncentered projection (X @ components.T rather than pca.transform)
    so that cumulative sums track yield levels rather than de-trended changes.
    Sign convention is enforced via _sign_corrected_components.

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

    components = _sign_corrected_components(pca.components_)
    scores = X @ components.T  # uncentered: cumsum tracks yield levels

    result = pd.DataFrame({"date": dates})
    for i in range(scores.shape[1]):
        result[f"score_{i + 1}"] = scores[:, i]

    return result


def save_pca_outputs() -> dict[str, Any]:
    """
    Persist PCA outputs to parquet for use in downstream weeks.

    Saves two files:
        data/processed/pca_loadings.parquet
            Columns: pc, var_ratio, y_1, y_2, y_3, y_5, y_7, y_10, y_20, y_30
            One row per component; loadings are sign-corrected.

        data/processed/factor_scores.parquet
            Columns: date, score_1, ..., score_N
            Daily factor scores in bp (same as compute_factor_scores()).

    Returns:
        dict with keys: loadings_path, scores_path, n_components,
                        n_score_rows, var_3pc
    """
    from pathlib import Path

    import joblib

    model_path    = Path(__file__).resolve().parents[3] / "data/processed/pca_model.joblib"
    loadings_path = Path(__file__).resolve().parents[3] / "data/processed/pca_loadings.parquet"
    scores_path   = Path(__file__).resolve().parents[3] / "data/processed/factor_scores.parquet"

    pca = joblib.load(model_path)
    components = _sign_corrected_components(pca.components_)

    # --- loadings parquet ---
    mat_cols = [f"y_{_col(float(m))}" for m in DEFAULT_MATURITIES]
    loadings_df = pd.DataFrame(components, columns=mat_cols)
    loadings_df.insert(0, "pc", range(1, len(components) + 1))
    loadings_df.insert(1, "var_ratio", pca.explained_variance_ratio_)
    loadings_df.to_parquet(loadings_path, index=False)

    # --- factor scores parquet ---
    scores_df = compute_factor_scores()
    scores_df.to_parquet(scores_path, index=False)

    return {
        "loadings_path": loadings_path,
        "scores_path":   scores_path,
        "n_components":  len(components),
        "n_score_rows":  len(scores_df),
        "var_3pc":       float(pca.explained_variance_ratio_[:3].sum()),
    }
