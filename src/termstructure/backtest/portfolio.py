import numpy as np
import pandas as pd
from pathlib import Path

from termstructure.bonds.pricing import dv01 as _dv01
from termstructure.curves.svensson import svensson_zero_rate
from termstructure.risk.hedge import build_hedge

_ROOT = Path(__file__).resolve().parents[3]
_DATA = _ROOT / "data" / "processed"

SIGNAL_MATURITIES = [3, 7]
HEDGE_MATURITIES = [2.0, 5.0, 10.0]
_FINE_GRID = np.array([0.5, 1, 2, 3, 5, 7, 10, 15, 20, 25, 30], dtype=float)


def _load_pca_loadings() -> np.ndarray:
    df = pd.read_parquet(_DATA / "pca_loadings.parquet")
    return df.iloc[:3, 2:].to_numpy()


def _sv_curve(sv: np.ndarray) -> np.ndarray:
    return np.array([svensson_zero_rate(t, *sv) for t in _FINE_GRID])


def _position_legs(
    signal_mat: int,
    direction: int,
    z_score: float,
    sv: np.ndarray,
    loadings: np.ndarray,
    target_dv01: float,
    date,
) -> list[dict]:
    ytm = svensson_zero_rate(float(signal_mat), *sv)
    dv01_per_dollar = _dv01(ytm, float(signal_mat), ytm, face=1.0)
    signal_face = target_dv01 / dv01_per_dollar

    curve_rates = _sv_curve(sv)
    hedge_coupons = [svensson_zero_rate(m, *sv) for m in HEDGE_MATURITIES]

    result = build_hedge(
        ytm, float(signal_mat),
        hedge_coupons, HEDGE_MATURITIES,
        _FINE_GRID, curve_rates,
        loadings,
    )
    weights = result["hedge_weights"]

    legs = [{
        "date": date,
        "signal_maturity": signal_mat,
        "leg_maturity": float(signal_mat),
        "leg_type": "signal",
        "direction": direction,
        "z_score": z_score,
        "notional": direction * signal_face,
        "dv01": target_dv01,
    }]

    for h_mat, w in zip(HEDGE_MATURITIES, weights):
        h_ytm = svensson_zero_rate(h_mat, *sv)
        h_notional = direction * w * signal_face
        h_dv01_per_dollar = _dv01(h_ytm, h_mat, h_ytm, face=1.0)
        legs.append({
            "date": date,
            "signal_maturity": signal_mat,
            "leg_maturity": h_mat,
            "leg_type": "hedge",
            "direction": direction,
            "z_score": float("nan"),
            "notional": h_notional,
            "dv01": abs(h_notional) * h_dv01_per_dollar,
        })

    return legs


def construct_portfolio(
    date,
    sv: np.ndarray,
    today_signals: pd.DataFrame,
    loadings: np.ndarray | None = None,
    target_dv01: float = 10_000,
    n_per_side: int = 5,
) -> pd.DataFrame:
    if loadings is None:
        loadings = _load_pca_loadings()

    today = today_signals[today_signals["maturity"].isin(SIGNAL_MATURITIES)]
    active = today[today["direction"] != 0]
    longs  = active[active["direction"] ==  1].sort_values("z_score", ascending=False).head(n_per_side)
    shorts = active[active["direction"] == -1].sort_values("z_score", ascending=True).head(n_per_side)
    selected = pd.concat([longs, shorts])

    all_legs: list[dict] = []
    for _, row in selected.iterrows():
        all_legs.extend(_position_legs(
            int(row["maturity"]),
            int(row["direction"]),
            float(row["z_score"]),
            sv, loadings, target_dv01, date,
        ))

    if not all_legs:
        return pd.DataFrame(columns=[
            "date", "signal_maturity", "leg_maturity", "leg_type",
            "direction", "z_score", "notional", "dv01",
        ])
    return pd.DataFrame(all_legs)


def build_portfolio_history(target_dv01: float = 10_000, n_per_side: int = 5) -> pd.DataFrame:
    signal = pd.read_parquet(_DATA / "signal_panel.parquet")
    signal["date"] = pd.to_datetime(signal["date"])

    params = pd.read_parquet(_DATA / "svensson_params.parquet")
    params["date"] = pd.to_datetime(params["date"])
    params = params.set_index("date")

    loadings = _load_pca_loadings()

    common_dates = sorted(set(signal["date"]) & set(params.index))
    all_positions: list[pd.DataFrame] = []

    for date in common_dates:
        try:
            sv = params.loc[date, ["beta0", "beta1", "beta2", "beta3", "lambda1", "lambda2"]].to_numpy(float)
        except KeyError:
            continue
        if np.any(np.isnan(sv)):
            continue

        today_sig = signal[signal["date"] == date]
        df = construct_portfolio(date, sv, today_sig, loadings, target_dv01, n_per_side)
        if not df.empty:
            all_positions.append(df)

    positions = pd.concat(all_positions, ignore_index=True)
    out = _DATA / "portfolio_positions.parquet"
    positions.to_parquet(out, index=False)
    print(f"Saved {len(positions):,} portfolio rows -> {out}")
    return positions
