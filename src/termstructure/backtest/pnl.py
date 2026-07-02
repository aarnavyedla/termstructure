import numpy as np
import pandas as pd
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
_DATA = _ROOT / "data" / "processed"

REPO_SPREAD   = 0.0005   # 5bp haircut above overnight rate
DAYS_PER_YEAR = 365
COST_BPS      = 0.5      # bp charged per leg per event (entry or exit); 1bp round-trip
POSITION_SCALE = 0.8     # haircut for unmodeled slippage and market impact


def compute_leg_pnl() -> pd.DataFrame:
    """Compute daily MTM + carry P&L for every portfolio leg.

    The 1-day execution lag: signal on date t is generated at EOD of t.
    We enter the position at EOD of t; the first P&L is earned from t to t+1.
    This means dy on date t uses the yield change from t to t+1 (shifted forward
    by one row).

    Returns DataFrame with columns:
        date, signal_maturity, leg_maturity, leg_type, direction, z_score,
        notional, dv01, dy_bps, ytm, repo_rate, mtm_pnl, carry_pnl, total_pnl
    """
    positions = pd.read_parquet(_DATA / "portfolio_positions.parquet")
    positions["date"] = pd.to_datetime(positions["date"])

    zp = pd.read_parquet(_DATA / "zero_panel.parquet")
    zp["date"] = pd.to_datetime(zp["date"])
    zp = zp.sort_values("date").reset_index(drop=True)

    repo = pd.read_parquet(_DATA / "repo_rate.parquet")
    repo["date"] = pd.to_datetime(repo["date"])

    mats = [1, 2, 3, 5, 7, 10, 20, 30]

    # Shift dy columns forward by 1: position on t earns change from t to t+1
    for m in mats:
        col = f"dy_{m}"
        if col in zp.columns:
            zp[col] = zp[col].shift(-1)

    # Melt zero_panel to long format: (date, leg_maturity) -> (ytm, dy_bps)
    long_pieces = []
    for m in mats:
        sub = zp[["date", f"y_{m}", f"dy_{m}"]].copy()
        sub = sub.rename(columns={f"y_{m}": "ytm", f"dy_{m}": "dy_bps"})
        sub["leg_maturity"] = float(m)
        long_pieces.append(sub)
    zp_long = pd.concat(long_pieces, ignore_index=True)

    df = positions.merge(zp_long, on=["date", "leg_maturity"], how="left")
    df = df.merge(repo, on="date", how="left")

    # MTM: short notional benefits from rising yields, so flip the sign
    df["mtm_pnl"] = -np.sign(df["notional"]) * df["dv01"] * df["dy_bps"]

    # Carry: net of financing cost; sign of notional handles long vs short
    df["carry_pnl"] = (
        df["notional"] * (df["ytm"] - (df["repo_rate"] + REPO_SPREAD)) / DAYS_PER_YEAR
    )

    df["total_pnl"] = df["mtm_pnl"] + df["carry_pnl"]

    return df


def compute_portfolio_pnl() -> pd.DataFrame:
    """Aggregate leg P&L to daily portfolio level (gross, before transaction costs).

    Returns DataFrame with columns:
        date, mtm_pnl, carry_pnl, total_pnl, cumulative_pnl, n_signal_legs
    Saves to data/processed/portfolio_pnl.parquet.
    """
    leg_pnl = compute_leg_pnl()

    agg = (
        leg_pnl
        .groupby("date")
        .agg(
            mtm_pnl=("mtm_pnl", "sum"),
            carry_pnl=("carry_pnl", "sum"),
            total_pnl=("total_pnl", "sum"),
            n_signal_legs=("leg_type", lambda x: (x == "signal").sum()),
        )
        .reset_index()
        .sort_values("date")
    )
    agg["cumulative_pnl"] = agg["total_pnl"].cumsum()

    out = _DATA / "portfolio_pnl.parquet"
    agg.to_parquet(out, index=False)
    print(f"Saved {len(agg):,} daily P&L rows -> {out}")
    return agg


# ─── Transaction costs ────────────────────────────────────────────────────────

def _detect_streaks(positions: pd.DataFrame) -> pd.DataFrame:
    """Identify entry and exit dates for every signal streak.

    A streak is a run of consecutive active days at the same (signal_maturity,
    direction). A gap of more than 5 calendar days starts a new streak.

    Returns DataFrame with columns:
        signal_maturity, direction, streak_id, entry_date, exit_date
    """
    sig = (
        positions[positions["leg_type"] == "signal"]
        [["date", "signal_maturity", "direction"]]
        .drop_duplicates()
        .sort_values(["signal_maturity", "direction", "date"])
        .copy()
    )

    def _streak_ids(grp: pd.DataFrame) -> pd.Series:
        gap = grp["date"].diff().dt.days.fillna(0) > 5
        return gap.cumsum()

    sig["streak_id"] = (
        sig.groupby(["signal_maturity", "direction"], group_keys=False)
        .apply(_streak_ids)
    )

    bounds = (
        sig.groupby(["signal_maturity", "direction", "streak_id"])["date"]
        .agg(entry_date="min", exit_date="max")
        .reset_index()
    )
    return bounds


def compute_transaction_costs(
    positions: pd.DataFrame | None = None,
    cost_bps: float = COST_BPS,
    position_scale: float = POSITION_SCALE,
) -> pd.DataFrame:
    """Compute daily transaction costs from position entry and exit events.

    Each entry (first active day of a streak) and exit (last active day)
    incurs cost_bps × dv01 × position_scale per leg.

    Args:
        positions:      portfolio_positions DataFrame (loaded from parquet if None)
        cost_bps:       basis points charged per leg per event (0.5 = 1bp round-trip)
        position_scale: multiply DV01 by this to reflect smaller effective size

    Returns:
        DataFrame with columns: date, transaction_cost
    """
    if positions is None:
        positions = pd.read_parquet(_DATA / "portfolio_positions.parquet")
        positions["date"] = pd.to_datetime(positions["date"])

    # Only charge costs on dates where zero_panel yield data exists.
    # The 1970s have positions but no yield changes (zero_panel drops those dates),
    # so gross P&L is zero there — charging costs would create spurious net-negative days.
    zp_dates = pd.read_parquet(_DATA / "zero_panel.parquet", columns=["date"])["date"]
    zp_dates = set(pd.to_datetime(zp_dates))

    streaks = _detect_streaks(positions)

    # Build a set of (date, signal_maturity) → number of events (entry + exit)
    # Same date can be both entry and exit (single-day streak) → 2 events
    entry_key = set(
        (d, m) for d, m in zip(streaks["entry_date"], streaks["signal_maturity"])
        if d in zp_dates
    )
    exit_key = set(
        (d, m) for d, m in zip(streaks["exit_date"], streaks["signal_maturity"])
        if d in zp_dates
    )

    key = list(zip(positions["date"], positions["signal_maturity"]))
    n_events = np.array([
        (k in entry_key) + (k in exit_key)
        for k in key
    ])

    cost_per_row = n_events * cost_bps * positions["dv01"].to_numpy() * position_scale
    positions = positions.copy()
    positions["_cost"] = cost_per_row

    daily = (
        positions.groupby("date")["_cost"]
        .sum()
        .reset_index()
        .rename(columns={"_cost": "transaction_cost"})
    )
    return daily


def compute_net_pnl(
    position_scale: float = POSITION_SCALE,
    cost_bps: float = COST_BPS,
) -> pd.DataFrame:
    """Gross P&L scaled by position_scale minus transaction costs.

    Returns DataFrame with columns:
        date, gross_pnl, scaled_pnl, transaction_cost, net_pnl, cumulative_net_pnl
    """
    positions = pd.read_parquet(_DATA / "portfolio_positions.parquet")
    positions["date"] = pd.to_datetime(positions["date"])

    gross = compute_portfolio_pnl()[["date", "total_pnl"]].rename(columns={"total_pnl": "gross_pnl"})
    costs = compute_transaction_costs(positions, cost_bps, position_scale)

    df = gross.merge(costs, on="date", how="left")
    df["transaction_cost"] = df["transaction_cost"].fillna(0.0)
    df["scaled_pnl"] = df["gross_pnl"] * position_scale
    df["net_pnl"] = df["scaled_pnl"] - df["transaction_cost"]
    df["cumulative_net_pnl"] = df["net_pnl"].cumsum()

    return df.sort_values("date").reset_index(drop=True)
