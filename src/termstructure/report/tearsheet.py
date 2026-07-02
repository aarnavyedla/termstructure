"""Tearsheet generator for the factor-neutral RV strategy backtest."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

from termstructure.backtest.pnl import (
    compute_leg_pnl,
    compute_portfolio_pnl,
    compute_transaction_costs,
    _detect_streaks,
)

_DATA = Path(__file__).resolve().parents[3] / "data" / "processed"

# Cost assumption used in tearsheet outputs unless overridden.
# 0.02bp per leg per event → 0.16bp round-trip for 4-leg position.
DEFAULT_COST_BPS     = 0.02
DEFAULT_POS_SCALE    = 0.80


# ─── Statistics ───────────────────────────────────────────────────────────────

def compute_stats(
    pnl: pd.Series,
    label: str = "",
    n_years: float | None = None,
) -> dict:
    """Compute standard performance metrics for a daily P&L series.

    Args:
        pnl:     daily P&L in dollars (indexed by date or integer)
        label:   optional name for display purposes
        n_years: calendar years spanned; inferred from date index if omitted

    Returns:
        dict with keys: label, total_pnl, ann_pnl, sharpe, max_dd, calmar,
                        hit_rate, n_active, n_zero
    """
    active = pnl[pnl != 0]
    n_active = len(active)

    if n_active < 2:
        return {"label": label, "total_pnl": pnl.sum(), "ann_pnl": 0,
                "sharpe": np.nan, "max_dd": 0, "calmar": np.nan,
                "hit_rate": np.nan, "n_active": n_active, "n_zero": len(pnl) - n_active}

    sharpe = active.mean() / active.std() * np.sqrt(252)
    cum = pnl.cumsum()
    max_dd = (cum - cum.cummax()).min()

    if n_years is None and hasattr(pnl.index, "min"):
        try:
            n_years = (pnl.index.max() - pnl.index.min()).days / 365.25
        except TypeError:
            n_years = n_active / 252

    ann_pnl = pnl.sum() / n_years if (n_years and n_years > 0) else active.mean() * 252
    calmar  = ann_pnl / abs(max_dd) if max_dd != 0 else np.nan

    return {
        "label":    label,
        "total_pnl": pnl.sum(),
        "ann_pnl":   ann_pnl,
        "sharpe":    sharpe,
        "max_dd":    max_dd,
        "calmar":    calmar,
        "hit_rate":  (active > 0).mean(),
        "n_active":  n_active,
        "n_zero":    len(pnl) - n_active,
    }


def print_stats_table(
    cost_bps: float = DEFAULT_COST_BPS,
    position_scale: float = DEFAULT_POS_SCALE,
) -> None:
    """Print a formatted performance summary to stdout.

    Shows gross P&L, scaled P&L, and net-of-cost P&L side by side.
    """
    positions = pd.read_parquet(_DATA / "portfolio_positions.parquet")
    positions["date"] = pd.to_datetime(positions["date"])

    daily = compute_portfolio_pnl()
    daily["date"] = pd.to_datetime(daily["date"])

    tc = compute_transaction_costs(positions, cost_bps=cost_bps, position_scale=position_scale)
    merged = daily[["date", "total_pnl"]].merge(
        tc.rename(columns={"transaction_cost": "_tc"}), on="date", how="left"
    ).fillna(0)
    merged["scaled_pnl"] = merged["total_pnl"] * position_scale
    merged["net_pnl"]    = merged["scaled_pnl"] - merged["_tc"]

    streaks = _detect_streaks(positions)
    streaks["duration_days"] = (streaks["exit_date"] - streaks["entry_date"]).dt.days + 1
    n_years = (daily["date"].max() - daily["date"].min()).days / 365.25
    rts_per_yr = len(streaks) / n_years

    s_gross  = compute_stats(merged.set_index("date")["total_pnl"],    "Gross",  n_years)
    s_scaled = compute_stats(merged.set_index("date")["scaled_pnl"], f"Scaled (×{position_scale})", n_years)
    s_net    = compute_stats(merged.set_index("date")["net_pnl"],      f"Net ({cost_bps}bp/leg/event)", n_years)

    w = 22
    hdr = f"{'Metric':<30} {'Gross':>{w}} {'Scaled':>{w}} {'Net':>{w}}"
    sep = "-" * len(hdr)
    print()
    print(f"  Factor-neutral RV strategy -- backtest tearsheet")
    print(f"  Cost assumption: {cost_bps}bp per leg per event  |  Position scale: {position_scale}x")
    print(sep)
    print(hdr)
    print(sep)

    def row(name, key, fmt):
        vals = [s_gross[key], s_scaled[key], s_net[key]]
        cells = [fmt.format(v) for v in vals]
        print(f"  {name:<28} {cells[0]:>{w}} {cells[1]:>{w}} {cells[2]:>{w}}")

    row("Total P&L ($)",        "total_pnl", "${:>,.0f}")
    row("Ann. avg P&L ($)",     "ann_pnl",   "${:>,.0f}")
    row("Sharpe (active days)", "sharpe",    "{:>.2f}")
    row("Max drawdown ($)",     "max_dd",    "${:>,.0f}")
    row("Calmar ratio",         "calmar",    "{:>.2f}")
    row("Hit rate",             "hit_rate",  "{:>.1%}")
    row("Active days",          "n_active",  "{:>,.0f}")

    print(sep)
    print(f"  {'Backtest span (years)':<28} {n_years:>{w}.1f}")
    print(f"  {'Round trips (total)':<28} {len(streaks):>{w},}")
    print(f"  {'Round trips / year':<28} {rts_per_yr:>{w}.1f}")
    print(f"  {'Avg holding (days)':<28} {streaks['duration_days'].mean():>{w}.1f}")
    tc_total = tc["transaction_cost"].sum()
    print(f"  {'Total transaction cost ($)':<28} {'${:>,.0f}'.format(tc_total):>{w}}")
    print(sep)
    print()
    print()


# ─── Plots ────────────────────────────────────────────────────────────────────

def _monthly_returns(pnl: pd.Series, dates: pd.Series) -> pd.DataFrame:
    """Pivot daily P&L into a (year × month) matrix of monthly sums."""
    df = pd.DataFrame({"date": dates, "pnl": pnl.values})
    df["year"]  = df["date"].dt.year
    df["month"] = df["date"].dt.month
    pivot = df.groupby(["year", "month"])["pnl"].sum().unstack(fill_value=0)
    pivot.columns = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"][:len(pivot.columns)]
    return pivot


def plot_tearsheet(
    cost_bps: float = DEFAULT_COST_BPS,
    position_scale: float = DEFAULT_POS_SCALE,
    figsize: tuple = (15, 20),
) -> plt.Figure:
    """Generate a 6-panel tearsheet figure.

    Panels:
      1. Equity curve: gross vs net cumulative P&L
      2. Monthly P&L heatmap (gross)
      3. Daily P&L distribution (net)
      4. Rolling 12-month Sharpe (net, active days)
      5. Attribution: cumulative P&L by maturity bucket (signal legs, gross)
      6. Attribution: cumulative P&L by direction (signal legs, gross)

    Returns the matplotlib Figure object.
    """
    positions = pd.read_parquet(_DATA / "portfolio_positions.parquet")
    positions["date"] = pd.to_datetime(positions["date"])

    daily = compute_portfolio_pnl()
    daily["date"] = pd.to_datetime(daily["date"])

    tc = compute_transaction_costs(positions, cost_bps=cost_bps, position_scale=position_scale)
    merged = daily[["date", "total_pnl"]].merge(
        tc.rename(columns={"transaction_cost": "_tc"}), on="date", how="left"
    ).fillna(0).sort_values("date").reset_index(drop=True)
    merged["scaled_pnl"] = merged["total_pnl"] * position_scale
    merged["net_pnl"]    = merged["scaled_pnl"] - merged["_tc"]
    merged["cum_gross"]  = merged["scaled_pnl"].cumsum()
    merged["cum_net"]    = merged["net_pnl"].cumsum()

    leg_pnl = compute_leg_pnl()
    sig_legs = leg_pnl[leg_pnl["leg_type"] == "signal"].copy()
    sig_legs["date"] = pd.to_datetime(sig_legs["date"])

    fig = plt.figure(figsize=figsize)
    gs  = gridspec.GridSpec(4, 2, figure=fig, hspace=0.45, wspace=0.35)
    ax_curve = fig.add_subplot(gs[0, :])   # full-width top
    ax_heat  = fig.add_subplot(gs[1, :])   # full-width heatmap
    ax_dist  = fig.add_subplot(gs[2, 0])
    ax_roll  = fig.add_subplot(gs[2, 1])
    ax_mat   = fig.add_subplot(gs[3, 0])
    ax_dir   = fig.add_subplot(gs[3, 1])

    # ── Panel 1: equity curve ────────────────────────────────────────────────
    ax = ax_curve
    ax.plot(merged["date"], merged["cum_gross"] / 1e3, color="steelblue",
            linewidth=1, alpha=0.7, label=f"Scaled gross (×{position_scale})")
    ax.plot(merged["date"], merged["cum_net"]   / 1e3, color="darkorange",
            linewidth=1.3, label=f"Net ({cost_bps}bp/leg/event)")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.fill_between(merged["date"], merged["cum_net"] / 1e3, 0,
                    where=merged["cum_net"] < 0, color="tomato", alpha=0.2)
    ax.set_ylabel("Cumulative P&L ($K)")
    ax.set_title("Equity curve — factor-neutral RV strategy", fontsize=12)
    ax.legend(loc="upper left")

    # ── Panel 2: monthly heatmap ─────────────────────────────────────────────
    monthly = _monthly_returns(merged["scaled_pnl"], merged["date"]) / 1e3
    ax = ax_heat
    vmax = np.percentile(np.abs(monthly.values[monthly.values != 0]), 95) if monthly.values.any() else 1
    im = ax.imshow(monthly.values, aspect="auto", cmap="RdYlGn",
                   vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(monthly.columns)))
    ax.set_xticklabels(monthly.columns, fontsize=7)
    ax.set_yticks(range(len(monthly.index)))
    ax.set_yticklabels(monthly.index, fontsize=7)
    for i in range(len(monthly.index)):
        for j in range(len(monthly.columns)):
            val = monthly.values[i, j]
            if val != 0:
                ax.text(j, i, f"{val:.0f}", ha="center", va="center",
                        fontsize=5.5, color="black")
    plt.colorbar(im, ax=ax, shrink=0.8, label="Monthly P&L ($K)")
    ax.set_title("Monthly P&L ($K) — scaled gross", fontsize=10)

    # ── Panel 3: return distribution ─────────────────────────────────────────
    ax = ax_dist
    net_active = merged.loc[merged["net_pnl"] != 0, "net_pnl"]
    ax.hist(net_active / 1e3, bins=60, color="steelblue", alpha=0.7, edgecolor="white")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.axvline(net_active.mean() / 1e3, color="darkorange", linewidth=1.2,
               linestyle="--", label=f"Mean = ${net_active.mean():,.0f}")
    ax.set_xlabel("Daily P&L ($K)")
    ax.set_ylabel("Days")
    ax.set_title("Daily return distribution (net)", fontsize=10)
    ax.legend(fontsize=8)

    # ── Panel 4: rolling 12-month Sharpe ─────────────────────────────────────
    ax = ax_roll
    net_s = merged.set_index("date")["net_pnl"]
    # Use 252-day window on active-day series
    active_net = net_s[net_s != 0]
    roll_sh = active_net.rolling(252, min_periods=126).apply(
        lambda x: x.mean() / x.std() * np.sqrt(252) if x.std() > 0 else np.nan
    )
    ax.plot(roll_sh.index, roll_sh.values, color="steelblue", linewidth=1)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.axhline(1, color="grey", linewidth=0.7, linestyle="--", label="Sharpe = 1")
    ax.set_ylabel("Rolling Sharpe (252 active days)")
    ax.set_title("Rolling 12-month Sharpe (net)", fontsize=10)
    ax.legend(fontsize=8)

    # ── Panel 5: attribution by maturity ─────────────────────────────────────
    ax = ax_mat
    for mat, color in [(3, "steelblue"), (7, "darkorange")]:
        s = (sig_legs[sig_legs["signal_maturity"] == mat]
             .set_index("date")["total_pnl"]
             .cumsum() / 1e3)
        ax.plot(s.index, s.values, label=f"{mat}Y", color=color, linewidth=1)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_ylabel("Cumulative P&L ($K)")
    ax.set_title("Attribution by maturity (signal legs, gross)", fontsize=10)
    ax.legend(fontsize=8)

    # ── Panel 6: attribution by direction ────────────────────────────────────
    ax = ax_dir
    for direction, label, color in [(1, "Long (cheap)", "steelblue"),
                                     (-1, "Short (rich)", "tomato")]:
        s = (sig_legs[sig_legs["direction"] == direction]
             .set_index("date")["total_pnl"]
             .cumsum() / 1e3)
        ax.plot(s.index, s.values, label=label, color=color, linewidth=1)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_ylabel("Cumulative P&L ($K)")
    ax.set_title("Attribution by direction (signal legs, gross)", fontsize=10)
    ax.legend(fontsize=8)

    fig.suptitle(
        f"Factor-neutral RV — tearsheet  "
        f"(cost_bps={cost_bps}, scale={position_scale}×)",
        fontsize=13, y=0.995,
    )
    return fig


def save_tearsheet(
    path: str | Path | None = None,
    cost_bps: float = DEFAULT_COST_BPS,
    position_scale: float = DEFAULT_POS_SCALE,
) -> Path:
    """Save tearsheet to a PNG file. Returns the output path."""
    if path is None:
        out = Path(__file__).resolve().parents[3] / "docs" / "tearsheet.png"
    else:
        out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig = plot_tearsheet(cost_bps=cost_bps, position_scale=position_scale)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved tearsheet -> {out}")
    return out
