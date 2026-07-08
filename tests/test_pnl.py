"""Tests for the PnL engine."""

from pathlib import Path

import pandas as pd
import pytest

from termstructure.backtest.pnl import (
    _detect_streaks,
    compute_leg_pnl,
    compute_net_pnl,
    compute_portfolio_pnl,
    compute_transaction_costs,
)

_POSITIONS = Path("data/processed/portfolio_positions.parquet")
_requires_positions = pytest.mark.skipif(
    not _POSITIONS.exists(), reason="portfolio_positions.parquet not on disk"
)


# ─── _detect_streaks integration tests using production positions ──────────────

def _load_positions() -> pd.DataFrame:
    pos = pd.read_parquet("data/processed/portfolio_positions.parquet")
    pos["date"] = pd.to_datetime(pos["date"])
    return pos


@_requires_positions
def test_detect_streaks_nonempty() -> None:
    """Production positions produce at least 100 streaks."""
    pos = _load_positions()
    streaks = _detect_streaks(pos)
    assert len(streaks) > 100


@_requires_positions
def test_detect_streaks_result_columns() -> None:
    """Result has the expected columns."""
    pos = _load_positions()
    streaks = _detect_streaks(pos)
    assert set(streaks.columns) >= {"signal_maturity", "direction", "streak_id",
                                    "entry_date", "exit_date"}


@_requires_positions
def test_detect_streaks_entry_leq_exit() -> None:
    """Entry date is always ≤ exit date for every streak."""
    pos = _load_positions()
    streaks = _detect_streaks(pos)
    assert (streaks["entry_date"] <= streaks["exit_date"]).all()


@_requires_positions
def test_detect_streaks_direction_values() -> None:
    """Direction column only contains -1 and +1 (never 0 in an active streak)."""
    pos = _load_positions()
    streaks = _detect_streaks(pos)
    assert set(streaks["direction"].unique()).issubset({-1, 1})


# ─── compute_transaction_costs unit test ──────────────────────────────────────

def test_compute_transaction_costs_no_positions():
    """Zero transaction costs when positions DataFrame is empty."""
    empty = pd.DataFrame(columns=[
        "date", "signal_maturity", "direction", "leg_type", "notional", "dv01",
        "leg_maturity", "z_score",
    ])
    empty["date"] = pd.to_datetime(empty["date"])
    costs = compute_transaction_costs(empty)
    assert costs.empty or costs["transaction_cost"].sum() == 0.0


_requires_leg_data = pytest.mark.skipif(
    not _POSITIONS.exists(), reason="portfolio_positions.parquet not on disk"
)


# ─── Integration tests (read production parquets) ─────────────────────────────

@_requires_leg_data
def test_compute_leg_pnl_schema() -> None:
    """compute_leg_pnl returns DataFrame with required columns."""
    df = compute_leg_pnl()
    required = {"date", "signal_maturity", "leg_maturity", "leg_type", "direction",
                "notional", "dv01", "dy_bps", "ytm", "mtm_pnl", "carry_pnl", "total_pnl"}
    assert required.issubset(set(df.columns))


@_requires_leg_data
def test_compute_leg_pnl_nonempty() -> None:
    """compute_leg_pnl returns at least 1000 rows."""
    df = compute_leg_pnl()
    assert len(df) > 1000


@_requires_leg_data
def test_compute_leg_pnl_leg_types() -> None:
    """Only 'signal' and 'hedge' leg types appear."""
    df = compute_leg_pnl()
    assert set(df["leg_type"].unique()).issubset({"signal", "hedge"})


@_requires_leg_data
def test_compute_portfolio_pnl_schema() -> None:
    """compute_portfolio_pnl returns expected columns."""
    df = compute_portfolio_pnl()
    assert set(df.columns) >= {"date", "mtm_pnl", "carry_pnl", "total_pnl", "cumulative_pnl"}


@_requires_leg_data
def test_compute_portfolio_pnl_cumulative_correct() -> None:
    """Last value of cumulative_pnl equals sum of total_pnl."""
    df = compute_portfolio_pnl()
    assert df["cumulative_pnl"].iloc[-1] == pytest.approx(df["total_pnl"].sum(), rel=1e-6)


@_requires_leg_data
def test_compute_net_pnl_schema() -> None:
    """compute_net_pnl returns expected columns."""
    df = compute_net_pnl()
    assert set(df.columns) >= {"date", "gross_pnl", "scaled_pnl", "transaction_cost",
                                "net_pnl", "cumulative_net_pnl"}


@_requires_leg_data
def test_compute_net_pnl_net_leq_scaled() -> None:
    """Aggregate net PnL ≤ scaled PnL (transaction costs are non-negative)."""
    df = compute_net_pnl()
    assert df["net_pnl"].sum() <= df["scaled_pnl"].sum() + 1e-6
