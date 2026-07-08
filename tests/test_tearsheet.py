"""Tests for tearsheet statistics and report functions."""

import matplotlib

matplotlib.use("Agg")  # non-interactive backend must be set before pyplot import

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from termstructure.report.tearsheet import (
    compute_stats,
    plot_tearsheet,
    print_stats_table,
    save_tearsheet,
)

_POSITIONS = Path("data/processed/portfolio_positions.parquet")
_requires_data = pytest.mark.skipif(
    not _POSITIONS.exists(), reason="portfolio_positions.parquet not on disk"
)


# ─── compute_stats unit tests ─────────────────────────────────────────────────

def _daily_pnl(n: int = 252, seed: int = 0, mean: float = 100.0, std: float = 500.0) -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-02", periods=n, freq="B")
    return pd.Series(rng.normal(mean, std, n), index=idx)


def test_compute_stats_keys():
    """Returns dict with all expected keys."""
    s = compute_stats(_daily_pnl())
    assert set(s.keys()) == {"label", "total_pnl", "ann_pnl", "sharpe", "max_dd", "calmar",
                              "hit_rate", "n_active", "n_zero"}


def test_compute_stats_total_pnl():
    """total_pnl equals series sum."""
    pnl = _daily_pnl()
    s = compute_stats(pnl)
    assert s["total_pnl"] == pytest.approx(pnl.sum(), rel=1e-9)


def test_compute_stats_sharpe_sign():
    """Positive-drift series has positive Sharpe; negative-drift has negative."""
    pos_pnl = _daily_pnl(mean=500.0, std=10.0)
    neg_pnl = _daily_pnl(mean=-500.0, std=10.0)
    assert compute_stats(pos_pnl)["sharpe"] > 0
    assert compute_stats(neg_pnl)["sharpe"] < 0


def test_compute_stats_max_dd_nonpositive():
    """Max drawdown is always ≤ 0."""
    s = compute_stats(_daily_pnl(seed=42))
    assert s["max_dd"] <= 0.0


def test_compute_stats_hit_rate_range():
    """Hit rate in [0, 1]."""
    s = compute_stats(_daily_pnl(seed=99))
    assert 0.0 <= s["hit_rate"] <= 1.0


def test_compute_stats_all_zero_pnl():
    """All-zero P&L: Sharpe is NaN, max_dd = 0."""
    idx = pd.date_range("2020-01-02", periods=100, freq="B")
    pnl = pd.Series(np.zeros(100), index=idx)
    s = compute_stats(pnl)
    assert np.isnan(s["sharpe"])
    assert s["max_dd"] == 0.0


def test_compute_stats_integer_index():
    """Integer-indexed series is handled without AttributeError (no .days)."""
    pnl = pd.Series([100.0, -50.0, 200.0, -30.0])  # integer RangeIndex
    s = compute_stats(pnl)
    assert np.isfinite(s["sharpe"])


def test_compute_stats_single_element():
    """Single-element series returns gracefully with Sharpe=NaN."""
    pnl = pd.Series([500.0])
    s = compute_stats(pnl)
    assert s["total_pnl"] == 500.0
    assert np.isnan(s["sharpe"])


def test_compute_stats_label_passthrough():
    """label string is passed through unchanged."""
    s = compute_stats(_daily_pnl(), label="my_strategy")
    assert s["label"] == "my_strategy"


def test_compute_stats_n_years_override():
    """Explicit n_years=1 gives 2× ann_pnl compared to n_years=2."""
    pnl = _daily_pnl(n=252)
    s1 = compute_stats(pnl, n_years=1.0)
    s2 = compute_stats(pnl, n_years=2.0)
    assert s1["ann_pnl"] == pytest.approx(2.0 * s2["ann_pnl"], rel=1e-9)


def test_compute_stats_n_active_counts_nonzero():
    """n_active counts only non-zero days; n_zero counts zero days."""
    idx = pd.date_range("2020-01-02", periods=5, freq="B")
    pnl = pd.Series([0.0, 100.0, -50.0, 0.0, 200.0], index=idx)
    s = compute_stats(pnl)
    assert s["n_active"] == 3
    assert s["n_zero"] == 2


# ─── Regression: flat-day Sharpe inflation (Week 7 Day 1 bug) ────────────────

def test_sharpe_includes_flat_days() -> None:
    """Sharpe denominator includes all calendar days, not just active (non-zero) days.

    Pre-fix code: sharpe = active.mean() / active.std() * sqrt(252)
                  where active = pnl[pnl != 0]      ← inflated

    Fixed code:   sharpe = pnl.mean() / pnl.std() * sqrt(252)   ← correct

    For a strategy trading ~14% of days the inflation factor is
    sqrt(1 / 0.14) ≈ 2.7×.  This test catches any regression that
    reintroduces active-only filtering in compute_stats.
    """
    rng = np.random.default_rng(0)
    n_total = 252 * 5        # 5-year backtest
    n_active = int(n_total * 0.14)  # 14% of days active, matching real strategy

    active_pnl = rng.normal(100, 400, n_active)

    idx = pd.date_range("2018-01-02", periods=n_total, freq="B")
    full = pd.Series(np.zeros(n_total), index=idx)
    full.iloc[:n_active] = active_pnl          # first 14% of days are active

    active_only = pd.Series(active_pnl, index=idx[:n_active])

    s = compute_stats(full)

    # Buggy computation that the old code produced
    buggy_sharpe = float(active_only.mean() / active_only.std() * np.sqrt(252))
    # Correct computation: pnl.mean() / pnl.std() * sqrt(252) over all days
    correct_sharpe = float(full.mean() / full.std() * np.sqrt(252))

    # 1. compute_stats returns the correct (all-days) Sharpe
    assert s["sharpe"] == pytest.approx(correct_sharpe, rel=1e-6)

    # 2. The correct Sharpe is substantially lower than the buggy active-only Sharpe
    assert s["sharpe"] < buggy_sharpe

    # 3. The ratio should match sqrt(n_active / n_total) within 5pp —
    #    this is the exact inflation formula for the flat-day distortion.
    expected_ratio = np.sqrt(n_active / n_total)
    actual_ratio = s["sharpe"] / buggy_sharpe
    assert actual_ratio == pytest.approx(expected_ratio, abs=0.05)


# ─── print_stats_table integration test ───────────────────────────────────────

@_requires_data
def test_print_stats_table_runs(capsys: pytest.CaptureFixture[str]) -> None:
    """print_stats_table runs without error and prints expected header."""
    print_stats_table()
    captured = capsys.readouterr()
    assert "Factor-neutral RV strategy" in captured.out
    assert "Sharpe" in captured.out


# ─── plot_tearsheet / save_tearsheet integration tests ────────────────────────

@_requires_data
def test_plot_tearsheet_returns_figure() -> None:
    """plot_tearsheet returns a matplotlib Figure without raising."""
    import matplotlib.figure
    fig = plot_tearsheet()
    assert isinstance(fig, matplotlib.figure.Figure)
    matplotlib.pyplot.close(fig)


@_requires_data
def test_save_tearsheet_writes_file(tmp_path: pytest.TempPathFactory) -> None:
    """save_tearsheet saves a PNG and returns the path."""
    out = tmp_path / "test_tearsheet.png"
    result = save_tearsheet(path=out)
    assert result == out
    assert out.exists()
    assert out.stat().st_size > 10_000  # non-trivial PNG
