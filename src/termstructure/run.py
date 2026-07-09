"""CLI entry point for the TermStructure pipeline.

Usage:
    python -m termstructure.run configs/base.yaml [stage]

Stages (run in order):
    data        — fetch raw FRED yields and Fed bond curves
    fit         — fit Svensson curve daily; build zero panel; run PCA
    signal      — compute residuals, z-score signal, signal panel
    portfolio   — construct factor-neutral positions across history
    backtest    — compute leg P&L and net P&L after costs
    report      — generate and save tearsheet PNG

Omit [stage] to run all stages in sequence.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml


def _load_config(path: str) -> dict[str, Any]:
    with open(path) as f:
        return dict(yaml.safe_load(f))


# ─── Stage runners ────────────────────────────────────────────────────────────

def run_data(cfg: dict[str, Any]) -> None:
    from termstructure.ingest.fred import fetch_fred_yields
    from termstructure.ingest.treasury_bonds import load_fed_curves

    start = cfg["data"]["start_date"]
    end   = cfg["data"]["end_date"]

    print(f"\n[data] Fetching FRED CMT yields {start} → {end}")
    fetch_fred_yields(start, end)

    print("\n[data] Downloading Fed GSW bond curves")
    load_fed_curves()


def run_fit(cfg: dict[str, Any]) -> None:
    from termstructure.curves.svensson import fit_svensson_daily
    from termstructure.pca.decomposition import fit_pca, save_pca_outputs
    from termstructure.pca.panel import build_zero_panel

    start = cfg["data"]["start_date"]
    end   = cfg["data"]["end_date"]

    print(f"\n[fit] Fitting Svensson curves {start} → {end}")
    fit_svensson_daily(start, end)

    print("\n[fit] Building zero-coupon yield panel")
    build_zero_panel()

    print("\n[fit] Running PCA decomposition")
    fit_pca()
    save_pca_outputs()


def run_signal(cfg: dict[str, Any]) -> None:
    from termstructure.signals.richness import (
        build_signal_panel,
        compute_residuals,
        compute_zscore_signal,
    )

    sc = cfg["signal"]

    print("\n[signal] Computing Svensson residuals")
    compute_residuals()

    print(f"\n[signal] Computing rolling z-score (window={sc['window']}, "
          f"min_periods={sc['min_periods']})")
    compute_zscore_signal(window=sc["window"], min_periods=sc["min_periods"])

    print(f"\n[signal] Building signal panel (entry_z={sc['entry_z']})")
    build_signal_panel(entry_z=sc["entry_z"])


def run_portfolio(cfg: dict[str, Any]) -> None:
    from termstructure.backtest.portfolio import build_portfolio_history

    pc = cfg["portfolio"]

    print(f"\n[portfolio] Constructing positions "
          f"(n_per_side={pc['n_per_side']}, target_dv01={pc['target_dv01']:,.0f})")
    build_portfolio_history(
        target_dv01=float(pc["target_dv01"]),
        n_per_side=int(pc["n_per_side"]),
    )


def run_backtest(cfg: dict[str, Any]) -> None:
    from termstructure.backtest.pnl import compute_net_pnl

    cost_bps       = float(cfg["cost_bps"])
    position_scale = float(cfg["position_scale"])

    print(f"\n[backtest] Computing net P&L "
          f"(cost_bps={cost_bps}, position_scale={position_scale})")
    df = compute_net_pnl(cost_bps=cost_bps, position_scale=position_scale)

    out = Path("data/processed/net_pnl.parquet")
    df.to_parquet(out, index=False)
    print(f"[backtest] Saved {len(df):,} rows → {out}")


def run_report(cfg: dict[str, Any]) -> None:
    from termstructure.report.tearsheet import save_tearsheet

    cost_bps       = float(cfg["cost_bps"])
    position_scale = float(cfg["position_scale"])
    out = Path(cfg["report"]["output_path"])
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n[report] Saving tearsheet → {out}")
    save_tearsheet(path=out, cost_bps=cost_bps, position_scale=position_scale)
    print(f"[report] Done: {out}")


# ─── Stage registry ───────────────────────────────────────────────────────────

_STAGES: dict[str, Any] = {
    "data":      run_data,
    "fit":       run_fit,
    "signal":    run_signal,
    "portfolio": run_portfolio,
    "backtest":  run_backtest,
    "report":    run_report,
}

_STAGE_ORDER = ["data", "fit", "signal", "portfolio", "backtest", "report"]


# ─── Entry point ──────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m termstructure.run",
        description="Run TermStructure pipeline stages",
    )
    parser.add_argument("config", help="Path to YAML config file (e.g. configs/base.yaml)")
    parser.add_argument(
        "stage",
        nargs="?",
        choices=list(_STAGES),
        help="Pipeline stage to run (default: all stages in order)",
    )
    args = parser.parse_args()

    cfg = _load_config(args.config)

    stages = [args.stage] if args.stage else _STAGE_ORDER

    for stage in stages:
        print(f"\n{'='*60}")
        print(f"  Stage: {stage}")
        print(f"{'='*60}")
        _STAGES[stage](cfg)

    print("\nDone.")


if __name__ == "__main__":
    main()
