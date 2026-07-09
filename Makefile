CONFIG ?= configs/base.yaml
PYTHON  = python

# ── Output-file groups ────────────────────────────────────────────────────────
# Each group lists every parquet written by that pipeline stage.
# Make compares timestamps across these lists to decide what to rebuild.
#
# Requires GNU Make ≥ 4.3 for &: (grouped-target) syntax.
# Check with: make --version

DATA_PARQUETS   := data/processed/treasury_bonds.parquet \
                   data/processed/cmt_yields.parquet

FIT_PARQUETS    := data/processed/svensson_params.parquet \
                   data/processed/pca_loadings.parquet \
                   data/processed/zero_panel.parquet \
                   data/processed/factor_scores.parquet

SIGNAL_PARQUETS := data/processed/bond_residuals.parquet \
                   data/processed/richness_signal.parquet \
                   data/processed/signal_panel.parquet \
                   data/processed/residual_means.parquet

PORTFOLIO_PARQUET := data/processed/portfolio_positions.parquet
BACKTEST_PARQUET  := data/processed/net_pnl.parquet
REPORT_FILE       := outputs/tearsheet.png

# ── Convenience phony shortcuts ───────────────────────────────────────────────
.PHONY: all clean data fit signal portfolio backtest report

all: $(REPORT_FILE)

data:      $(DATA_PARQUETS)
fit:       $(FIT_PARQUETS)
signal:    $(SIGNAL_PARQUETS)
portfolio: $(PORTFOLIO_PARQUET)
backtest:  $(BACKTEST_PARQUET)
report:    $(REPORT_FILE)

# ── Real file targets with dependency chain ───────────────────────────────────
# &: groups all targets on the left: one recipe run updates every file in the group.
# Make only rebuilds a group when any target is older than any prerequisite.

$(DATA_PARQUETS) &:
	$(PYTHON) -m termstructure.run $(CONFIG) data

$(FIT_PARQUETS) &: $(DATA_PARQUETS)
	$(PYTHON) -m termstructure.run $(CONFIG) fit

$(SIGNAL_PARQUETS) &: $(FIT_PARQUETS)
	$(PYTHON) -m termstructure.run $(CONFIG) signal

$(PORTFOLIO_PARQUET): $(SIGNAL_PARQUETS)
	$(PYTHON) -m termstructure.run $(CONFIG) portfolio

$(BACKTEST_PARQUET): $(PORTFOLIO_PARQUET)
	$(PYTHON) -m termstructure.run $(CONFIG) backtest

$(REPORT_FILE): $(BACKTEST_PARQUET)
	$(PYTHON) -m termstructure.run $(CONFIG) report

# ── Clean ─────────────────────────────────────────────────────────────────────
clean:
	rm -f data/processed/*.parquet outputs/*.png
