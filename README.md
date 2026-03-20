# TideSwitch

TideSwitch is a 4-hour crypto trading framework designed to:

- capture BTC beta in bull markets, and
- improve downside behavior and excess return stability via alpha overlays in range/defensive regimes.

## Strategy Overview

1. Dual-engine architecture
- Beta engine: directional exposure, primarily via BTC.
- Alpha engine: low-correlation enhancements in non-strong-trend periods (cross-sectional signals, funding carry, basis/term structure).

2. Four-state regime model (`core_strategy/regime_signaler.py`)
- `EXPLOSIVE_BULL`
- `TREND_BULL`
- `RANGE`
- `DEFENSIVE`

3. Execution modes
- `HOLD_BTC_BETA`: prioritize directional beta capture.
- `ALPHA_MODE`: reduce directional exposure and emphasize alpha.

4. Backtest engine (`core_backtest/engine.py`)
- 4H frequency
- includes trading fees, funding, rebalance threshold, minimum hold bars, and cost buffer
- mark-to-market portfolio accounting each bar

## Repository Structure

- `config/settings.py`: strategy settings and runtime switches
- `core_data/loader.py`: Binance Futures data ETL
- `core_strategy/algo.py`: main strategy logic and portfolio construction
- `core_strategy/regime_signaler.py`: regime signal generation
- `core_backtest/engine.py`: backtest engine
- `run_backtest.py`: backtest entrypoint
- `run_regime_signal.py`: standalone regime signal runner

## Quick Start

1. Install dependencies

```bash
python3 -m pip install pandas numpy matplotlib pyarrow ccxt python-dotenv
```

2. Configure environment (optional)

- Set `BINANCE_API_KEY` and `BINANCE_SECRET_KEY` in `.env` (not required for local backtests).
- Main parameters are defined in `config/settings.py`.
- The repository keeps only public baseline parameters; production/private tuned parameters are redacted.
- Override local/private values via `.env` (for example: `CORE_BTC_WEIGHT`, `STOP_LOSS_PCT`, `TRAILING_STOP_PCT`).

3. Run backtest

```bash
python3 run_backtest.py
```

4. Build regime signal report

```bash
python3 run_regime_signal.py --start-date 2023-01-01 --end-date 2026-03-16
```

## Typical Outputs

- `outputs/yearly_comparison_*.csv`: annual performance comparison
- `outputs/*with_btc_sharpe*.csv`: strategy vs BTC Sharpe comparison
- `outputs/regime_signal_*.csv`: regime time series and summaries
- `outputs/trade_signals.csv`: trade signal details
