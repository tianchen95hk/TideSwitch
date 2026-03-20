# TideSwitch

一个面向加密市场的 4H 策略框架，核心目标是：

- 牛市阶段尽量抓住 BTC beta。
- 震荡/防守阶段用 alpha 引擎降低回撤并提升超额收益稳定性。

## 核心逻辑（最简版）

1. 双引擎
- Beta 引擎：主要用 BTC 仓位吃趋势。
- Alpha 引擎：在非强趋势阶段做低相关增强（横截面/资金费率/basis-term）。

2. 四状态 Regime（`core_strategy/regime_signaler.py`）
- `EXPLOSIVE_BULL`
- `TREND_BULL`
- `RANGE`
- `DEFENSIVE`

3. 二元决策
- `HOLD_BTC_BETA`：提升 BTC 暴露，优先 beta。
- `ALPHA_MODE`：降低方向暴露，优先 alpha。

4. 回测执行（`core_backtest/engine.py`）
- 4H 频率。
- 含交易费用、资金费率、调仓门槛、最短持仓、成本缓冲。
- 持仓逐 bar mark-to-market。

## 目录

- `config/settings.py`：参数与开关。
- `core_data/loader.py`：数据下载与更新（Binance Futures）。
- `core_strategy/algo.py`：主策略与仓位构建。
- `core_strategy/regime_signaler.py`：状态信号器。
- `core_backtest/engine.py`：回测引擎。
- `run_backtest.py`：一键回测入口。
- `run_regime_signal.py`：单独生成 Regime 信号与一致性报告。

## 快速开始

1. 安装依赖（示例）

```bash
python3 -m pip install pandas numpy matplotlib pyarrow ccxt python-dotenv
```

2. 配置（可选）

- 在 `.env` 中配置 `BINANCE_API_KEY`、`BINANCE_SECRET_KEY`（无也可跑回测）。
- 主要参数在 `config/settings.py`。
- 仓库只保留公开基线参数，生产/私有优化参数已脱敏，不在仓库明文保存。
- 如需本地覆盖，请在 `.env` 注入参数（例如：`CORE_BTC_WEIGHT`、`STOP_LOSS_PCT`、`TRAILING_STOP_PCT`）。

3. 运行回测

```bash
python3 run_backtest.py
```

4. 生成 Regime 信号

```bash
python3 run_regime_signal.py --start-date 2023-01-01 --end-date 2026-03-16
```

## 常见输出

- `outputs/yearly_comparison_*.csv`：年度收益与对比。
- `outputs/*with_btc_sharpe*.csv`：策略 vs BTC Sharpe 对比。
- `outputs/regime_signal_*.csv`：状态序列与统计。
- `outputs/trade_signals.csv`：交易信号明细。
