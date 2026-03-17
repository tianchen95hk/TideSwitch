# core_backtest/engine.py
import pandas as pd
import numpy as np
from config.settings import Settings
from core_strategy.factor_fusion import enrich_factor_mining_features
from core_strategy.regime_signaler import RegimeSignaler
import os

class BacktestEngine:
    def __init__(self, df, strategy):
        self.raw_df = df
        self.strategy = strategy
        self.equity = Settings.INITIAL_CAPITAL
        self.history = []
        self.positions = {}
        self.last_prices = {}
        self.trade_signals = []  # 记录买卖信号
        self.last_state = 0  # 记录上一个状态，用于检测状态变化
        self.last_signal_time = None  # 记录最后一个信号时间，防止重复
        self.stoploss_cooldown_until = None  # 硬止损后的冷却截止时间（含）
        
    def calculate_indicators(self, df):
        close = df['close']
        ma_min_periods = max(180, int(Settings.MA_WINDOW // 4))
        # 1. 基础指标
        df['ma'] = close.rolling(window=Settings.MA_WINDOW, min_periods=ma_min_periods).mean()
        df['short_ma'] = close.rolling(window=Settings.SHORT_FAST_MA_WINDOW).mean()
        std = close.rolling(window=Settings.MA_WINDOW, min_periods=ma_min_periods).std()
        df['upper_band'] = df['ma'] + (std * Settings.BB_STD)
        cum_max = close.cummax()
        df['drawdown'] = (close - cum_max) / cum_max
        
        # 2. Diffusion Index
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=Settings.DIFF_RSI_WIN).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=Settings.DIFF_RSI_WIN).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        sig_rsi = (rsi > 50).astype(int) * 2 - 1
        ma_long = close.rolling(Settings.DIFF_MA_LONG).mean()
        ma_short = close.rolling(Settings.DIFF_MA_SHORT).mean()
        sig_price = (close > ma_long).astype(int) * 2 - 1
        sig_cross = (ma_short > ma_long).astype(int) * 2 - 1
        df['diffusion_index'] = ((sig_rsi + sig_price + sig_cross) / 3) * 100
        
        # 3. Y-Index
        roc = close.diff(Settings.Y_ROC_PERIOD) / close.shift(Settings.Y_ROC_PERIOD)
        roc_mean = roc.rolling(Settings.Y_ZSCORE_WIN).mean()
        roc_std = roc.rolling(Settings.Y_ZSCORE_WIN).std()
        z_score = (roc - roc_mean) / roc_std
        df['y_index'] = z_score.cumsum().fillna(0)
        
        # 4. [新增] 波动率 (Volatility)
        # 计算 30日(180周期) 滚动波动率，并年化
        # 4H线年化系数 sqrt(365 * 6) ≈ 46.8
        ret = close.pct_change()
        vol = ret.rolling(window=180).std() * np.sqrt(365 * 6)
        df['volatility'] = vol
        
        return df

    def preprocess_factors(self):
        print("⚡️ 正在计算高级指标 (融合 FactorMining 多因子)...")
        enhanced_df = enrich_factor_mining_features(
            self.raw_df,
            signal_smooth_span=Settings.FACTOR_SIGNAL_SMOOTH_SPAN,
        )
        enhanced_df['momentum'] = enhanced_df.groupby('symbol')['close'].pct_change(periods=Settings.MA_WINDOW)

        btc_df = enhanced_df.xs('BTCUSDT', level='symbol').copy()
        btc_df = self.calculate_indicators(btc_df)

        # 叠加四状态 Regime 信号：用于 beta/alpha 双引擎动态配比
        signaler = RegimeSignaler()
        ts_all = enhanced_df.index.get_level_values('timestamp')
        s_start = pd.Timestamp(ts_all.min()).strftime('%Y-%m-%d')
        s_end = pd.Timestamp(ts_all.max()).strftime('%Y-%m-%d')
        regime_df = signaler.build(enhanced_df, start_date=s_start, end_date=s_end)
        attach_cols = ['state', 'mode', 'regime_score', 'confidence', 'target_beta_hint']
        btc_df = btc_df.join(regime_df[attach_cols], how='left')

        # 将 volatility 加入 btc_metrics
        self.btc_metrics = btc_df[
            [
                'ma', 'short_ma', 'upper_band', 'drawdown',
                'diffusion_index', 'y_index', 'close', 'volatility',
                'state', 'mode', 'regime_score', 'confidence', 'target_beta_hint'
            ]
        ]
        
        self.data = enhanced_df.sort_index()
        print("✅ 指标计算完成")

    def run(self):
        self.preprocess_factors()
        
        timestamps = self.data.index.get_level_values(0).unique()
        start = pd.to_datetime(Settings.BACKTEST_START_DATE)
        end = pd.to_datetime(Settings.BACKTEST_END_DATE)
        timestamps = [t for t in timestamps if start <= t <= end]
        history_lookback = max(10, Settings.BUY_Y_RISE_DAYS + 2, Settings.SELL_Y_FALL_DAYS + 2)
        btc_metrics_window = self.btc_metrics.reindex(timestamps)
        
        print(f"🚀 开始回测 {len(timestamps)} 个周期 (动态平滑版)...")
        
        current_state = 0 
        entry_price = None 
        highest_price = None 
        lowest_price = None
        cooldown_until = None
        positions_timeline = []
        prev_weights_snapshot = {}
        state_holding_bars = 0
        total_turnover = 0.0
        rebalance_trade_count = 0

        def _weights_from_positions(positions, equity):
            if not equity:
                return {}
            return {sym: (val / equity) for sym, val in positions.items() if val is not None and abs(val) > 0}

        def _format_weight_changes(from_w, to_w, min_abs_change=0.02, max_items=6):
            keys = set(from_w.keys()) | set(to_w.keys())
            changes = []
            for k in keys:
                f = float(from_w.get(k, 0.0) or 0.0)
                t = float(to_w.get(k, 0.0) or 0.0)
                d = t - f
                if abs(d) >= min_abs_change:
                    changes.append((k, f, t, d))
            changes.sort(key=lambda x: abs(x[3]), reverse=True)
            changes = changes[:max_items]
            if not changes:
                return ""
            parts = []
            for sym, f, t, _d in changes:
                parts.append(f"{sym}:{f:.0%}→{t:.0%}")
            return " | ".join(parts)

        def _format_top_weights(weights, top_n=6):
            if not weights:
                return ""
            items = sorted(weights.items(), key=lambda kv: abs(kv[1]), reverse=True)[:top_n]
            return " | ".join([f"{sym}:{w:.0%}" for sym, w in items])

        def _calculate_rebalance_fee(current_positions, target_weights, equity, tradable_symbols):
            """
            按调仓换手额收取 taker fee：
            fee = sum(|target_notional - current_notional|) * TAKER_FEE
            """
            if equity <= 0 or Settings.TAKER_FEE <= 0:
                return 0.0

            symbols = (set(current_positions.keys()) | set(target_weights.keys())) & set(tradable_symbols)
            if not symbols:
                return 0.0

            turnover = 0.0
            for sym in symbols:
                now_val = float(current_positions.get(sym, 0.0) or 0.0)
                tgt_val = float(target_weights.get(sym, 0.0) or 0.0) * equity
                turnover += abs(tgt_val - now_val)
            return turnover * Settings.TAKER_FEE

        def _clean_weights(weights, tradable_symbols):
            clean = {}
            for sym, w in (weights or {}).items():
                if sym not in tradable_symbols:
                    continue
                w = float(w)
                if not np.isfinite(w) or abs(w) < 1e-8:
                    continue
                clean[sym] = w
            return clean

        def _cap_gross_exposure(weights):
            clean = dict(weights) if weights else {}
            if not clean:
                return {}
            gross_cap = max(float(Settings.TARGET_GROSS_EXPOSURE), 0.0)
            if gross_cap <= 0:
                return {}
            gross = float(sum(abs(v) for v in clean.values()))
            if gross <= gross_cap or gross <= 1e-12:
                return clean
            scale = gross_cap / gross
            return {k: v * scale for k, v in clean.items()}

        def _blend_weights(prev_weights, target_weights, alpha):
            symbols = set(prev_weights.keys()) | set(target_weights.keys())
            blended = {}
            for sym in symbols:
                p = float(prev_weights.get(sym, 0.0) or 0.0)
                t = float(target_weights.get(sym, 0.0) or 0.0)
                v = p + alpha * (t - p)
                if abs(v) >= 1e-8:
                    blended[sym] = v
            return blended

        def _weights_turnover(from_w, to_w):
            keys = set(from_w.keys()) | set(to_w.keys())
            return float(sum(abs(float(to_w.get(k, 0.0) or 0.0) - float(from_w.get(k, 0.0) or 0.0)) for k in keys))

        def _estimate_signal_edge(weights, market_slice):
            if not weights:
                return 0.0
            edge = 0.0
            for sym, w in weights.items():
                if sym not in market_slice.index:
                    continue
                row = market_slice.loc[sym]
                fm = float(row.get('fm_composite', 0.0) or 0.0)
                funding = float(row.get('fundingRate', row.get('funding_rate', 0.0)) or 0.0)
                carry = -funding
                signal = 0.6 * fm + 0.4 * carry
                if w >= 0:
                    edge += float(w) * signal
                else:
                    edge += abs(float(w)) * (-signal)
            return float(edge)

        def _bar_regime(btc_row):
            close = float(btc_row.get('close', np.nan))
            ma = float(btc_row.get('ma', np.nan))
            if (not np.isfinite(close)) or (not np.isfinite(ma)) or ma <= 0:
                return 'neutral'
            band = float(Settings.REGIME_HYSTERESIS_PCT)
            bull_th = ma * (1.0 + band)
            bear_th = ma * (1.0 - band)
            is_bull = close > bull_th
            is_bear = close < bear_th
            if is_bull:
                trend = close / ma - 1.0
                breadth = float(np.clip(float(btc_row.get('diffusion_index', 0.0)) / 100.0, -1.0, 1.0))
                drawdown = float(btc_row.get('drawdown', -0.2) or -0.2)
                is_super_bull = (
                    trend >= float(Settings.SUPERBULL_TREND_THRESHOLD)
                    and breadth >= float(Settings.SUPERBULL_BREADTH_THRESHOLD)
                    and drawdown >= float(Settings.SUPERBULL_DRAWDOWN_THRESHOLD)
                )
                return 'super_bull' if is_super_bull else 'bull'
            if is_bear:
                return 'bear'
            return 'neutral'

        def _execution_profile(btc_row):
            regime = _bar_regime(btc_row)
            if regime == 'super_bull':
                return regime, float(Settings.MIN_REBALANCE_DELTA_SUPER_BULL), int(Settings.MIN_HOLD_BARS_SUPER_BULL)
            if regime == 'bull':
                return regime, float(Settings.MIN_REBALANCE_DELTA_BULL), int(Settings.MIN_HOLD_BARS_BULL)
            if regime == 'bear':
                return regime, float(Settings.MIN_REBALANCE_DELTA_BEAR), int(Settings.MIN_HOLD_BARS_BEAR)
            return regime, float(Settings.MIN_REBALANCE_DELTA_NEUTRAL), int(Settings.MIN_HOLD_BARS_NEUTRAL)

        def _stoploss_cooldown_bars(btc_row):
            mode = str(btc_row.get('mode', '') or '').strip().upper()
            if mode == 'HOLD_BTC_BETA':
                return int(max(Settings.STOPLOSS_COOLDOWN_BARS_BETA, 0))
            if mode == 'ALPHA_MODE':
                return int(max(Settings.STOPLOSS_COOLDOWN_BARS_ALPHA, 0))
            return int(max(Settings.STOPLOSS_COOLDOWN_BARS, 0))

        rebalance_interval = max(int(Settings.REBALANCE_INTERVAL), 1)
        execution_alpha = float(np.clip(Settings.EXECUTION_ALPHA, 0.05, 1.0))
        
        for i, ts in enumerate(timestamps):
            try:
                slice_df = self.data.loc[ts]
            except:
                continue

            btc_row = btc_metrics_window.iloc[i]
            if btc_row.isna().all():
                continue
            bar_regime, bar_min_delta, bar_min_hold = _execution_profile(btc_row)
            
            # --- 1. 结算 PnL ---
            pnl_total = 0
            is_funding_hour = (ts.hour % 8 == 0)
            marked_positions = {}
            
            for sym, pos_val in self.positions.items():
                if sym not in slice_df.index:
                    marked_positions[sym] = pos_val
                    continue
                row = slice_df.loc[sym]
                curr_price = row['close']
                last_price = self.last_prices.get(sym, row['open'])

                if (not np.isfinite(last_price)) or last_price <= 0:
                    last_price = curr_price
                if (not np.isfinite(curr_price)) or curr_price <= 0:
                    curr_price = last_price

                price_pnl = 0.0
                if pos_val > 0:
                    price_pnl = (curr_price - last_price) / last_price * pos_val
                    new_pos_val = pos_val + price_pnl
                else:
                    price_pnl = (last_price - curr_price) / last_price * abs(pos_val)
                    new_abs = max(abs(pos_val) - price_pnl, 0.0)
                    new_pos_val = -new_abs

                pnl = price_pnl
                if is_funding_hour:
                    fr = row.get('fundingRate', 0)
                    if pd.isna(fr): fr = 0
                    funding_cost = abs(pos_val) * fr
                    if pos_val > 0: pnl -= funding_cost
                    else: pnl += funding_cost 
                
                pnl_total += pnl
                marked_positions[sym] = float(new_pos_val)
                self.last_prices[sym] = curr_price
            
            self.equity += pnl_total
            self.positions = marked_positions
            
            # --- 2. 策略信号 ---
            start_i = max(0, i - history_lookback + 1)
            btc_history_subset = btc_metrics_window.iloc[start_i:i + 1]

            from_weights = _weights_from_positions(self.positions, self.equity)

            # 冷却期内强制空仓：避免硬止损后立即抄底
            if cooldown_until is not None and ts <= cooldown_until:
                target_weights, new_state = {}, 0
            else:
                target_weights, new_state = self.strategy.compute_signal(
                    slice_df, 
                    btc_history_subset, 
                    current_state,
                    entry_price,
                    highest_price,
                    lowest_price
                )

            target_weights = _clean_weights(target_weights, slice_df.index)
            target_weights = _cap_gross_exposure(target_weights)
            to_weights = dict(target_weights) if target_weights else {}

            # 最短持仓约束：噪音反转不立即切仓（硬止损除外）
            if current_state != 0 and new_state != current_state and state_holding_bars < max(int(bar_min_hold), 0):
                force_exit = False
                if entry_price and current_state == 1:
                    long_pnl = (btc_row['close'] - entry_price) / entry_price
                    force_exit = long_pnl < -Settings.STOP_LOSS_PCT
                elif entry_price and current_state == -1:
                    short_pnl = (entry_price - btc_row['close']) / entry_price
                    force_exit = short_pnl < -Settings.STOP_LOSS_PCT

                if not force_exit:
                    new_state = current_state
                    target_weights = dict(from_weights)
                    to_weights = dict(from_weights)
            
            # --- 状态维护与信号记录 ---
            signal_reason = None
            sold_this_bar = False
            
            # 防止同一时间点记录多个信号
            if self.last_signal_time == ts:
                signal_reason = None
            elif current_state == 0 and new_state != 0:
                entry_price = btc_row['close']
                highest_price = btc_row['close']
                lowest_price = btc_row['close']

                if new_state == 1:
                    is_bull = btc_row['close'] > btc_row['ma']
                    signal_reason = "牛市进场" if is_bull else "熊市抄底"
                    action = 'buy'
                    icon = '🟢'
                    verb = 'BUY '
                else:
                    signal_reason = "开空仓"
                    action = 'short'
                    icon = '🟣'
                    verb = 'SHORT'

                delta_summary = _format_weight_changes(from_weights, to_weights)

                self.trade_signals.append({
                    'timestamp': ts,
                    'action': action,
                    'state': new_state,
                    'price': btc_row['close'],
                    'reason': signal_reason,
                    'from_weights': from_weights,
                    'to_weights': to_weights,
                    'delta_summary': delta_summary,
                    'equity': self.equity
                })
                pos_txt = _format_top_weights(to_weights)
                print(
                    f"{icon} {verb} [{ts}] {signal_reason}"
                    + (f" | {delta_summary}" if delta_summary else "")
                )
                print(f"POS   [{ts}] {pos_txt if pos_txt else 'Flat'}")
                self.last_signal_time = ts
                
            elif current_state != 0 and new_state == current_state:
                if current_state == 1:
                    if btc_row['close'] > highest_price:
                        highest_price = btc_row['close']
                elif current_state == -1:
                    if lowest_price is None or btc_row['close'] < lowest_price:
                        lowest_price = btc_row['close']
                        
            elif new_state == 0:
                # 记录卖出信号
                if current_state != 0 and self.last_signal_time != ts:
                    sold_this_bar = True
                    # 卖出时目标仓位为0
                    to_weights = {}
                    # 判断卖出原因
                    if current_state == 1:
                        # 优先判断止损/止盈，避免被“跌破均线”覆盖导致冷却期失效
                        pct_change = (btc_row['close'] - entry_price) / entry_price if entry_price else 0
                        if pct_change < -Settings.STOP_LOSS_PCT:
                            signal_reason = f"硬止损({pct_change:.1%})"
                            cooldown_until = ts + pd.Timedelta(hours=4 * _stoploss_cooldown_bars(btc_row))
                        elif highest_price and highest_price > entry_price:
                            drawdown_from_peak = (btc_row['close'] - highest_price) / highest_price
                            if drawdown_from_peak < -Settings.TRAILING_STOP_PCT:
                                signal_reason = f"移动止盈({drawdown_from_peak:.1%})"
                            else:
                                signal_reason = "平多仓"
                        elif btc_row['close'] < btc_row['ma']:
                            signal_reason = "跌破均线离场"
                        else:
                            upper_band = btc_row.get('upper_band', np.nan)
                            if pd.notna(upper_band) and btc_row['close'] >= upper_band * Settings.SELL_BB_FACTOR:
                                signal_reason = "触及布林上轨止盈"
                            else:
                                signal_reason = "平多仓"
                    else:
                        # 空头平仓原因
                        pct_change = (entry_price - btc_row['close']) / entry_price if entry_price else 0
                        if pct_change < -Settings.STOP_LOSS_PCT:
                            signal_reason = f"空头止损({pct_change:.1%})"
                            cooldown_until = ts + pd.Timedelta(hours=4 * _stoploss_cooldown_bars(btc_row))
                        elif lowest_price and entry_price and lowest_price < entry_price:
                            drawup_from_low = (btc_row['close'] - lowest_price) / lowest_price
                            if drawup_from_low > Settings.TRAILING_STOP_PCT:
                                signal_reason = f"空头移动止盈({drawup_from_low:.1%})"
                            elif btc_row['close'] > btc_row['ma']:
                                signal_reason = "站上均线平空"
                            else:
                                signal_reason = "平空仓"
                        else:
                            signal_reason = "站上均线平空" if btc_row['close'] > btc_row['ma'] else "平空仓"

                    delta_summary = _format_weight_changes(from_weights, to_weights)
                    
                    self.trade_signals.append({
                        'timestamp': ts,
                        'action': 'sell' if current_state == 1 else 'cover',
                        'state': current_state,
                        'price': btc_row['close'],
                        'reason': signal_reason,
                        'from_weights': from_weights,
                        'to_weights': to_weights,
                        'delta_summary': delta_summary,
                        'equity': self.equity
                    })
                    icon = '🔴' if current_state == 1 else '🟠'
                    verb = 'SELL ' if current_state == 1 else 'COVER'
                    print(
                        f"{icon} {verb} [{ts}] {signal_reason}"
                        + (f" | {delta_summary}" if delta_summary else "")
                    )
                    print(f"POS   [{ts}] Flat")
                    self.last_signal_time = ts

                entry_price = None
                highest_price = None
                lowest_price = None

            # 同一根bar里如果发生卖出，禁止立刻反手买入（避免“止损后立刻抄底”）
            if sold_this_bar and new_state != 0:
                new_state = 0
                target_weights = {}

            state_changed = (new_state != current_state)
            current_state = new_state
            
            # --- 3. 调仓 ---
            executed_weights = dict(target_weights)
            if current_state == 0:
                executed_weights = {}
            elif (not state_changed):
                should_rebalance = (i == 0) or (i % rebalance_interval == 0)
                if should_rebalance:
                    executed_weights = _blend_weights(
                        prev_weights=from_weights,
                        target_weights=target_weights,
                        alpha=execution_alpha,
                    )
                else:
                    # 非调仓窗口持仓不动，降低噪音换手
                    executed_weights = dict(from_weights)
            executed_weights = _clean_weights(executed_weights, slice_df.index)
            executed_weights = _cap_gross_exposure(executed_weights)

            turnover_ratio = _weights_turnover(from_weights, executed_weights)
            min_delta = float(max(bar_min_delta, 0.0))
            # 调仓门槛：权重变化不足不交易
            if (not state_changed) and turnover_ratio < min_delta:
                executed_weights = dict(from_weights)
                turnover_ratio = 0.0

            # 成本缓冲：信号净收益需高于交易成本缓冲才调仓
            if turnover_ratio > 0 and (not state_changed):
                edge = _estimate_signal_edge(executed_weights, slice_df)
                expected_cost = turnover_ratio * float(Settings.TAKER_FEE) * float(max(Settings.COST_BUFFER_MULTIPLIER, 0.0))
                if edge <= expected_cost + float(Settings.SIGNAL_EDGE_BUFFER):
                    executed_weights = dict(from_weights)
                    turnover_ratio = 0.0

            rebalance_fee = _calculate_rebalance_fee(
                current_positions=self.positions,
                target_weights=executed_weights,
                equity=self.equity,
                tradable_symbols=slice_df.index,
            )
            if rebalance_fee > 0:
                self.equity -= rebalance_fee

            new_positions = {}
            for sym, w in executed_weights.items():
                if sym in slice_df.index:
                    val = self.equity * w
                    new_positions[sym] = val
                    self.last_prices[sym] = slice_df.loc[sym]['close']

            self.positions = new_positions
            total_turnover += float(turnover_ratio)
            if turnover_ratio > 1e-10:
                rebalance_trade_count += 1

            curr_weights_snapshot = _weights_from_positions(self.positions, self.equity)
            weights_delta_summary = _format_weight_changes(prev_weights_snapshot, curr_weights_snapshot)

            if weights_delta_summary:
                print(f"🔧 REBAL [{ts}] {weights_delta_summary}")

            timeline_row = {
                'timestamp': ts,
                'equity': self.equity,
                'state': current_state,
                'delta_summary': weights_delta_summary,
                'turnover': float(turnover_ratio),
            }
            for sym, w in curr_weights_snapshot.items():
                timeline_row[f"w_{sym}"] = w
            positions_timeline.append(timeline_row)
            prev_weights_snapshot = curr_weights_snapshot
            
            # [修改] 记录历史时，增加 BTC 价格作为 Benchmark
            self.history.append({
                'timestamp': ts, 
                'equity': self.equity,
                'btc_close': btc_row['close'], # 记录基准价格
                'turnover': float(turnover_ratio),
            })
            
            if i % 20 == 0:
                state_str = ["Flat", "Long", "Short"][current_state]
                top_pos = _format_top_weights(curr_weights_snapshot)
                print(f"EQUITY[{ts}] ${self.equity:,.0f} | State: {state_str}" + (f" | Pos: {top_pos}" if top_pos else ""))

            if state_changed:
                state_holding_bars = 0
            else:
                state_holding_bars += 1
        
        # 转换为 DataFrame
        res_df = pd.DataFrame(self.history).set_index('timestamp')
        self.total_turnover = float(total_turnover)
        self.rebalance_trade_count = int(rebalance_trade_count)
        
        # [新增] 运行结束后直接打印详细报告
        self.generate_report(res_df)

        if hasattr(self, 'trade_signals') and self.trade_signals:
            out_dir = Settings.OUTPUT_PATH
            os.makedirs(out_dir, exist_ok=True)
            signals_df = pd.DataFrame(self.trade_signals)
            if 'timestamp' in signals_df.columns:
                signals_df = signals_df.sort_values('timestamp')
            out_path = os.path.join(out_dir, 'trade_signals.csv')
            signals_df.to_csv(out_path, index=False)
            print(f"\n🧾 Trade signals exported: {out_path}")

        if positions_timeline:
            out_dir = Settings.OUTPUT_PATH
            os.makedirs(out_dir, exist_ok=True)
            pos_df = pd.DataFrame(positions_timeline)
            if 'timestamp' in pos_df.columns:
                pos_df = pos_df.sort_values('timestamp')
            out_path = os.path.join(out_dir, 'positions_timeseries.csv')
            pos_df.to_csv(out_path, index=False)
            print(f"🧾 Positions timeline exported: {out_path}")
        
        return res_df

    def _compute_metrics(self, returns_series, periods_per_year=365*6):
        """
        内部工具：计算量化核心指标
        """
        if len(returns_series) < 2:
            return {}
        
        # 1. 累计收益
        total_ret = (1 + returns_series).prod() - 1
        
        # 2. 年化收益 (CAGR)
        # 简单年化：mean * periods
        ann_ret = returns_series.mean() * periods_per_year
        
        # 3. 年化波动率
        ann_vol = returns_series.std() * np.sqrt(periods_per_year)
        
        # 4. 夏普比率 (Sharpe Ratio) - 假设无风险利率为0
        sharpe = ann_ret / ann_vol if ann_vol != 0 else 0
        
        # 5. 索提诺比率 (Sortino Ratio)
        downside_returns = returns_series[returns_series < 0]
        downside_std = downside_returns.std() * np.sqrt(periods_per_year)
        sortino = ann_ret / downside_std if downside_std != 0 else 0
        
        # 6. 最大回撤 (Max Drawdown)
        cum_ret_series = (1 + returns_series).cumprod()
        peak = cum_ret_series.cummax()
        drawdown = (cum_ret_series - peak) / peak
        max_dd = drawdown.min()
        
        # 7. 卡玛比率 (Calmar Ratio)
        calmar = ann_ret / abs(max_dd) if max_dd != 0 else 0
        
        return {
            'Total Return': total_ret,
            'CAGR': ann_ret,
            'Volatility': ann_vol,
            'Sharpe': sharpe,
            'Sortino': sortino,
            'Max Drawdown': max_dd,
            'Calmar': calmar
        }

    def generate_report(self, df):
        """
        生成月度对比报表
        """
        print("\n" + "="*60)
        print("📊 4Alpha Pro 深度量化分析报告")
        print("="*60)
        
        # 1. 准备数据
        # 计算 4H 收益率
        df['strat_ret'] = df['equity'].pct_change().fillna(0)
        df['btc_ret'] = df['btc_close'].pct_change().fillna(0)
        
        # 2. 全局指标对比
        strat_metrics = self._compute_metrics(df['strat_ret'])
        btc_metrics = self._compute_metrics(df['btc_ret'])
        
        print(f"\n🏆 全局表现 (Strategy vs BTC):")
        print(f"{'Metric':<20} | {'Strategy':<15} | {'BTC Benchmark':<15} | {'Diff'}")
        print("-" * 65)
        
        metrics_order = ['Total Return', 'CAGR', 'Volatility', 'Sharpe', 'Sortino', 'Max Drawdown', 'Calmar']
        for m in metrics_order:
            v1 = strat_metrics.get(m, 0)
            v2 = btc_metrics.get(m, 0)
            diff = v1 - v2
            
            # 格式化输出
            if m in ['Sharpe', 'Sortino', 'Calmar']:
                fmt = "{:.2f}"
                diff_fmt = "{:+.2f}"
            else: # 百分比
                fmt = "{:.2%}"
                diff_fmt = "{:+.2%}"
                
            print(f"{m:<20} | {fmt.format(v1):<15} | {fmt.format(v2):<15} | {diff_fmt.format(diff)}")
            
        # 3. 月度详细数据
        print(f"\n📅 月度回报热力图:")
        print("-" * 85)
        print(f"{'Month':<10} | {'Strat Ret':<12} | {'BTC Ret':<12} | {'Alpha':<10} | {'Strat Sharpe':<12} | {'Strat MaxDD'}")
        print("-" * 85)
        
        # 按月分组
        # 使用 resample('M') 或 groupby
        monthly_groups = df.groupby(pd.Grouper(freq='ME')) # pandas新版用 'ME' 代表 Month End
        
        for date, group in monthly_groups:
            if len(group) < 10: continue # 跳过数据太少的月份
            
            # 计算该月的指标
            m_strat_ret = (group['equity'].iloc[-1] / group['equity'].iloc[0]) - 1
            m_btc_ret = (group['btc_close'].iloc[-1] / group['btc_close'].iloc[0]) - 1
            alpha = m_strat_ret - m_btc_ret
            
            # 计算该月内的 Sharpe 和 DD
            # 注意：月内 Sharpe 年化系数依然用 4H 频率
            sub_metrics = self._compute_metrics(group['strat_ret'])
            
            print(f"{date.strftime('%Y-%m'):<10} | {m_strat_ret:>11.2%} | {m_btc_ret:>11.2%} | {alpha:>9.2%} | {sub_metrics['Sharpe']:>12.2f} | {sub_metrics['Max Drawdown']:>10.2%}")
            
        print("-" * 85)
