import json
import os
from typing import Dict, List, Tuple

import ccxt
import numpy as np
import pandas as pd

from config.settings import Settings
from core_strategy.algo import WhiteboxAlgo
from core_strategy.factor_fusion import enrich_factor_mining_features


class LiveTrader:
    def __init__(self, dry_run: bool | None = None):
        self.dry_run = Settings.LIVE_DRY_RUN if dry_run is None else dry_run
        self.strategy = WhiteboxAlgo()
        self.state_file = Settings.LIVE_STATE_FILE

        options = {
            "defaultType": "future",
            "timeout": 30000,
            "enableRateLimit": True,
        }
        if Settings.USE_PROXY:
            options["proxies"] = {"http": Settings.PROXY_URL, "https": Settings.PROXY_URL}

        self.exchange = ccxt.binance(options)
        self.exchange.options["fetchCurrencies"] = False
        if Settings.API_KEY:
            self.exchange.apiKey = Settings.API_KEY
            self.exchange.secret = Settings.SECRET_KEY

        self.exchange.load_markets()

        self.symbols = self._prepare_symbols(Settings.LIVE_SYMBOLS)
        if "BTCUSDT" not in self.symbols:
            self.symbols.insert(0, "BTCUSDT")

        os.makedirs(os.path.dirname(self.state_file), exist_ok=True)

    @staticmethod
    def _prepare_symbols(raw_symbols: List[str]) -> List[str]:
        cleaned = []
        for sym in raw_symbols:
            if not sym:
                continue
            s = str(sym).strip().upper()
            if "/" in s:
                base = s.split("/")[0]
                s = f"{base}USDT"
            if not s.endswith("USDT"):
                continue
            if s not in cleaned:
                cleaned.append(s)
        return cleaned

    @staticmethod
    def _to_ccxt_symbol(symbol: str) -> str:
        base = symbol.replace("USDT", "")
        return f"{base}/USDT:USDT"

    @staticmethod
    def _from_ccxt_symbol(ccxt_symbol: str) -> str:
        if "/" not in ccxt_symbol:
            return ccxt_symbol.replace("/", "").replace(":", "").upper()
        base = ccxt_symbol.split("/")[0]
        return f"{base.upper()}USDT"

    def _load_state(self) -> Dict:
        default_state = {
            "current_state": 0,
            "entry_price": None,
            "highest_price": None,
            "lowest_price": None,
            "last_timestamp": None,
        }
        if not os.path.exists(self.state_file):
            return default_state
        try:
            with open(self.state_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            default_state.update(data)
            return default_state
        except Exception:
            return default_state

    def _save_state(self, state: Dict) -> None:
        with open(self.state_file, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)

    @staticmethod
    def _calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
        close = df["close"]
        df["ma"] = close.rolling(window=Settings.MA_WINDOW).mean()
        df["short_ma"] = close.rolling(window=Settings.SHORT_FAST_MA_WINDOW).mean()
        std = close.rolling(window=Settings.MA_WINDOW).std()
        df["upper_band"] = df["ma"] + (std * Settings.BB_STD)

        cum_max = close.cummax()
        df["drawdown"] = (close - cum_max) / cum_max

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
        df["diffusion_index"] = ((sig_rsi + sig_price + sig_cross) / 3) * 100

        roc = close.diff(Settings.Y_ROC_PERIOD) / close.shift(Settings.Y_ROC_PERIOD)
        roc_mean = roc.rolling(Settings.Y_ZSCORE_WIN).mean()
        roc_std = roc.rolling(Settings.Y_ZSCORE_WIN).std()
        z_score = (roc - roc_mean) / roc_std
        df["y_index"] = z_score.cumsum().fillna(0)

        ret = close.pct_change()
        vol = ret.rolling(window=180).std() * np.sqrt(365 * 6)
        df["volatility"] = vol
        return df

    def _fetch_symbol_bars(self, symbol: str, limit: int) -> pd.DataFrame:
        ccxt_symbol = self._to_ccxt_symbol(symbol)
        ohlcv = self.exchange.fetch_ohlcv(ccxt_symbol, timeframe=Settings.TIMEFRAME, limit=limit)
        if not ohlcv:
            return pd.DataFrame()

        df = pd.DataFrame(
            ohlcv,
            columns=["timestamp", "open", "high", "low", "close", "volume"],
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df["symbol"] = symbol
        return df

    def _build_market_snapshot(
        self,
    ) -> Tuple[pd.Timestamp, pd.DataFrame, pd.DataFrame, pd.Series, Dict[str, float]]:
        frames = []
        for symbol in self.symbols:
            try:
                bars = self._fetch_symbol_bars(symbol, Settings.LIVE_LOOKBACK_BARS)
                if not bars.empty:
                    frames.append(bars)
            except Exception as exc:
                print(f"⚠️ 跳过 {symbol}: {exc}")

        if not frames:
            raise RuntimeError("没有获取到可用K线数据，无法执行策略。")

        merged = pd.concat(frames, ignore_index=True)
        merged = merged.sort_values(["timestamp", "symbol"])

        panel = merged.set_index(["timestamp", "symbol"]).sort_index()
        panel = enrich_factor_mining_features(
            panel,
            signal_smooth_span=Settings.FACTOR_SIGNAL_SMOOTH_SPAN,
        )
        panel["momentum"] = panel.groupby(level="symbol")["close"].pct_change(periods=Settings.MA_WINDOW)

        if "BTCUSDT" not in panel.index.get_level_values("symbol"):
            raise RuntimeError("缺少 BTCUSDT 数据，策略无法计算。")

        btc_df = panel.xs("BTCUSDT", level="symbol").copy()
        btc_df = self._calculate_indicators(btc_df)
        btc_metrics = btc_df[
            ["ma", "short_ma", "upper_band", "drawdown", "diffusion_index", "y_index", "close", "volatility"]
        ]

        current_ts = btc_metrics.index.max()
        if pd.isna(current_ts):
            raise RuntimeError("BTC 指标数据为空，无法生成信号。")

        slice_df = panel.loc[current_ts]
        if isinstance(slice_df, pd.Series):
            slice_df = slice_df.to_frame().T

        if "BTCUSDT" not in slice_df.index:
            raise RuntimeError(f"{current_ts} 时刻缺少 BTCUSDT 快照，无法交易。")

        btc_history_subset = btc_metrics.loc[:current_ts].tail(10)
        current_btc_row = btc_metrics.loc[current_ts]
        price_map = slice_df["close"].astype(float).to_dict()
        return current_ts, slice_df, btc_history_subset, current_btc_row, price_map

    def _fetch_equity(self) -> float:
        if not Settings.API_KEY:
            if self.dry_run:
                return float(Settings.INITIAL_CAPITAL)
            raise RuntimeError("未配置 Binance API Key，无法实盘下单。")

        try:
            balance = self.exchange.fetch_balance(params={"type": "future"})
            usdt = balance.get("USDT", {})
            total = usdt.get("total")
            if total is None:
                total_map = balance.get("total", {})
                total = total_map.get("USDT")
            if total is None:
                free = usdt.get("free", 0) or 0
                used = usdt.get("used", 0) or 0
                total = free + used
            total = float(total or 0)
            if total <= 0:
                raise RuntimeError("账户权益为0，停止执行。")
            return total
        except Exception as exc:
            if self.dry_run:
                print(f"⚠️ 获取账户权益失败，回退为初始资金: {exc}")
                return float(Settings.INITIAL_CAPITAL)
            raise

    def _fetch_positions(self, prices: Dict[str, float]) -> Dict[str, Dict[str, float]]:
        tracked = set(self.symbols)
        positions = {}
        try:
            raw_positions = self.exchange.fetch_positions()
        except Exception as exc:
            print(f"⚠️ 获取持仓失败，按空仓处理: {exc}")
            return positions

        for pos in raw_positions:
            internal_symbol = self._from_ccxt_symbol(str(pos.get("symbol", "")))
            if internal_symbol not in tracked:
                continue

            info = pos.get("info", {}) or {}
            qty = 0.0

            try:
                position_amt = info.get("positionAmt")
                if position_amt is not None:
                    qty = float(position_amt)
                else:
                    contracts = float(pos.get("contracts") or 0.0)
                    side = str(pos.get("side", "")).lower()
                    if side == "long":
                        qty = contracts
                    elif side == "short":
                        qty = -contracts
            except Exception:
                qty = 0.0

            if abs(qty) < 1e-12:
                continue

            mark_price = float(
                pos.get("markPrice")
                or pos.get("entryPrice")
                or prices.get(internal_symbol, 0.0)
                or 0.0
            )
            if mark_price <= 0:
                continue

            positions[internal_symbol] = {
                "qty": qty,
                "notional": qty * mark_price,
            }

        return positions

    @staticmethod
    def _infer_state_from_positions(positions: Dict[str, Dict[str, float]]) -> int:
        btc_notional = positions.get("BTCUSDT", {}).get("notional", 0.0)
        if abs(btc_notional) < Settings.LIVE_MIN_ORDER_USDT:
            return 0
        return 1 if btc_notional > 0 else -1

    def _apply_execution_risk(self, target_weights: Dict[str, float], next_state: int) -> Tuple[Dict[str, float], int]:
        clean_weights = {}
        for sym, w in target_weights.items():
            if sym not in self.symbols:
                continue
            w = float(w)
            if not Settings.LIVE_ALLOW_SHORT and w < 0:
                w = 0.0
            if abs(w) > 1e-8:
                clean_weights[sym] = w

        if not Settings.LIVE_ALLOW_SHORT and next_state == -1:
            next_state = 0
            clean_weights = {}

        gross = sum(abs(v) for v in clean_weights.values())
        max_gross = max(0.0, float(Settings.LIVE_MAX_GROSS_EXPOSURE))
        if gross > max_gross > 0:
            scale = max_gross / gross
            clean_weights = {k: v * scale for k, v in clean_weights.items()}

        return clean_weights, next_state

    def _build_orders(
        self,
        target_weights: Dict[str, float],
        current_positions: Dict[str, Dict[str, float]],
        equity: float,
        prices: Dict[str, float],
    ) -> List[Dict]:
        deploy_ratio = max(0.0, min(1.0, float(Settings.LIVE_CAPITAL_UTILIZATION)))
        deployable_equity = equity * deploy_ratio

        current_notional = {k: v["notional"] for k, v in current_positions.items()}
        symbols = sorted(set(self.symbols) | set(current_notional.keys()) | set(target_weights.keys()))

        orders = []
        for sym in symbols:
            if sym not in prices:
                continue
            px = float(prices[sym] or 0.0)
            if px <= 0:
                continue

            target_notional = float(target_weights.get(sym, 0.0)) * deployable_equity
            now_notional = float(current_notional.get(sym, 0.0))
            delta = target_notional - now_notional

            if abs(delta) < float(Settings.LIVE_MIN_ORDER_USDT):
                continue

            raw_amount = abs(delta) / px
            ccxt_symbol = self._to_ccxt_symbol(sym)

            try:
                amount = float(self.exchange.amount_to_precision(ccxt_symbol, raw_amount))
            except Exception:
                amount = raw_amount

            if amount <= 0:
                continue

            side = "buy" if delta > 0 else "sell"
            orders.append(
                {
                    "symbol": sym,
                    "ccxt_symbol": ccxt_symbol,
                    "side": side,
                    "amount": amount,
                    "delta_notional": delta,
                    "target_notional": target_notional,
                    "current_notional": now_notional,
                }
            )

        return orders

    def _execute_orders(self, orders: List[Dict]) -> List[Dict]:
        if not orders:
            print("ℹ️ 无需调仓。")
            return []

        results = []
        for od in orders:
            print(
                f"🧾 ORDER {od['symbol']} {od['side'].upper()} amt={od['amount']:.8f} "
                f"delta=${od['delta_notional']:.2f}"
            )

            if self.dry_run:
                results.append({"symbol": od["symbol"], "status": "simulated"})
                continue

            params = {}
            if Settings.LIVE_POSITION_SIDE:
                params["positionSide"] = Settings.LIVE_POSITION_SIDE

            try:
                resp = self.exchange.create_order(
                    symbol=od["ccxt_symbol"],
                    type="market",
                    side=od["side"],
                    amount=od["amount"],
                    params=params,
                )
                results.append({"symbol": od["symbol"], "status": "submitted", "id": resp.get("id")})
            except Exception as exc:
                print(f"❌ 下单失败 {od['symbol']}: {exc}")
                results.append({"symbol": od["symbol"], "status": "failed", "error": str(exc)})

        return results

    def run_once(self) -> Dict:
        ts, slice_df, btc_history, btc_row, prices = self._build_market_snapshot()
        equity = self._fetch_equity()
        positions = self._fetch_positions(prices)
        state = self._load_state()

        inferred_state = self._infer_state_from_positions(positions)
        current_state = int(state.get("current_state", inferred_state))
        if current_state != inferred_state:
            current_state = inferred_state

        entry_price = state.get("entry_price")
        highest_price = state.get("highest_price")
        lowest_price = state.get("lowest_price")

        btc_close = float(btc_row["close"])
        if current_state == 0:
            entry_price, highest_price, lowest_price = None, None, None
        elif current_state == 1:
            if not entry_price:
                entry_price = btc_close
            if highest_price is None:
                highest_price = btc_close
        elif current_state == -1:
            if not entry_price:
                entry_price = btc_close
            if lowest_price is None:
                lowest_price = btc_close

        target_weights, next_state = self.strategy.compute_signal(
            df_slice=slice_df,
            btc_history=btc_history,
            current_state=current_state,
            entry_price=entry_price,
            highest_price=highest_price,
            lowest_price=lowest_price,
        )
        target_weights, next_state = self._apply_execution_risk(target_weights, next_state)

        orders = self._build_orders(
            target_weights=target_weights,
            current_positions=positions,
            equity=equity,
            prices=prices,
        )
        order_results = self._execute_orders(orders)

        if next_state != current_state:
            if next_state == 0:
                entry_price, highest_price, lowest_price = None, None, None
            else:
                entry_price = btc_close
                highest_price = btc_close
                lowest_price = btc_close
        else:
            if next_state == 1:
                if highest_price is None or btc_close > float(highest_price):
                    highest_price = btc_close
            elif next_state == -1:
                if lowest_price is None or btc_close < float(lowest_price):
                    lowest_price = btc_close

        new_state = {
            "current_state": int(next_state),
            "entry_price": float(entry_price) if entry_price is not None else None,
            "highest_price": float(highest_price) if highest_price is not None else None,
            "lowest_price": float(lowest_price) if lowest_price is not None else None,
            "last_timestamp": str(ts),
            "last_equity": float(equity),
            "dry_run": bool(self.dry_run),
        }
        self._save_state(new_state)

        summary = {
            "timestamp": str(ts),
            "equity": float(equity),
            "current_state": int(current_state),
            "next_state": int(next_state),
            "target_weights": target_weights,
            "orders": order_results,
            "dry_run": bool(self.dry_run),
        }

        print(
            f"✅ 执行完成 | ts={summary['timestamp']} | state {current_state}->{next_state} | "
            f"orders={len(order_results)} | dry_run={self.dry_run}"
        )
        return summary
