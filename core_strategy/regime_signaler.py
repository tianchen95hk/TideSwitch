import numpy as np
import pandas as pd


class RegimeSignaler:
    """
    4-state regime signaler:
    - EXPLOSIVE_BULL: 爆发行情（优先吃 BTC beta）
    - TREND_BULL: 趋势牛（保持 BTC beta）
    - RANGE: 震荡（alpha 优先）
    - DEFENSIVE: 防守（alpha/对冲优先）
    """

    EXPLOSIVE_BULL = "EXPLOSIVE_BULL"
    TREND_BULL = "TREND_BULL"
    RANGE = "RANGE"
    DEFENSIVE = "DEFENSIVE"

    MODE_BETA = "HOLD_BTC_BETA"
    MODE_ALPHA = "ALPHA_MODE"

    def __init__(self):
        # 4h bars
        self.ma_long_win = 930   # 155 天
        self.ma_mid_win = 360    # 60 天
        self.mom_30d_bars = 180
        self.mom_90d_bars = 540
        self.vol_30d_bars = 180
        self.breadth_impulse_bars = 18  # 约 3 天

        # 滞后确认（防抖）
        self.switch_confirm_bars = {
            self.EXPLOSIVE_BULL: 2,
            self.TREND_BULL: 3,
            self.RANGE: 4,
            self.DEFENSIVE: 4,
        }
        self.min_hold_bars = {
            self.EXPLOSIVE_BULL: 20,
            self.TREND_BULL: 16,
            self.RANGE: 8,
            self.DEFENSIVE: 10,
        }

    @staticmethod
    def _clip_tanh(x, scale):
        if scale <= 0:
            return np.zeros_like(x, dtype=float)
        return np.tanh(np.asarray(x, dtype=float) / float(scale))

    def _compute_feature_frame(self, df):
        close_mat = df["close"].unstack("symbol").sort_index()
        btc = close_mat["BTCUSDT"].to_frame("close").copy()

        ma_long_all = close_mat.rolling(self.ma_long_win, min_periods=max(180, self.ma_long_win // 4)).mean()
        ma_mid_all = close_mat.rolling(self.ma_mid_win, min_periods=max(90, self.ma_mid_win // 3)).mean()

        breadth_long = (close_mat > ma_long_all).astype(float).mean(axis=1) * 2.0 - 1.0
        breadth_mid = (close_mat > ma_mid_all).astype(float).mean(axis=1) * 2.0 - 1.0

        btc["ma_long"] = btc["close"].rolling(
            self.ma_long_win, min_periods=max(180, self.ma_long_win // 4)
        ).mean()
        btc["ma_mid"] = btc["close"].rolling(
            self.ma_mid_win, min_periods=max(90, self.ma_mid_win // 3)
        ).mean()
        btc["ret_4h"] = btc["close"].pct_change()
        btc["mom_30d"] = btc["close"].pct_change(self.mom_30d_bars)
        btc["mom_90d"] = btc["close"].pct_change(self.mom_90d_bars)
        btc["vol_30d_ann"] = (
            btc["ret_4h"].rolling(self.vol_30d_bars, min_periods=120).std() * np.sqrt(365 * 6)
        )
        btc["vol_ema"] = btc["vol_30d_ann"].ewm(span=180, adjust=False, min_periods=60).mean()
        btc["vol_stress"] = (btc["vol_30d_ann"] / btc["vol_ema"]) - 1.0
        btc["cummax"] = btc["close"].cummax()
        btc["drawdown"] = btc["close"] / btc["cummax"] - 1.0
        trend_ma = btc["ma_long"].where(btc["ma_long"].notna(), btc["ma_mid"])
        btc["trend"] = btc["close"] / trend_ma - 1.0
        btc["ma_long_slope_5d"] = btc["ma_long"].pct_change(30)
        btc["breadth_long"] = breadth_long.reindex(btc.index)
        btc["breadth_mid"] = breadth_mid.reindex(btc.index)
        btc["breadth_impulse"] = btc["breadth_mid"] - btc["breadth_mid"].shift(self.breadth_impulse_bars)

        # 市场离散度：越高越偏防守（常见于轮动混乱或风险偏好切换）
        ret_1d = close_mat.pct_change(6)
        btc["dispersion_1d"] = ret_1d.std(axis=1).reindex(btc.index)

        beta_score = (
            0.38 * self._clip_tanh(btc["trend"].fillna(0.0), 0.10)
            + 0.23 * self._clip_tanh(btc["mom_90d"].fillna(0.0), 0.30)
            + 0.20 * btc["breadth_long"].fillna(0.0).clip(-1.0, 1.0)
            + 0.12 * self._clip_tanh(btc["breadth_impulse"].fillna(0.0), 0.20)
            + 0.07 * self._clip_tanh((0.65 - btc["vol_30d_ann"].fillna(0.65)), 0.35)
        )
        risk_score = (
            0.34 * self._clip_tanh((-btc["drawdown"].fillna(0.0) - 0.10), 0.14)
            + 0.26 * self._clip_tanh((btc["vol_30d_ann"].fillna(0.0) - 0.75), 0.30)
            + 0.18 * self._clip_tanh((-btc["trend"].fillna(0.0)), 0.08)
            + 0.12 * self._clip_tanh((-btc["breadth_long"].fillna(0.0)), 0.25)
            + 0.10 * self._clip_tanh((btc["dispersion_1d"].fillna(0.03) - 0.035), 0.025)
        )
        btc["beta_score"] = np.clip(beta_score, -1.0, 1.0)
        btc["risk_score"] = np.clip(risk_score, -1.0, 1.0)
        btc["regime_score"] = np.clip(0.62 * btc["beta_score"] - 0.38 * btc["risk_score"], -1.0, 1.0)

        explosive_cond = (
            (btc["trend"] > 0.12)
            & (btc["mom_90d"] > 0.18)
            & (btc["breadth_long"] > 0.30)
            & (btc["drawdown"] > -0.12)
            & (btc["beta_score"] > 0.45)
        )
        trend_bull_cond = (
            (btc["trend"] > 0.025)
            & (btc["mom_30d"] > 0.00)
            & (btc["breadth_long"] > -0.02)
            & (btc["ma_long_slope_5d"] > 0.00)
            & (btc["regime_score"] > 0.10)
            & (btc["risk_score"] < 0.55)
        )
        defensive_cond = (
            ((btc["drawdown"] < -0.28) & (btc["mom_90d"] < -0.08))
            | (
                (btc["trend"] < -0.08)
                & (btc["breadth_long"] < -0.10)
                & (btc["mom_30d"] < 0.0)
            )
            | ((btc["risk_score"] > 0.72) & (btc["beta_score"] < 0.10))
        )

        btc["raw_state"] = np.select(
            [explosive_cond, defensive_cond, trend_bull_cond],
            [self.EXPLOSIVE_BULL, self.DEFENSIVE, self.TREND_BULL],
            default=self.RANGE,
        )
        return btc

    def _apply_state_hysteresis(self, feat):
        states = []
        current = self.RANGE
        current_hold = 0
        pending_state = None
        pending_count = 0

        for _, row in feat.iterrows():
            raw_state = row["raw_state"]
            trend = float(row.get("trend", 0.0) or 0.0)
            breadth = float(row.get("breadth_long", 0.0) or 0.0)
            mom90 = float(row.get("mom_90d", 0.0) or 0.0)
            drawdown = float(row.get("drawdown", 0.0) or 0.0)
            risk_score = float(row.get("risk_score", 0.0) or 0.0)

            # 极端行情立即切换，防止慢半拍
            hard_explosive = (
                trend > 0.20 and breadth > 0.45 and mom90 > 0.25 and drawdown > -0.10
            )
            hard_defensive = (drawdown < -0.35) or (risk_score > 0.85)

            if hard_explosive:
                current = self.EXPLOSIVE_BULL
                current_hold = 0
                pending_state, pending_count = None, 0
                states.append(current)
                continue
            if hard_defensive:
                current = self.DEFENSIVE
                current_hold = 0
                pending_state, pending_count = None, 0
                states.append(current)
                continue

            if raw_state == current:
                current_hold += 1
                pending_state, pending_count = None, 0
                states.append(current)
                continue

            if pending_state == raw_state:
                pending_count += 1
            else:
                pending_state = raw_state
                pending_count = 1

            need = int(self.switch_confirm_bars.get(raw_state, 3))
            hold_need = int(self.min_hold_bars.get(current, 0))
            can_switch_by_hold = current_hold >= hold_need
            if pending_count >= need and can_switch_by_hold:
                current = raw_state
                current_hold = 0
                pending_state, pending_count = None, 0
            else:
                current_hold += 1

            states.append(current)

        return pd.Series(states, index=feat.index, name="state")

    def build(self, df, start_date="2023-01-01", end_date="2026-03-16"):
        feat = self._compute_feature_frame(df)
        feat["state"] = self._apply_state_hysteresis(feat)
        range_beta = (
            (feat["state"] == self.RANGE)
            & (feat["regime_score"] > -0.05)
            & (feat["trend"] > -0.08)
            & (feat["mom_30d"] > -0.03)
            & (feat["drawdown"] > -0.30)
        )
        feat["mode"] = np.where(
            feat["state"].isin([self.EXPLOSIVE_BULL, self.TREND_BULL]) | range_beta,
            self.MODE_BETA,
            self.MODE_ALPHA,
        )

        # 可直接给执行层使用的 beta 目标建议（后续可被策略读取）
        beta_target = np.where(
            feat["state"] == self.EXPLOSIVE_BULL,
            np.clip(1.08 + 0.28 * feat["beta_score"], 1.05, 1.30),
            np.where(
                feat["state"] == self.TREND_BULL,
                np.clip(0.88 + 0.22 * feat["beta_score"], 0.85, 1.12),
                np.where(
                    feat["state"] == self.RANGE,
                    np.where(
                        feat["mode"] == self.MODE_BETA,
                        np.clip(0.95 + 0.20 * feat["beta_score"], 0.90, 1.20),
                        np.clip(0.45 + 0.20 * feat["beta_score"], 0.35, 0.70),
                    ),
                    np.clip(0.22 + 0.15 * feat["beta_score"], 0.15, 0.45),
                ),
            ),
        )
        feat["target_beta_hint"] = beta_target.astype(float)

        # 置信度：beta 与 risk 的分离度 + 规则命中强度
        raw_conf = np.abs(feat["regime_score"]).fillna(0.0)
        state_bonus = np.where(
            feat["state"] == self.EXPLOSIVE_BULL,
            np.clip((feat["trend"].fillna(0.0) - 0.08) / 0.18, 0.0, 1.0),
            np.where(
                feat["state"] == self.DEFENSIVE,
                np.clip((-feat["drawdown"].fillna(0.0) - 0.15) / 0.25, 0.0, 1.0),
                0.0,
            ),
        )
        feat["confidence"] = np.clip(0.75 * raw_conf + 0.25 * state_bonus, 0.0, 1.0)

        start_ts = pd.Timestamp(start_date + " 00:00:00")
        end_ts = pd.Timestamp(end_date + " 23:59:59")
        feat = feat.loc[(feat.index >= start_ts) & (feat.index <= end_ts)].copy()

        cols = [
            "close",
            "ma_long",
            "trend",
            "mom_30d",
            "mom_90d",
            "vol_30d_ann",
            "vol_stress",
            "breadth_long",
            "breadth_impulse",
            "drawdown",
            "dispersion_1d",
            "beta_score",
            "risk_score",
            "regime_score",
            "confidence",
            "state",
            "mode",
            "target_beta_hint",
        ]
        out = feat[cols].copy()
        out.index.name = "timestamp"
        return out

    @staticmethod
    def annual_summary(signal_df):
        rows = []
        for y in sorted(signal_df.index.year.unique()):
            d = signal_df[signal_df.index.year == y]
            if d.empty:
                continue
            total = len(d)
            rows.append(
                {
                    "year": int(y),
                    "bars": int(total),
                    "explosive_share": float((d["state"] == "EXPLOSIVE_BULL").mean()),
                    "trend_bull_share": float((d["state"] == "TREND_BULL").mean()),
                    "range_share": float((d["state"] == "RANGE").mean()),
                    "defensive_share": float((d["state"] == "DEFENSIVE").mean()),
                    "beta_mode_share": float((d["mode"] == "HOLD_BTC_BETA").mean()),
                    "alpha_mode_share": float((d["mode"] == "ALPHA_MODE").mean()),
                    "switch_count": int((d["state"] != d["state"].shift(1)).sum()),
                    "mean_regime_score": float(d["regime_score"].mean()),
                }
            )
        return pd.DataFrame(rows)
