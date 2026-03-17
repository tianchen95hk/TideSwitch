import numpy as np
import pandas as pd
from config.settings import Settings

class WhiteboxAlgo:
    def __init__(self):
        self.BTC = 'BTCUSDT'
        self.top_n = Settings.TOP_N
        self.last_log_month = None
        self.STATE_EXPLOSIVE = "EXPLOSIVE_BULL"
        self.STATE_TREND = "TREND_BULL"
        self.STATE_RANGE = "RANGE"
        self.STATE_DEFENSIVE = "DEFENSIVE"
        self.MODE_BETA = "HOLD_BTC_BETA"
        self.MODE_ALPHA = "ALPHA_MODE"

    def _factor_breadth_scale(self, df_slice):
        if 'fm_composite' not in df_slice.columns:
            return 1.0
        alts = df_slice[df_slice.index != self.BTC]
        if alts.empty:
            return 1.0
        breadth = pd.to_numeric(alts['fm_composite'], errors='coerce').dropna()
        if breadth.empty:
            return 1.0
        # 用截面平均强度做 beta 暴露微调：强则放大，弱则收敛
        score = float(np.tanh(breadth.mean() / 1.5))
        return float(np.clip(1.0 + 0.20 * score, 0.80, 1.15))

    def _allocate_satellite_weights(self, df_slice, budget):
        if budget <= 0:
            return {}

        alts = df_slice[df_slice.index != self.BTC].copy()
        if alts.empty:
            return {}

        use_factor_mode = {'fm_composite', 'fm_rank'}.issubset(alts.columns)
        if use_factor_mode:
            min_rank = float(np.clip(Settings.FACTOR_MIN_RANK, 0.50, 0.95))
            candidates = alts.dropna(subset=['fm_composite', 'fm_rank']).copy()
            candidates = candidates[candidates['fm_rank'] >= min_rank]
            candidates = candidates[candidates['fm_composite'] > 0]
            candidates = candidates.sort_values('fm_composite', ascending=False).head(self.top_n)
            if not candidates.empty:
                raw_score = (candidates['fm_rank'] - min_rank + 0.05).clip(lower=0.01)
                if 'asset_volatility' in candidates.columns:
                    vol = pd.to_numeric(candidates['asset_volatility'], errors='coerce')
                    vol = vol.replace([np.inf, -np.inf], np.nan)
                    vol = vol.fillna(vol.median())
                    vol = vol.clip(lower=0.08)
                    inv_vol = 1.0 / np.power(vol, max(Settings.FACTOR_INV_VOL_POWER, 0.0))
                    raw_score = raw_score * inv_vol

                score_sum = float(raw_score.sum())
                if np.isfinite(score_sum) and score_sum > 0:
                    return {sym: float(budget * raw_score.loc[sym] / score_sum) for sym in candidates.index}

        # 保守模式：未通过 fm_rank/fm_composite 门槛时，不配卫星仓位，预算回流 BTC。
        return {}

    def _build_market_neutral_overlay(self, df_slice, gross_budget):
        if (not Settings.ENABLE_MARKET_NEUTRAL_OVERLAY) or gross_budget <= 1e-8:
            return {}

        alts = df_slice[df_slice.index != self.BTC].copy()
        if alts.empty or (not {'fm_rank', 'fm_composite'}.issubset(alts.columns)):
            return {}

        min_names = max(int(Settings.OVERLAY_MIN_NAMES), 1)
        max_names = max(int(Settings.OVERLAY_MAX_NAMES), min_names)
        long_rank = float(np.clip(Settings.OVERLAY_LONG_RANK, 0.50, 0.99))
        short_rank = float(np.clip(Settings.OVERLAY_SHORT_RANK, 0.01, 0.50))

        candidates = alts.dropna(subset=['fm_rank', 'fm_composite']).copy()
        if candidates.empty:
            return {}

        longs = (
            candidates[(candidates['fm_rank'] >= long_rank) & (candidates['fm_composite'] > 0)]
            .sort_values('fm_composite', ascending=False)
            .head(max_names)
        )
        shorts = (
            candidates[(candidates['fm_rank'] <= short_rank) & (candidates['fm_composite'] < 0)]
            .sort_values('fm_composite', ascending=True)
            .head(max_names)
        )

        if len(longs) < min_names or len(shorts) < min_names:
            return {}

        # 可选 beta-neutral：根据 long/short 组的 beta 均值调整名义敞口，让净 beta 更接近 0
        long_notional = gross_budget * 0.5
        short_notional = gross_budget * 0.5
        if Settings.OVERLAY_BETA_NEUTRAL and ('fm_beta_to_btc' in candidates.columns):
            long_beta = pd.to_numeric(longs.get('fm_beta_to_btc'), errors='coerce').abs().replace([np.inf, -np.inf], np.nan).dropna()
            short_beta = pd.to_numeric(shorts.get('fm_beta_to_btc'), errors='coerce').abs().replace([np.inf, -np.inf], np.nan).dropna()
            if (not long_beta.empty) and (not short_beta.empty):
                beta_l = float(max(long_beta.mean(), 0.05))
                beta_s = float(max(short_beta.mean(), 0.05))
                long_notional = float(gross_budget * beta_s / (beta_l + beta_s))
                short_notional = float(max(gross_budget - long_notional, 0.0))

        long_w = long_notional / len(longs)
        short_w = -short_notional / len(shorts)
        out = {sym: float(long_w) for sym in longs.index}
        for sym in shorts.index:
            out[sym] = float(out.get(sym, 0.0) + short_w)
        return out

    def _apply_bull_beta_floor(self, weights, is_bull_regime, is_super_bull=False, forced_floor=None):
        if (not weights) or ((not is_bull_regime) and (forced_floor is None)):
            return weights

        floor = float(max(Settings.BULL_BETA_FLOOR, 0.0))
        if is_super_bull:
            floor = float(max(floor, Settings.SUPERBULL_BTC_FLOOR))
        if forced_floor is not None:
            floor = float(max(floor, forced_floor))
        btc_w = float(weights.get(self.BTC, 0.0))
        if btc_w >= floor:
            return weights

        need = floor - btc_w
        alt_syms = [sym for sym, w in weights.items() if sym != self.BTC and float(w) > 0]
        alt_sum = float(sum(float(weights[s]) for s in alt_syms))

        shift = min(need, alt_sum)
        if alt_sum > 1e-12 and shift > 0:
            scale = (alt_sum - shift) / alt_sum
            for sym in alt_syms:
                weights[sym] = float(max(0.0, weights[sym] * scale))

        weights[self.BTC] = float(btc_w + shift)
        remaining = need - shift
        if remaining > 1e-8:
            # 若卫星预算不足，则允许提高 BTC，后续由总暴露上限裁剪。
            weights[self.BTC] = float(weights[self.BTC] + remaining)

        return {k: float(v) for k, v in weights.items() if abs(float(v)) > 1e-8}

    def _is_super_bull(self, curr, is_bull_regime):
        if not is_bull_regime:
            return False
        ma = float(curr.get('ma', np.nan))
        close = float(curr.get('close', np.nan))
        if (not np.isfinite(ma)) or ma <= 0 or (not np.isfinite(close)):
            return False
        trend = close / ma - 1.0
        breadth = float(np.clip(float(curr.get('diffusion_index', 0.0)) / 100.0, -1.0, 1.0))
        drawdown = float(curr.get('drawdown', -0.2) or -0.2)
        return bool(
            trend >= float(Settings.SUPERBULL_TREND_THRESHOLD)
            and breadth >= float(Settings.SUPERBULL_BREADTH_THRESHOLD)
            and drawdown >= float(Settings.SUPERBULL_DRAWDOWN_THRESHOLD)
        )

    def _extract_regime_signal(self, curr, is_bull_regime, is_bear_regime, is_super_bull):
        raw_state = str(curr.get('state', '') or '').strip().upper()
        valid_states = {self.STATE_EXPLOSIVE, self.STATE_TREND, self.STATE_RANGE, self.STATE_DEFENSIVE}
        if raw_state not in valid_states:
            if is_super_bull:
                raw_state = self.STATE_EXPLOSIVE
            elif is_bull_regime:
                raw_state = self.STATE_TREND
            elif is_bear_regime:
                raw_state = self.STATE_DEFENSIVE
            else:
                raw_state = self.STATE_RANGE

        raw_mode = str(curr.get('mode', '') or '').strip().upper()
        if raw_mode not in {self.MODE_BETA, self.MODE_ALPHA}:
            raw_mode = self.MODE_BETA if raw_state in {self.STATE_EXPLOSIVE, self.STATE_TREND} else self.MODE_ALPHA

        conf = curr.get('confidence', np.nan)
        conf = 0.5 if pd.isna(conf) else float(conf)
        conf = float(np.clip(conf, 0.0, 1.0))

        hint = curr.get('target_beta_hint', np.nan)
        hint = np.nan if pd.isna(hint) else float(hint)
        return raw_state, raw_mode, conf, hint

    def _dynamic_beta_target(self, curr, is_bull_regime, is_super_bull, target_beta_hint=np.nan, regime_mode=None):
        if pd.notna(target_beta_hint):
            low = 0.0 if regime_mode == self.MODE_ALPHA else float(Settings.TARGET_BETA_MIN)
            beta = float(np.clip(float(target_beta_hint), low, float(Settings.TARGET_BETA_MAX)))
            if is_super_bull:
                beta = max(beta, float(Settings.SUPERBULL_BETA_MIN))
            elif is_bull_regime:
                beta = max(beta, float(Settings.BULL_BETA_MIN))
            return beta

        ma = float(curr.get('ma', np.nan))
        close = float(curr.get('close', np.nan))
        if (not np.isfinite(ma)) or ma <= 0 or (not np.isfinite(close)):
            return float(np.clip(Settings.CORE_BTC_WEIGHT, Settings.TARGET_BETA_MIN, Settings.TARGET_BETA_MAX))

        trend_strength = np.clip((close / ma - 1.0) / 0.10, -1.0, 1.0)
        curr_vol = curr.get('volatility', Settings.TARGET_VOLATILITY)
        if pd.isna(curr_vol) or curr_vol <= 0:
            curr_vol = Settings.TARGET_VOLATILITY
        vol_pressure = np.clip((float(curr_vol) / float(Settings.TARGET_VOLATILITY)) - 1.0, -1.0, 1.5)

        breadth = np.clip(float(curr.get('diffusion_index', 0.0)) / 100.0, -1.0, 1.0)
        score = (
            float(Settings.BETA_TREND_WEIGHT) * float(trend_strength)
            - float(Settings.BETA_VOL_WEIGHT) * float(vol_pressure)
            + float(Settings.BETA_BREADTH_WEIGHT) * float(breadth)
        )
        score = float(np.clip(score, -1.0, 1.0))
        beta = float(Settings.TARGET_BETA_MIN) + ((score + 1.0) * 0.5) * (
            float(Settings.TARGET_BETA_MAX) - float(Settings.TARGET_BETA_MIN)
        )
        beta = float(np.clip(beta, Settings.TARGET_BETA_MIN, Settings.TARGET_BETA_MAX))

        if is_super_bull:
            beta = max(beta, float(Settings.SUPERBULL_BETA_MIN))
        elif is_bull_regime:
            beta = max(beta, float(Settings.BULL_BETA_MIN))
        return beta

    def _blend_ratio(self, is_super_bull, is_bull_regime, is_bear_regime, regime_state=None, regime_mode=None, regime_confidence=0.5):
        if regime_state == self.STATE_EXPLOSIVE:
            base = float(Settings.REGIME_BETA_RATIO_EXPLOSIVE)
        elif regime_state == self.STATE_TREND:
            base = float(Settings.REGIME_BETA_RATIO_TREND)
        elif regime_state == self.STATE_DEFENSIVE:
            base = float(Settings.REGIME_BETA_RATIO_DEFENSIVE)
        elif regime_state == self.STATE_RANGE:
            if regime_mode == self.MODE_BETA:
                base = float(Settings.REGIME_BETA_RATIO_RANGE_BETA_MODE)
            else:
                base = float(Settings.REGIME_BETA_RATIO_RANGE_ALPHA_MODE)
        else:
            base = None

        if base is not None:
            tilt = float(Settings.REGIME_BETA_CONFIDENCE_TILT) * (float(regime_confidence) - 0.5)
            if regime_mode == self.MODE_BETA:
                base += tilt
            else:
                base -= tilt
            return float(np.clip(base, 0.02, 1.0))

        if is_super_bull:
            return float(np.clip(Settings.BLEND_BETA_SUPER_BULL, 0.0, 1.0))
        if is_bull_regime:
            return float(np.clip(Settings.BLEND_BETA_BULL, 0.0, 1.0))
        if is_bear_regime:
            return float(np.clip(Settings.BLEND_BETA_BEAR, 0.0, 1.0))
        return float(np.clip(Settings.BLEND_BETA_NEUTRAL, 0.0, 1.0))

    def _merge_weights(self, left, right, left_ratio):
        lr = float(np.clip(left_ratio, 0.0, 1.0))
        rr = 1.0 - lr
        out = {}
        for sym in set(left.keys()) | set(right.keys()):
            w = lr * float(left.get(sym, 0.0)) + rr * float(right.get(sym, 0.0))
            if abs(w) > 1e-8:
                out[sym] = float(w)
        return out

    def _sym_beta_to_btc(self, sym, df_slice):
        if sym == self.BTC:
            return 1.0
        if (sym in df_slice.index) and ('fm_beta_to_btc' in df_slice.columns):
            v = pd.to_numeric(df_slice.loc[sym].get('fm_beta_to_btc'), errors='coerce')
            if pd.notna(v) and np.isfinite(v):
                return float(v)
        return 0.70

    def _portfolio_beta_estimate(self, weights, df_slice):
        beta = 0.0
        for sym, w in (weights or {}).items():
            b = self._sym_beta_to_btc(sym, df_slice)
            beta += float(w) * float(b)
        return float(beta)

    def _enforce_beta_guard(self, weights, df_slice, regime_state, regime_mode):
        if not weights or regime_mode != self.MODE_BETA:
            return weights

        out = {k: float(v) for k, v in weights.items() if abs(float(v)) > 1e-8}
        if not out:
            return out

        if Settings.REGIME_BETA_NO_SHORT:
            for sym in list(out.keys()):
                if sym == self.BTC:
                    continue
                if out[sym] < 0:
                    out[sym] = 0.0
            out = {k: float(v) for k, v in out.items() if abs(float(v)) > 1e-8}

        beta_floor = float(Settings.REGIME_NET_BETA_FLOOR_TREND)
        beta_target = float(Settings.REGIME_NET_BETA_TARGET_TREND)
        if regime_state == self.STATE_EXPLOSIVE:
            beta_floor = float(Settings.REGIME_NET_BETA_FLOOR_EXPLOSIVE)
            beta_target = float(Settings.REGIME_NET_BETA_TARGET_EXPLOSIVE)

        # 优先把正向卫星仓位回流到 BTC，先满足 floor
        beta_now = self._portfolio_beta_estimate(out, df_slice)
        if beta_now < beta_floor:
            need = beta_floor - beta_now
            positive_alts = [s for s, w in out.items() if s != self.BTC and w > 0]
            for sym in sorted(positive_alts, key=lambda x: out[x], reverse=True):
                if need <= 1e-9:
                    break
                b = max(self._sym_beta_to_btc(sym, df_slice), 0.05)
                unit_gain = max(1.0 - b, 1e-6)
                reduce_w = min(out[sym], need / unit_gain)
                out[sym] -= reduce_w
                out[self.BTC] = float(out.get(self.BTC, 0.0) + reduce_w)
                need -= reduce_w * unit_gain

        # floor 仍不够时，增加 BTC 主仓（后续由总暴露上限裁剪）
        beta_now = self._portfolio_beta_estimate(out, df_slice)
        if beta_now < beta_floor:
            out[self.BTC] = float(out.get(self.BTC, 0.0) + (beta_floor - beta_now))

        # 在 beta 模式中尽量靠近 target（但不主动加空头）
        beta_now = self._portfolio_beta_estimate(out, df_slice)
        if beta_now < beta_target:
            out[self.BTC] = float(out.get(self.BTC, 0.0) + (beta_target - beta_now))

        return {k: float(v) for k, v in out.items() if abs(float(v)) > 1e-8}

    def _build_carry_overlay(self, df_slice, gross_budget):
        if gross_budget <= 1e-8:
            return {}
        alts = df_slice[df_slice.index != self.BTC].copy()
        if alts.empty:
            return {}

        funding_col = None
        if 'fundingRate' in alts.columns:
            funding_col = 'fundingRate'
        elif 'funding_rate' in alts.columns:
            funding_col = 'funding_rate'
        if funding_col is None:
            return {}

        candidates = alts.dropna(subset=[funding_col]).copy()
        if candidates.empty:
            return {}

        long_leg = candidates.sort_values(funding_col, ascending=True).head(max(Settings.OVERLAY_MIN_NAMES, 2))
        short_leg = candidates.sort_values(funding_col, ascending=False).head(max(Settings.OVERLAY_MIN_NAMES, 2))
        if long_leg.empty or short_leg.empty:
            return {}

        half = gross_budget * 0.5
        lw = half / len(long_leg)
        sw = -half / len(short_leg)
        out = {sym: float(lw) for sym in long_leg.index}
        for sym in short_leg.index:
            out[sym] = float(out.get(sym, 0.0) + sw)
        return out

    def _bull_trailing_confirmed(self, btc_history, confirm_bars):
        n = max(int(confirm_bars), 1)
        if len(btc_history) < n:
            return False, 0
        tail = btc_history.iloc[-n:]
        vol_th = float(Settings.TARGET_VOLATILITY) * float(Settings.TRAILING_VOL_SPIKE_MULT)
        cond_trend = (tail['close'] < tail['short_ma']).fillna(False)
        cond_vol = (tail['volatility'] > vol_th).fillna(False)
        cond_breadth = (tail['diffusion_index'] < 0).fillna(False)
        cond_all = cond_trend & cond_vol & cond_breadth
        score = int(cond_trend.iloc[-1]) + int(cond_vol.iloc[-1]) + int(cond_breadth.iloc[-1])
        return bool(cond_all.all()), score

    def _build_basis_term_overlay(self, df_slice, gross_budget):
        if gross_budget <= 1e-8:
            return {}
        alts = df_slice[df_slice.index != self.BTC].copy()
        need_cols = {'fm_basis_spread_24', 'fm_term_structure_proxy_12_72'}
        if alts.empty or (not need_cols.issubset(alts.columns)):
            return {}

        candidates = alts.dropna(subset=['fm_basis_spread_24', 'fm_term_structure_proxy_12_72']).copy()
        if candidates.empty:
            return {}
        score = 0.6 * pd.to_numeric(candidates['fm_basis_spread_24'], errors='coerce') + 0.4 * pd.to_numeric(candidates['fm_term_structure_proxy_12_72'], errors='coerce')
        candidates['bt_score'] = score.replace([np.inf, -np.inf], np.nan)
        candidates = candidates.dropna(subset=['bt_score'])
        if candidates.empty:
            return {}

        n = max(Settings.OVERLAY_MIN_NAMES, 2)
        longs = candidates.sort_values('bt_score', ascending=False).head(n)
        shorts = candidates.sort_values('bt_score', ascending=True).head(n)
        if longs.empty or shorts.empty:
            return {}

        half = gross_budget * 0.5
        lw = half / len(longs)
        sw = -half / len(shorts)
        out = {sym: float(lw) for sym in longs.index}
        for sym in shorts.index:
            out[sym] = float(out.get(sym, 0.0) + sw)
        return out

    def _build_beta_engine(self, df_slice, curr, next_state, is_bull_regime, is_super_bull, target_beta_hint=np.nan, regime_mode=None):
        weights = {}
        if next_state == 0:
            return weights

        target_beta = self._dynamic_beta_target(
            curr=curr,
            is_bull_regime=is_bull_regime,
            is_super_bull=is_super_bull,
            target_beta_hint=target_beta_hint,
            regime_mode=regime_mode
        )
        current_vol = curr.get('volatility', Settings.TARGET_VOLATILITY)
        if pd.isna(current_vol) or current_vol <= 0:
            current_vol = Settings.TARGET_VOLATILITY
        if regime_mode == self.MODE_BETA:
            vol_discount = float(np.clip(Settings.TARGET_VOLATILITY / float(current_vol), 0.85, 1.25))
            if is_super_bull:
                vol_discount = float(max(vol_discount, 0.95))
        else:
            vol_discount = float(np.clip(Settings.TARGET_VOLATILITY / float(current_vol), 0.55, 1.20))

        if next_state == -1:
            short_scale = float(np.clip(target_beta * vol_discount, Settings.SHORT_MIN_WEIGHT, Settings.SHORT_MAX_WEIGHT))
            weights[self.BTC] = -short_scale
            return weights

        gross_long = max(target_beta * vol_discount, 0.0)
        # beta 模式尽量用纯 BTC 承载方向暴露，避免卫星仓稀释 beta
        if regime_mode == self.MODE_BETA:
            return {self.BTC: float(gross_long)} if gross_long > 1e-8 else {}

        core_btc = gross_long * float(np.clip(Settings.CORE_BTC_WEIGHT, 0.0, 1.0))
        sat_budget = max(gross_long - core_btc, 0.0)
        weights[self.BTC] = float(core_btc)

        satellite = self._allocate_satellite_weights(df_slice=df_slice, budget=sat_budget)
        if satellite:
            for sym, w in satellite.items():
                weights[sym] = float(weights.get(sym, 0.0) + w)
        else:
            weights[self.BTC] = float(weights.get(self.BTC, 0.0) + sat_budget)

        if is_bull_regime:
            bull_floor = max(float(Settings.BULL_BETA_FLOOR), float(Settings.BULL_BETA_MIN))
            if is_super_bull:
                bull_floor = max(bull_floor, float(Settings.SUPERBULL_BTC_FLOOR))
            if weights.get(self.BTC, 0.0) < bull_floor:
                need = bull_floor - float(weights.get(self.BTC, 0.0))
                positive_alts = [s for s, w in weights.items() if s != self.BTC and w > 0]
                alt_sum = float(sum(weights[s] for s in positive_alts))
                shift = min(need, alt_sum)
                if shift > 0 and alt_sum > 1e-9:
                    scale = (alt_sum - shift) / alt_sum
                    for s in positive_alts:
                        weights[s] = float(max(0.0, weights[s] * scale))
                weights[self.BTC] = float(weights.get(self.BTC, 0.0) + shift)
                if need > shift:
                    weights[self.BTC] = float(weights[self.BTC] + (need - shift))
        return {k: float(v) for k, v in weights.items() if abs(float(v)) > 1e-8}

    def _build_alpha_engine(self, df_slice, is_super_bull, is_bull_regime, is_bear_regime, regime_state=None, regime_mode=None, regime_confidence=0.5):
        if not Settings.ENABLE_MARKET_NEUTRAL_OVERLAY:
            return {}

        if is_super_bull and Settings.DISABLE_ALPHA_IN_SUPERBULL:
            return {}
        if is_bull_regime and Settings.DISABLE_ALPHA_IN_BULL:
            return {}

        if is_super_bull:
            regime_scale = float(Settings.ALPHA_ENGINE_SCALE_SUPER_BULL)
        elif is_bull_regime:
            regime_scale = float(Settings.ALPHA_ENGINE_SCALE_BULL)
        elif is_bear_regime:
            regime_scale = float(Settings.ALPHA_ENGINE_SCALE_BEAR)
        else:
            regime_scale = float(Settings.ALPHA_ENGINE_SCALE_NEUTRAL)
        gross_budget = float(np.clip(Settings.ALPHA_ENGINE_MAX_GROSS * regime_scale, 0.0, 1.0))

        # Regime 进一步控制 alpha 预算：beta 模式下抑制，alpha 模式下放大
        conf = float(np.clip(regime_confidence, 0.0, 1.0))
        if regime_mode == self.MODE_BETA:
            gross_budget = 0.0
        elif regime_mode == self.MODE_ALPHA:
            boost = 1.0 + max(conf - 0.40, 0.0) * 0.35
            if regime_state == self.STATE_DEFENSIVE:
                boost += 0.10
            gross_budget = float(min(gross_budget * boost, 1.0))

        if gross_budget <= 1e-8:
            return {}

        w_xs = float(np.clip(Settings.ALPHA_XS_WEIGHT, 0.0, 1.0))
        w_carry = float(np.clip(Settings.ALPHA_CARRY_WEIGHT, 0.0, 1.0))
        w_basis_term = float(np.clip(Settings.ALPHA_BASIS_TERM_WEIGHT, 0.0, 1.0))
        w_sum = max(w_xs + w_carry + w_basis_term, 1e-9)
        xs_budget = gross_budget * (w_xs / w_sum)
        carry_budget = gross_budget * (w_carry / w_sum)
        basis_term_budget = gross_budget * (w_basis_term / w_sum)

        xs = self._build_market_neutral_overlay(df_slice=df_slice, gross_budget=xs_budget)
        carry = self._build_carry_overlay(df_slice=df_slice, gross_budget=carry_budget)
        basis_term = self._build_basis_term_overlay(df_slice=df_slice, gross_budget=basis_term_budget)
        out = {}
        for sym, w in xs.items():
            out[sym] = float(out.get(sym, 0.0) + w)
        for sym, w in carry.items():
            out[sym] = float(out.get(sym, 0.0) + w)
        for sym, w in basis_term.items():
            out[sym] = float(out.get(sym, 0.0) + w)
        return {k: float(v) for k, v in out.items() if abs(float(v)) > 1e-8}
        
    def compute_signal(self, df_slice, btc_history, current_state, entry_price, highest_price, lowest_price=None):
        """
        🚀 4Alpha Pro: 动态平滑最终版
        """
        if btc_history.empty: return {}, 0
        curr = btc_history.iloc[-1]
        
        # 🛡️ 数据防御
        if pd.isna(curr.get('ma')): return {}, 0

        # === 1. 宏观定调 (Regime) ===
        # 使用 MA 附近的缓冲带，减少频繁切换造成的噪音交易
        regime_band = Settings.REGIME_HYSTERESIS_PCT
        bull_threshold = curr['ma'] * (1.0 + regime_band)
        bear_threshold = curr['ma'] * (1.0 - regime_band)
        base_is_bull_regime = curr['close'] > bull_threshold
        base_is_bear_regime = curr['close'] < bear_threshold
        base_is_super_bull = self._is_super_bull(curr=curr, is_bull_regime=base_is_bull_regime)

        regime_state, regime_mode, regime_conf, target_beta_hint = self._extract_regime_signal(
            curr=curr,
            is_bull_regime=base_is_bull_regime,
            is_bear_regime=base_is_bear_regime,
            is_super_bull=base_is_super_bull,
        )

        # 以四状态信号为主，原 MA 规则为兜底
        if regime_state == self.STATE_DEFENSIVE:
            is_bull_regime = False
            is_bear_regime = True
            is_super_bull = False
        elif regime_mode == self.MODE_BETA:
            is_bull_regime = True
            is_bear_regime = False
            is_super_bull = (regime_state == self.STATE_EXPLOSIVE) or base_is_super_bull
        else:
            is_bull_regime = bool(base_is_bull_regime)
            is_bear_regime = bool(base_is_bear_regime and (not base_is_bull_regime))
            is_super_bull = bool((regime_state == self.STATE_EXPLOSIVE) or base_is_super_bull)
        short_ma = curr.get('short_ma', curr['ma'])
        if pd.isna(short_ma) or short_ma <= 0:
            short_ma = curr['ma']
        short_band = Settings.SHORT_FAST_MA_BAND_PCT
        short_bear_threshold = short_ma * (1.0 - short_band)
        short_bull_threshold = short_ma * (1.0 + short_band)
        short_trend_down = curr['close'] < short_bear_threshold
        short_trend_up = curr['close'] > short_bull_threshold
        
        # 调试日志
        current_date = btc_history.index[-1]
        if self.last_log_month != current_date.month:
            vol_info = f"Vol: {curr.get('volatility', 0):.1%}"
            if regime_state == self.STATE_EXPLOSIVE:
                regime_str = "🚀 爆发牛"
            elif regime_state == self.STATE_TREND:
                regime_str = "🐂 趋势牛"
            elif regime_state == self.STATE_DEFENSIVE:
                regime_str = "🛡️ 防守"
            else:
                regime_str = "⚖️ 震荡"
            print(
                f"   📅 [{current_date.strftime('%Y-%m')}] {regime_str} | "
                f"Mode: {regime_mode} | Conf: {regime_conf:.2f} | "
                f"BetaHint: {target_beta_hint if pd.notna(target_beta_hint) else np.nan:.2f} | "
                f"{vol_info} | Price: {curr['close']:.0f}"
            )
            self.last_log_month = current_date.month

        # === 2. 风控检查 (熔断 + 移动止盈) ===
        long_trailing_breach = False
        long_trailing_drawdown = 0.0
        if current_state == 1 and entry_price is not None:
            # A. 硬止损 (8%)
            pct_change = (curr['close'] - entry_price) / entry_price
            if pct_change < -Settings.STOP_LOSS_PCT:
                print(f"      🛑 硬止损触发! 亏损: {pct_change:.2%}")
                return {}, 0
            
            # B. 移动止盈：在强牛市需要额外确认，避免过早下车
            if highest_price and highest_price > entry_price:
                drawdown_from_peak = (curr['close'] - highest_price) / highest_price
                if drawdown_from_peak < -Settings.TRAILING_STOP_PCT:
                    long_trailing_breach = True
                    long_trailing_drawdown = float(drawdown_from_peak)

        if current_state == -1 and entry_price is not None:
            # A. 空头硬止损 (价格反向上涨超过阈值)
            short_pnl = (entry_price - curr['close']) / entry_price
            if short_pnl < -Settings.STOP_LOSS_PCT:
                print(f"      🛑 空头止损触发! 亏损: {short_pnl:.2%}")
                return {}, 0

            # B. 空头移动止盈：从最低点反弹超过阈值则平空锁利
            if lowest_price is not None and entry_price is not None and lowest_price < entry_price:
                bounce_from_low = (curr['close'] - lowest_price) / lowest_price
                if bounce_from_low > Settings.TRAILING_STOP_PCT:
                    print(f"      🛡️ 空头移动止盈触发! 低点反弹: {bounce_from_low:.2%} (锁定利润)")
                    return {}, 0

        # === 3. 信号计算 ===
        next_state = current_state

        if len(btc_history) > Settings.BUY_Y_RISE_DAYS + 1:
            y_idx_start = btc_history['y_index'].iloc[-(Settings.BUY_Y_RISE_DAYS + 1)]
            cond_y_rise = curr['y_index'] > y_idx_start
        else:
            cond_y_rise = False

        if len(btc_history) > Settings.SELL_Y_FALL_DAYS + 1:
            y_idx_start_s = btc_history['y_index'].iloc[-(Settings.SELL_Y_FALL_DAYS + 1)]
            cond_y_fall = curr['y_index'] < y_idx_start_s
        else:
            cond_y_fall = False

        trailing_exit_long = False
        trailing_exit_score = 0
        if current_state == 1 and is_bull_regime and Settings.BULL_REGIME_CONDITIONAL_TRAILING:
            trailing_exit_long, trailing_exit_score = self._bull_trailing_confirmed(
                btc_history=btc_history,
                confirm_bars=Settings.TRAILING_CONFIRM_BARS_BULL
            )
        elif current_state == 1 and long_trailing_breach:
            curr_vol = curr.get('volatility', np.nan)
            vol_spike = pd.notna(curr_vol) and (
                float(curr_vol) > float(Settings.TARGET_VOLATILITY) * float(Settings.TRAILING_VOL_SPIKE_MULT)
            )
            trend_weak = curr['close'] < short_ma
            breadth_weak = curr['diffusion_index'] < 0
            trailing_exit_score = int(trend_weak) + int(breadth_weak) + int(cond_y_fall) + int(vol_spike)
            need_score = (
                int(Settings.TRAILING_CONFIRM_SCORE_BULL)
                if is_bull_regime
                else int(Settings.TRAILING_CONFIRM_SCORE_OTHER)
            )
            trailing_exit_long = trailing_exit_score >= max(need_score, 1)

        # beta 模式下，非极端破坏不轻易触发止盈下车，避免牛市 beta 丢失
        if (
            current_state == 1
            and trailing_exit_long
            and regime_mode == self.MODE_BETA
        ):
            severe_break = (
                (curr['close'] < bear_threshold)
                and (float(curr.get('drawdown', 0.0) or 0.0) < -0.15)
                and cond_y_fall
            )
            if not severe_break:
                trailing_exit_long = False
        
        if is_bull_regime:
            # 🐂 牛市逻辑: 趋势跟随
            # 只要空仓，立刻进场！
            if current_state == 0:
                print(f"      🚀 确认牛市趋势，进场做多")
                next_state = 1
            elif current_state == -1:
                # 不允许同bar内从空头直接翻多，先平空，下一根bar再决定是否做多
                print(f"      🔄 牛市信号出现，先平空")
                next_state = 0
            
            # 只有明显跌回 MA 下方才离场（用 bear_threshold 避免微小抖动）
            elif current_state == 1:
                if trailing_exit_long:
                    print(
                        f"      🛡️ Regime止盈触发(牛市确认{trailing_exit_score})! "
                        f"高点回撤: {long_trailing_drawdown:.2%}"
                    )
                    next_state = 0
                elif curr['close'] < bear_threshold and regime_mode != self.MODE_BETA:
                    print(f"      📉 跌破生命线，离场")
                    next_state = 0
                    
        elif is_bear_regime:
            # 🐻 熊市逻辑: 均值回归
            cond_drawdown = curr['drawdown'] < Settings.BUY_DRAWDOWN
            cond_diff_low = curr['diffusion_index'] < Settings.BUY_DIFFUSION
            is_buy_signal = cond_drawdown and cond_diff_low and cond_y_rise
            allow_directional_short = (regime_state == self.STATE_DEFENSIVE) and (regime_conf >= 0.65)

            is_short_signal = (
                allow_directional_short
                and
                short_trend_down
                and (curr['diffusion_index'] > Settings.SELL_DIFFUSION)
                and cond_y_fall
            )

            upper_band = curr.get('upper_band')
            hit_upper_band = pd.notna(upper_band) and (curr['close'] >= upper_band * Settings.SELL_BB_FACTOR)
            momentum_exhausted = (curr['diffusion_index'] > Settings.SELL_DIFFUSION) and cond_y_fall
            
            # 抄底
            if is_buy_signal and current_state == 0:
                 print("      💎 熊市抄底信号完成，进场!")
                 next_state = 1

            elif is_buy_signal and current_state == -1:
                 # 不允许从空头直接翻多，先平空
                 print("      🔄 抄底信号出现，先平空")
                 next_state = 0

            # 做空：仅在空仓时开空，且不与抄底信号冲突
            elif is_short_signal and current_state == 0:
                 print("      📉 熊市空头信号，开空!")
                 next_state = -1

            elif is_short_signal and current_state == 1:
                 # 不允许从多头直接翻空，先平多
                 print("      🔄 空头信号出现，先平多")
                 next_state = 0

            # 熊市里做多的退出：反弹触发止盈/动量衰竭
            elif current_state == 1:
                 if trailing_exit_long or hit_upper_band or momentum_exhausted:
                     print("      ✅ 熊市反弹达标，平多锁利")
                     next_state = 0

            # 空头退出：动量反转
            elif current_state == -1:
                 if (not allow_directional_short) or short_trend_up or (curr['diffusion_index'] < 0 and cond_y_rise):
                     print("      🔄 空头动量反转/快均线回收，平空")
                     next_state = 0

        else:
            # ⚖️ 中性区间：允许空头使用快均线趋势择时（不改变主多头机制）
            neutral_short_signal = (
                Settings.ENABLE_NEUTRAL_SHORT
                and regime_mode == self.MODE_ALPHA
                and regime_state == self.STATE_DEFENSIVE
                and regime_conf >= 0.65
                and short_trend_down
                and (curr['diffusion_index'] > Settings.SELL_DIFFUSION)
                and cond_y_fall
            )
            if current_state == 0 and neutral_short_signal:
                print("      ⚡ 中性区间快线转弱，试探性开空")
                next_state = -1
            elif current_state == 1 and trailing_exit_long:
                print(
                    f"      🛡️ Regime止盈触发(中性确认{trailing_exit_score})! "
                    f"高点回撤: {long_trailing_drawdown:.2%}"
                )
                next_state = 0
            elif current_state == 1 and curr['close'] < curr['ma']:
                print("      ⚖️ 中性区间跌破均线，平多")
                next_state = 0
            elif current_state == -1 and short_trend_up:
                print("      ⚖️ 中性区间快线回收，平空")
                next_state = 0

        # 四状态处于 beta 模式时，优先保持多头 beta 底仓（空仓 -> 做多）
        if next_state == 0 and regime_mode == self.MODE_BETA and current_state != -1:
            next_state = 1

        # ==========================================
        # ⚖️ 4. 动态仓位构建 (核心平滑器)
        # ==========================================
        beta_weights = self._build_beta_engine(
            df_slice=df_slice,
            curr=curr,
            next_state=next_state,
            is_bull_regime=is_bull_regime,
            is_super_bull=is_super_bull,
            target_beta_hint=target_beta_hint,
            regime_mode=regime_mode,
        )
        alpha_weights = self._build_alpha_engine(
            df_slice=df_slice,
            is_super_bull=is_super_bull,
            is_bull_regime=is_bull_regime,
            is_bear_regime=is_bear_regime,
            regime_state=regime_state,
            regime_mode=regime_mode,
            regime_confidence=regime_conf,
        )
        beta_ratio = self._blend_ratio(
            is_super_bull=is_super_bull,
            is_bull_regime=is_bull_regime,
            is_bear_regime=is_bear_regime,
            regime_state=regime_state,
            regime_mode=regime_mode,
            regime_confidence=regime_conf,
        )

        if next_state == -1:
            # 空头阶段不叠加 alpha 引擎，避免净敞口被冲淡
            weights = dict(beta_weights)
        elif next_state == 0 and current_state == 0:
            # 纯 alpha 引擎阶段（震荡/下行更依赖低相关 alpha）
            weights = dict(alpha_weights) if regime_mode == self.MODE_ALPHA else dict(beta_weights)
        else:
            weights = self._merge_weights(beta_weights, alpha_weights, beta_ratio)

        forced_floor = None
        if regime_state == self.STATE_EXPLOSIVE:
            forced_floor = max(
                float(Settings.REGIME_BTC_FLOOR_EXPLOSIVE),
                float(Settings.SUPERBULL_BTC_FLOOR),
                float(Settings.BULL_BETA_MIN),
            )
        elif regime_state == self.STATE_TREND:
            forced_floor = max(
                float(Settings.REGIME_BTC_FLOOR_TREND),
                float(Settings.BULL_BETA_FLOOR),
            )
        elif regime_mode == self.MODE_BETA:
            forced_floor = max(float(Settings.BULL_BETA_FLOOR), float(Settings.BULL_BETA_MIN))

        weights = self._apply_bull_beta_floor(
            weights=weights,
            is_bull_regime=is_bull_regime,
            is_super_bull=is_super_bull,
            forced_floor=forced_floor,
        )
        weights = self._enforce_beta_guard(
            weights=weights,
            df_slice=df_slice,
            regime_state=regime_state,
            regime_mode=regime_mode,
        )
        weights = {k: float(v) for k, v in weights.items() if abs(float(v)) > 1e-8}

        return weights, next_state
