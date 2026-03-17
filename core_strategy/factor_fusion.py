from __future__ import annotations

import numpy as np
import pandas as pd


EPS = 1e-9
ANNUALIZATION_4H = np.sqrt(365 * 6)


def _cross_sectional_zscore(panel: pd.DataFrame, col: str) -> pd.Series:
    grouped = panel.groupby(level="timestamp")[col]
    mean = grouped.transform("mean")
    std = grouped.transform("std")
    z = (panel[col] - mean) / (std + EPS)
    return z.replace([np.inf, -np.inf], np.nan)


def _normalize_weights(weights: dict[str, float]) -> dict[str, float]:
    if not weights:
        return {}
    total = float(sum(abs(v) for v in weights.values()))
    if total <= EPS:
        return {}
    return {k: float(v) / total for k, v in weights.items()}


def enrich_factor_mining_features(
    panel: pd.DataFrame,
    signal_smooth_span: int = 8,
) -> pd.DataFrame:
    """
    将 FactorMining 的多因子截面思想注入 TideSwitch 数据面板。
    输入输出均为 MultiIndex(timestamp, symbol) 的 DataFrame。
    """
    if panel.empty:
        return panel

    out = panel.sort_index().copy()
    g_symbol = out.groupby(level="symbol", sort=False)

    out["momentum"] = g_symbol["close"].pct_change(periods=24)
    out["fm_mom_24"] = g_symbol["close"].pct_change(24)
    out["fm_mom_72"] = g_symbol["close"].pct_change(72)
    out["fm_short_reversal_6"] = -g_symbol["close"].pct_change(6)

    mean_48 = g_symbol["close"].transform(lambda s: s.rolling(48, min_periods=24).mean())
    std_48 = g_symbol["close"].transform(lambda s: s.rolling(48, min_periods=24).std())
    out["fm_volatility_adjusted_trend_48"] = (out["close"] - mean_48) / (std_48 + EPS)

    vol_mean_48 = g_symbol["volume"].transform(lambda s: s.rolling(48, min_periods=24).mean())
    out["fm_volume_shock_48"] = out["volume"] / (vol_mean_48 + EPS) - 1.0

    funding_col = None
    if "fundingRate" in out.columns:
        funding_col = "fundingRate"
    elif "funding_rate" in out.columns:
        funding_col = "funding_rate"
    if funding_col is not None:
        rolling_funding = g_symbol[funding_col].transform(
            lambda s: s.rolling(9, min_periods=3).mean()
        )
        out["fm_funding_crowding_unwind_9"] = -rolling_funding
        # carry: 资金费率越高代表多头越拥挤，反向更优
        out["fm_carry_funding_9"] = -rolling_funding
    else:
        out["fm_funding_crowding_unwind_9"] = 0.0
        out["fm_carry_funding_9"] = 0.0

    # perp-spot / basis 因子：若有 mark/index 用真实价差；否则用价格偏离 EMA 代理
    if {"markPrice", "indexPrice"}.issubset(out.columns):
        mark = pd.to_numeric(out["markPrice"], errors="coerce")
        index = pd.to_numeric(out["indexPrice"], errors="coerce")
        out["fm_basis_spread_24"] = -(mark - index) / (index + EPS)
    else:
        ema_24 = g_symbol["close"].transform(lambda s: s.ewm(span=24, adjust=False, min_periods=8).mean())
        out["fm_basis_spread_24"] = -(out["close"] - ema_24) / (ema_24 + EPS)

    # term-structure proxy：短周期动量与长周期动量的斜率差（可映射到期限结构变化）
    short_mom = g_symbol["close"].pct_change(12)
    long_mom = g_symbol["close"].pct_change(72)
    out["fm_term_structure_proxy_12_72"] = short_mom - long_mom

    out["ret_1"] = g_symbol["close"].pct_change()

    # 每个资产相对 BTC 的滚动 beta（用于市场中性腿做 beta-neutral）
    if "BTCUSDT" in out.index.get_level_values("symbol"):
        btc_ret = out.xs("BTCUSDT", level="symbol")["ret_1"]
        beta_parts = []
        for sym, g in out.groupby(level="symbol", sort=False):
            ts = g.index.get_level_values("timestamp")
            s_ret = pd.Series(pd.to_numeric(g["ret_1"], errors="coerce").values, index=g.index)
            if sym == "BTCUSDT":
                beta_parts.append(pd.Series(1.0, index=g.index))
                continue
            b_ret = pd.Series(pd.to_numeric(btc_ret.reindex(ts), errors="coerce").values, index=g.index)
            cov = s_ret.rolling(90, min_periods=30).cov(b_ret)
            var = b_ret.rolling(90, min_periods=30).var()
            beta = cov / (var + EPS)
            beta_parts.append(beta)
        out["fm_beta_to_btc"] = pd.concat(beta_parts).sort_index()
    else:
        out["fm_beta_to_btc"] = np.nan

    short_vol = g_symbol["ret_1"].transform(lambda s: s.rolling(24, min_periods=12).std())
    long_vol = g_symbol["ret_1"].transform(lambda s: s.rolling(168, min_periods=48).std())
    out["fm_vol_regime_reversion_24_168"] = -(short_vol / (long_vol + EPS) - 1.0)

    factor_weights = _normalize_weights(
        {
            "fm_mom_24": 0.22,
            "fm_mom_72": 0.22,
            "fm_short_reversal_6": 0.14,
            "fm_volatility_adjusted_trend_48": 0.18,
            "fm_volume_shock_48": 0.10,
            "fm_funding_crowding_unwind_9": 0.06,
            "fm_vol_regime_reversion_24_168": 0.08,
            "fm_carry_funding_9": 0.08,
            "fm_basis_spread_24": 0.06,
            "fm_term_structure_proxy_12_72": 0.06,
        }
    )

    composite = pd.Series(0.0, index=out.index, dtype=float)
    for factor_col, weight in factor_weights.items():
        z_col = f"{factor_col}_z"
        out[z_col] = _cross_sectional_zscore(out, factor_col)
        composite = composite.add(out[z_col].fillna(0.0) * weight, fill_value=0.0)

    out["fm_composite_raw"] = composite.replace([np.inf, -np.inf], np.nan)
    span = max(int(signal_smooth_span), 1)
    if span > 1:
        out["fm_composite"] = g_symbol["fm_composite_raw"].transform(
            lambda s: s.ewm(span=span, adjust=False, min_periods=1).mean()
        )
    else:
        out["fm_composite"] = out["fm_composite_raw"]

    out["fm_rank"] = out.groupby(level="timestamp")["fm_composite"].rank(pct=True)
    out["asset_volatility"] = (
        g_symbol["ret_1"].transform(lambda s: s.rolling(180, min_periods=60).std()) * ANNUALIZATION_4H
    )
    return out
