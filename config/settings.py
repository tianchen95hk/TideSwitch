# config/settings.py
import os
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(BASE_DIR, '.env'))

class Settings:
    # === 基础配置 ===
    DB_PATH = os.path.join(BASE_DIR, 'database')
    OUTPUT_PATH = os.path.join(BASE_DIR, 'outputs')
    FACTOR_FILE = os.path.join(DB_PATH, 'market_data_4h.parquet')
    
    API_KEY = os.getenv("BINANCE_API_KEY", "")
    SECRET_KEY = os.getenv("BINANCE_SECRET_KEY", "")
    USE_PROXY = str(os.getenv("USE_PROXY", "False")).lower() == "true"
    PROXY_URL = os.getenv("PROXY_URL", "")

    LIMIT = 7400 
    TIMEFRAME = '4h'
    target_symbols_count = 40
    FUNDING_HISTORY_MAX_PAGES = int(os.getenv("FUNDING_HISTORY_MAX_PAGES", "50"))
    # 扩大回测区间，覆盖 2024 牛市
    BACKTEST_START_DATE = "2024-01-01" 
    BACKTEST_END_DATE = "2026-03-12"
    INITIAL_CAPITAL = 100000
    TAKER_FEE = 0.001
    
    # === 策略参数 ===
    # 核心牛熊分界线：155天 (STH-RP)
    # 155 * 6 = 930
    MA_WINDOW = 930
    # 牛熊分界缓冲带：避免价格在 MA 附近抖动导致频繁切换
    REGIME_HYSTERESIS_PCT = 0.003
    # SuperBull 判定：牛市中的加速上行状态
    SUPERBULL_TREND_THRESHOLD = float(os.getenv("SUPERBULL_TREND_THRESHOLD", "0.10"))
    SUPERBULL_BREADTH_THRESHOLD = float(os.getenv("SUPERBULL_BREADTH_THRESHOLD", "0.15"))
    SUPERBULL_DRAWDOWN_THRESHOLD = float(os.getenv("SUPERBULL_DRAWDOWN_THRESHOLD", "-0.06"))
    # 空头快速趋势线（仅用于空头入场/退出过滤，不改变主 TideSwitch 机制）
    SHORT_FAST_MA_WINDOW = 96
    SHORT_FAST_MA_BAND_PCT = 0.0
    ENABLE_NEUTRAL_SHORT = True
    BB_STD = 2.0
    
    # 短期指数移动平均线长度
    DIFF_MA_SHORT = 12  # 12 bars = 0.5 days (4h线)
    
    # 长期指数移动平均线长度
    DIFF_MA_LONG = 24  # 26 bars = 1 day (4h线)
    
    # RSI 窗口大小
    DIFF_RSI_WIN = 14
    
    # 价格变化率窗口大小，用于计算走势向上和向下
    Y_ROC_PERIOD = 14
    
    # 标准差窗口大小，用于计算波动率
    Y_ZSCORE_WIN = 30
    
    # === 交易触发 ===
    BUY_DRAWDOWN = -0.20 #当价格相对近期高点的回撤达到 -20%（即跌了 20%）以上，才允许进入”考虑买入”的候选区间
    BUY_DIFFUSION = -20 #当扩散指标低于 -20，表示市场处在更偏“恐慌/广泛走弱”的区间，作为买入的过滤条件之一
    BUY_Y_RISE_DAYS = 2 #要求 Y 指标连续上行达到 2 个 bar 才确认买入（在 4h 级别就是连续 2 根 4h K）
    
    SELL_BB_FACTOR = 1.0 #当价格突破布林带上限时，触发卖出
    SELL_DIFFUSION = 20 #当扩散指标高于 20，表示市场进入偏“亢奋/广泛走强”的区间，作为卖出/止盈条件之一
    SELL_Y_FALL_DAYS = 2 #要求 Y 指标连续下跌达到 2 个 bar 才确认卖出（在 4h 级别就是连续 2 根 4h K）
    
    # === 仓位与风控 (核心优化) ===
    TOP_N = 1
    CORE_BTC_WEIGHT = 0.9
    # 动态 beta 目标范围
    TARGET_BETA_MIN = float(os.getenv("TARGET_BETA_MIN", "0.30"))
    TARGET_BETA_MAX = float(os.getenv("TARGET_BETA_MAX", "1.30"))
    # 强牛市 beta 地板（引擎A）
    BULL_BETA_MIN = float(os.getenv("BULL_BETA_MIN", "1.00"))
    SUPERBULL_BETA_MIN = float(os.getenv("SUPERBULL_BETA_MIN", "1.15"))
    SUPERBULL_BTC_FLOOR = float(os.getenv("SUPERBULL_BTC_FLOOR", "1.10"))
    # 动态 beta 因子权重: trend/vol/breadth
    BETA_TREND_WEIGHT = float(os.getenv("BETA_TREND_WEIGHT", "0.50"))
    BETA_VOL_WEIGHT = float(os.getenv("BETA_VOL_WEIGHT", "0.30"))
    BETA_BREADTH_WEIGHT = float(os.getenv("BETA_BREADTH_WEIGHT", "0.20"))
    
    # 1. 硬止损 (保命)
    STOP_LOSS_PCT = 0.08  

    # 止损后冷却期：硬止损触发后，必须等待N个bar才允许重新开仓
    # TIMEFRAME='4h' 时：12 bars = 2 天
    STOPLOSS_COOLDOWN_BARS = 30
    STOPLOSS_COOLDOWN_BARS_BETA = int(os.getenv("STOPLOSS_COOLDOWN_BARS_BETA", "8"))
    STOPLOSS_COOLDOWN_BARS_ALPHA = int(os.getenv("STOPLOSS_COOLDOWN_BARS_ALPHA", "30"))
    
    # 2. [新增] 移动止盈 (锁利)
    # 从高点回撤 10% 离场，防止利润回吐
    TRAILING_STOP_PCT = 0.20
    TRAILING_VOL_SPIKE_MULT = float(os.getenv("TRAILING_VOL_SPIKE_MULT", "1.20"))
    TRAILING_CONFIRM_SCORE_BULL = int(os.getenv("TRAILING_CONFIRM_SCORE_BULL", "2"))
    TRAILING_CONFIRM_SCORE_OTHER = int(os.getenv("TRAILING_CONFIRM_SCORE_OTHER", "1"))
    TRAILING_CONFIRM_BARS_BULL = int(os.getenv("TRAILING_CONFIRM_BARS_BULL", "2"))
    # 强趋势牛市用条件止盈（趋势转弱 + 波动放大 + 宽度转负）替代固定阈值
    BULL_REGIME_CONDITIONAL_TRAILING = str(os.getenv("BULL_REGIME_CONDITIONAL_TRAILING", "True")).lower() == "true"
    
    # 3. [新增] 波动率控制 (平滑曲线)
    # 目标年化波动率 50% (机构常用)
    # 当市场太疯时，降低仓位
    TARGET_VOLATILITY = 0.45
    # 空头仓位上限/下限：避免熊市里单边 -100% 过于激进
    SHORT_MIN_WEIGHT = 0.35
    SHORT_MAX_WEIGHT = 1.3

    # === Factor Mining 融合层 ===
    # 卫星仓位改为多因子截面评分（融合 mom/reversal/volume/funding 等）
    FACTOR_SIGNAL_SMOOTH_SPAN = int(os.getenv("FACTOR_SIGNAL_SMOOTH_SPAN", "8"))
    FACTOR_MIN_RANK = float(os.getenv("FACTOR_MIN_RANK", "0.70"))
    FACTOR_INV_VOL_POWER = float(os.getenv("FACTOR_INV_VOL_POWER", "0.5"))
    BULL_BETA_FLOOR = float(os.getenv("BULL_BETA_FLOOR", "0.85"))

    # 市场中性 alpha overlay（小权重）
    ENABLE_MARKET_NEUTRAL_OVERLAY = str(os.getenv("ENABLE_MARKET_NEUTRAL_OVERLAY", "True")).lower() == "true"
    OVERLAY_GROSS_EXPOSURE = float(os.getenv("OVERLAY_GROSS_EXPOSURE", "0.20"))
    OVERLAY_LONG_RANK = float(os.getenv("OVERLAY_LONG_RANK", "0.80"))
    OVERLAY_SHORT_RANK = float(os.getenv("OVERLAY_SHORT_RANK", "0.20"))
    OVERLAY_MIN_NAMES = int(os.getenv("OVERLAY_MIN_NAMES", "2"))
    OVERLAY_MAX_NAMES = int(os.getenv("OVERLAY_MAX_NAMES", "4"))
    OVERLAY_BETA_NEUTRAL = str(os.getenv("OVERLAY_BETA_NEUTRAL", "True")).lower() == "true"
    # 引擎A/B 按 regime 动态配比
    BLEND_BETA_SUPER_BULL = float(os.getenv("BLEND_BETA_SUPER_BULL", "1.00"))
    BLEND_BETA_BULL = float(os.getenv("BLEND_BETA_BULL", "0.92"))
    BLEND_BETA_NEUTRAL = float(os.getenv("BLEND_BETA_NEUTRAL", "0.35"))
    BLEND_BETA_BEAR = float(os.getenv("BLEND_BETA_BEAR", "0.15"))
    DISABLE_ALPHA_IN_SUPERBULL = str(os.getenv("DISABLE_ALPHA_IN_SUPERBULL", "True")).lower() == "true"
    DISABLE_ALPHA_IN_BULL = str(os.getenv("DISABLE_ALPHA_IN_BULL", "False")).lower() == "true"
    ALPHA_ENGINE_MAX_GROSS = float(os.getenv("ALPHA_ENGINE_MAX_GROSS", "0.50"))
    ALPHA_ENGINE_SCALE_SUPER_BULL = float(os.getenv("ALPHA_ENGINE_SCALE_SUPER_BULL", "0.00"))
    ALPHA_ENGINE_SCALE_BULL = float(os.getenv("ALPHA_ENGINE_SCALE_BULL", "0.03"))
    ALPHA_ENGINE_SCALE_NEUTRAL = float(os.getenv("ALPHA_ENGINE_SCALE_NEUTRAL", "1.15"))
    ALPHA_ENGINE_SCALE_BEAR = float(os.getenv("ALPHA_ENGINE_SCALE_BEAR", "1.35"))
    ALPHA_CARRY_WEIGHT = float(os.getenv("ALPHA_CARRY_WEIGHT", "0.40"))
    ALPHA_BASIS_TERM_WEIGHT = float(os.getenv("ALPHA_BASIS_TERM_WEIGHT", "0.20"))
    ALPHA_XS_WEIGHT = float(os.getenv("ALPHA_XS_WEIGHT", "0.60"))

    # === Regime 信号驱动的引擎配比（基于四状态） ===
    REGIME_BETA_RATIO_EXPLOSIVE = float(os.getenv("REGIME_BETA_RATIO_EXPLOSIVE", "1.00"))
    REGIME_BETA_RATIO_TREND = float(os.getenv("REGIME_BETA_RATIO_TREND", "1.00"))
    REGIME_BETA_RATIO_RANGE_BETA_MODE = float(os.getenv("REGIME_BETA_RATIO_RANGE_BETA_MODE", "0.90"))
    REGIME_BETA_RATIO_RANGE_ALPHA_MODE = float(os.getenv("REGIME_BETA_RATIO_RANGE_ALPHA_MODE", "0.30"))
    REGIME_BETA_RATIO_DEFENSIVE = float(os.getenv("REGIME_BETA_RATIO_DEFENSIVE", "0.10"))
    REGIME_BETA_CONFIDENCE_TILT = float(os.getenv("REGIME_BETA_CONFIDENCE_TILT", "0.15"))
    REGIME_BTC_FLOOR_EXPLOSIVE = float(os.getenv("REGIME_BTC_FLOOR_EXPLOSIVE", "1.10"))
    REGIME_BTC_FLOOR_TREND = float(os.getenv("REGIME_BTC_FLOOR_TREND", "0.95"))
    REGIME_NET_BETA_FLOOR_EXPLOSIVE = float(os.getenv("REGIME_NET_BETA_FLOOR_EXPLOSIVE", "1.15"))
    REGIME_NET_BETA_FLOOR_TREND = float(os.getenv("REGIME_NET_BETA_FLOOR_TREND", "1.10"))
    REGIME_NET_BETA_TARGET_EXPLOSIVE = float(os.getenv("REGIME_NET_BETA_TARGET_EXPLOSIVE", "1.28"))
    REGIME_NET_BETA_TARGET_TREND = float(os.getenv("REGIME_NET_BETA_TARGET_TREND", "1.20"))
    REGIME_BETA_NO_SHORT = str(os.getenv("REGIME_BETA_NO_SHORT", "True")).lower() == "true"

    # === 执行控制（来自 FactorMining 回测思想） ===
    EXECUTION_ALPHA = float(os.getenv("EXECUTION_ALPHA", "0.35"))
    REBALANCE_INTERVAL = int(os.getenv("REBALANCE_INTERVAL", "6"))
    TARGET_GROSS_EXPOSURE = float(os.getenv("TARGET_GROSS_EXPOSURE", "1.40"))
    # 交易摩擦约束
    MIN_REBALANCE_DELTA = float(os.getenv("MIN_REBALANCE_DELTA", "0.05"))
    MIN_HOLD_BARS = int(os.getenv("MIN_HOLD_BARS", "6"))
    MIN_REBALANCE_DELTA_SUPER_BULL = float(os.getenv("MIN_REBALANCE_DELTA_SUPER_BULL", "0.02"))
    MIN_REBALANCE_DELTA_BULL = float(os.getenv("MIN_REBALANCE_DELTA_BULL", "0.03"))
    MIN_REBALANCE_DELTA_NEUTRAL = float(os.getenv("MIN_REBALANCE_DELTA_NEUTRAL", "0.06"))
    MIN_REBALANCE_DELTA_BEAR = float(os.getenv("MIN_REBALANCE_DELTA_BEAR", "0.07"))
    MIN_HOLD_BARS_SUPER_BULL = int(os.getenv("MIN_HOLD_BARS_SUPER_BULL", "4"))
    MIN_HOLD_BARS_BULL = int(os.getenv("MIN_HOLD_BARS_BULL", "6"))
    MIN_HOLD_BARS_NEUTRAL = int(os.getenv("MIN_HOLD_BARS_NEUTRAL", "12"))
    MIN_HOLD_BARS_BEAR = int(os.getenv("MIN_HOLD_BARS_BEAR", "10"))
    SIGNAL_EDGE_BUFFER = float(os.getenv("SIGNAL_EDGE_BUFFER", "0.0008"))
    COST_BUFFER_MULTIPLIER = float(os.getenv("COST_BUFFER_MULTIPLIER", "1.2"))
    # 多目标优化约束（用于 walk-forward 搜索脚本）
    OPT_ALPHA24_FLOOR = float(os.getenv("OPT_ALPHA24_FLOOR", "-0.15"))
    OPT_WF_ALPHA_FLOOR = float(os.getenv("OPT_WF_ALPHA_FLOOR", "-0.20"))
    OPT_TARGET = os.getenv("OPT_TARGET", "alpha25")

    # === 实盘交易配置 ===
    # 双保险开关：只有 LIVE_ENABLED=True 且运行脚本使用 --execute 才会真实下单
    LIVE_ENABLED = str(os.getenv("LIVE_ENABLED", "False")).lower() == "true"
    LIVE_DRY_RUN = str(os.getenv("LIVE_DRY_RUN", "True")).lower() == "true"
    LIVE_ALLOW_SHORT = str(os.getenv("LIVE_ALLOW_SHORT", "True")).lower() == "true"

    # 执行层
    LIVE_MIN_ORDER_USDT = float(os.getenv("LIVE_MIN_ORDER_USDT", "25"))
    LIVE_CAPITAL_UTILIZATION = float(os.getenv("LIVE_CAPITAL_UTILIZATION", "0.95"))
    LIVE_MAX_GROSS_EXPOSURE = float(os.getenv("LIVE_MAX_GROSS_EXPOSURE", "1.0"))
    LIVE_LOOKBACK_BARS = int(os.getenv("LIVE_LOOKBACK_BARS", "1300"))
    LIVE_POSITION_SIDE = os.getenv("LIVE_POSITION_SIDE", "BOTH")

    # 交易标的（USDT 永续），格式示例：BTCUSDT,ETHUSDT,SOLUSDT
    LIVE_SYMBOLS_RAW = os.getenv("LIVE_SYMBOLS", "BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT")
    LIVE_SYMBOLS = [s.strip().upper() for s in LIVE_SYMBOLS_RAW.split(",") if s.strip()]

    LIVE_STATE_FILE = os.path.join(OUTPUT_PATH, "live_state.json")
