import contextlib
import os
import random
from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import pandas as pd

from config.settings import Settings
from core_backtest.engine import BacktestEngine
from core_strategy.algo import WhiteboxAlgo


SEARCH_SPACE = {
    "MA_WINDOW": [780, 930, 1100],
    "CORE_BTC_WEIGHT": [0.85, 0.90, 0.95],
    "BULL_BETA_FLOOR": [0.85, 0.95, 1.00],
    "TRAILING_STOP_PCT": [0.20, 0.25, 0.30],
    "TARGET_GROSS_EXPOSURE": [1.00, 1.05, 1.10, 1.20],
    "FACTOR_MIN_RANK": [0.70, 0.80, 0.85],
    "TOP_N": [1, 2],
    "TARGET_VOLATILITY": [0.40, 0.45, 0.55],
    "REGIME_HYSTERESIS_PCT": [0.001, 0.003, 0.005],
    "EXECUTION_ALPHA": [0.20, 0.35, 0.50],
    "REBALANCE_INTERVAL": [6, 12],
    "FACTOR_INV_VOL_POWER": [0.5, 1.0],
    "ENABLE_NEUTRAL_SHORT": [False, True],
    "BLEND_BETA_BULL": [0.80, 0.90, 1.00],
    "BLEND_BETA_NEUTRAL": [0.35, 0.45, 0.55],
    "BLEND_BETA_BEAR": [0.15, 0.25, 0.35],
    "BLEND_BETA_SUPER_BULL": [0.95, 1.00],
    "TARGET_BETA_MIN": [0.20, 0.30, 0.40],
    "TARGET_BETA_MAX": [1.10, 1.20, 1.30],
    "SUPERBULL_BETA_MIN": [1.10, 1.20, 1.30],
    "SUPERBULL_BTC_FLOOR": [1.00, 1.10, 1.20],
    "ALPHA_ENGINE_MAX_GROSS": [0.25, 0.35, 0.45],
    "DISABLE_ALPHA_IN_SUPERBULL": [True, False],
    "DISABLE_ALPHA_IN_BULL": [False, True],
    "MIN_REBALANCE_DELTA": [0.03, 0.05, 0.07],
    "MIN_HOLD_BARS": [6, 12],
    "SIGNAL_EDGE_BUFFER": [0.0006, 0.0008, 0.0012],
    "COST_BUFFER_MULTIPLIER": [1.1, 1.2, 1.4],
}

FIXED_PARAMS = {
    "ENABLE_MARKET_NEUTRAL_OVERLAY": True,
    "OVERLAY_GROSS_EXPOSURE": 0.20,
}

EVAL_WINDOWS = {
    "fy24": ("2024-01-01", "2024-12-31"),
    "fy25": ("2025-01-01", "2025-12-31"),
}

# train -> validate 的 walk-forward 校验窗口（用 validate 做目标统计）
WALK_FORWARD_VALIDATE_WINDOWS = [
    ("2024-07-01", "2024-12-31"),
    ("2025-01-01", "2025-06-30"),
    ("2025-07-01", "2025-12-31"),
]


@dataclass
class Constraints:
    max_dd: float = -0.35
    max_annual_turnover: float = 40.0
    max_trade_count: int = 140
    alpha24_floor: float = Settings.OPT_ALPHA24_FLOOR
    wf_alpha_floor: float = Settings.OPT_WF_ALPHA_FLOOR


def _nearest_choice(value, choices):
    if value in choices:
        return value
    arr = np.array(choices, dtype=float)
    idx = int(np.argmin(np.abs(arr - float(value))))
    return choices[idx]


def _candidate_from_settings():
    out = {}
    for key, choices in SEARCH_SPACE.items():
        out[key] = _nearest_choice(getattr(Settings, key), choices)
    return out


def _apply_params(params):
    for k, v in params.items():
        setattr(Settings, k, v)


def _run_window(df, start_date, end_date):
    old_start, old_end = Settings.BACKTEST_START_DATE, Settings.BACKTEST_END_DATE
    Settings.BACKTEST_START_DATE = start_date
    Settings.BACKTEST_END_DATE = end_date
    try:
        with open(os.devnull, "w", encoding="utf-8") as devnull:
            with contextlib.redirect_stdout(devnull):
                engine = BacktestEngine(df, WhiteboxAlgo())
                res = engine.run()
        if res.empty:
            return {"ret": np.nan, "btc": np.nan, "alpha": np.nan, "max_dd": np.nan, "turnover": np.nan, "trades": 0}

        ret = float(res["equity"].iloc[-1] / Settings.INITIAL_CAPITAL - 1.0)
        btc = float(res["btc_close"].iloc[-1] / res["btc_close"].iloc[0] - 1.0)
        alpha = ret - btc
        max_dd = float((res["equity"] / res["equity"].cummax() - 1.0).min())
        turnover = float(getattr(engine, "total_turnover", np.nan))
        trades = int(len(getattr(engine, "trade_signals", [])))
        return {"ret": ret, "btc": btc, "alpha": alpha, "max_dd": max_dd, "turnover": turnover, "trades": trades}
    finally:
        Settings.BACKTEST_START_DATE, Settings.BACKTEST_END_DATE = old_start, old_end


def _mutate(parent, rng, intensity=0.25):
    child = deepcopy(parent)
    keys = list(SEARCH_SPACE.keys())
    mutate_n = max(1, int(len(keys) * intensity))
    chosen = rng.sample(keys, k=mutate_n)
    for key in chosen:
        vals = SEARCH_SPACE[key]
        cur = child[key]
        alt = [x for x in vals if x != cur]
        if alt:
            child[key] = rng.choice(alt)
    return child


def evaluate_candidate(df, candidate, constraints):
    original = {k: getattr(Settings, k) for k in set(SEARCH_SPACE) | set(FIXED_PARAMS)}
    try:
        _apply_params(FIXED_PARAMS)
        _apply_params(candidate)

        fy24 = _run_window(df, *EVAL_WINDOWS["fy24"])
        fy25 = _run_window(df, *EVAL_WINDOWS["fy25"])

        wf_alphas = []
        wf_max_dd = []
        wf_turnover = []
        wf_trades = []
        for ws, we in WALK_FORWARD_VALIDATE_WINDOWS:
            m = _run_window(df, ws, we)
            wf_alphas.append(float(m["alpha"]))
            wf_max_dd.append(float(m["max_dd"]))
            wf_turnover.append(float(m["turnover"]))
            wf_trades.append(int(m["trades"]))

        alpha24 = float(fy24["alpha"])
        alpha25 = float(fy25["alpha"])
        wf_min_alpha = float(np.nanmin(wf_alphas)) if wf_alphas else np.nan
        objective = float(alpha25 if Settings.OPT_TARGET == "alpha25" else min(alpha24, alpha25))

        max_dd = float(np.nanmin([fy24["max_dd"], fy25["max_dd"]] + wf_max_dd))
        annual_turnover = float(np.nanmax([fy24["turnover"], fy25["turnover"]] + wf_turnover))
        trade_count = int(max(fy24["trades"], fy25["trades"], *wf_trades))

        feasible = (
            np.isfinite(objective)
            and (alpha24 >= constraints.alpha24_floor)
            and (wf_min_alpha >= constraints.wf_alpha_floor)
            and (max_dd >= constraints.max_dd)
            and (annual_turnover <= constraints.max_annual_turnover)
            and (trade_count <= constraints.max_trade_count)
        )
        if not feasible:
            objective -= 10.0

        row = dict(candidate)
        row.update(
            {
                "ret24": float(fy24["ret"]),
                "btc24": float(fy24["btc"]),
                "alpha24": alpha24,
                "max_dd24": float(fy24["max_dd"]),
                "turnover24": float(fy24["turnover"]),
                "trades24": int(fy24["trades"]),
                "ret25": float(fy25["ret"]),
                "btc25": float(fy25["btc"]),
                "alpha25": alpha25,
                "max_dd25": float(fy25["max_dd"]),
                "turnover25": float(fy25["turnover"]),
                "trades25": int(fy25["trades"]),
                "wf_min_alpha": wf_min_alpha,
                "objective": objective,
                "feasible": bool(feasible),
                "constraint_max_dd": constraints.max_dd,
                "constraint_max_annual_turnover": constraints.max_annual_turnover,
                "constraint_max_trade_count": constraints.max_trade_count,
                "constraint_alpha24_floor": constraints.alpha24_floor,
                "constraint_wf_alpha_floor": constraints.wf_alpha_floor,
            }
        )
        return row
    finally:
        _apply_params(original)


def adaptive_search(df, constraints, seed=20260317, rounds=4, batch_size=10, elite_size=4):
    rng = random.Random(seed)
    seen = {}
    baseline = _candidate_from_settings()
    elites = [baseline]

    def _key(c):
        return tuple((k, c[k]) for k in sorted(c.keys()))

    all_rows = []
    for r in range(rounds):
        pool = []
        if r == 0:
            pool.append(deepcopy(baseline))
            for _ in range(batch_size - 1):
                c = deepcopy(baseline)
                c = _mutate(c, rng, intensity=0.35)
                pool.append(c)
        else:
            for _ in range(batch_size):
                parent = rng.choice(elites)
                c = _mutate(parent, rng, intensity=max(0.10, 0.35 - 0.05 * r))
                pool.append(c)

        round_rows = []
        for cand in pool:
            k = _key(cand)
            if k in seen:
                round_rows.append(seen[k])
                continue
            row = evaluate_candidate(df=df, candidate=cand, constraints=constraints)
            seen[k] = row
            round_rows.append(row)
            all_rows.append(row)

        ranked = sorted(round_rows, key=lambda x: (x["objective"], x["alpha24"]), reverse=True)
        elites = [dict((k, v) for k, v in rr.items() if k in SEARCH_SPACE) for rr in ranked[:elite_size]]

    out = pd.DataFrame(all_rows).drop_duplicates()
    out = out.sort_values(["objective", "alpha24", "alpha25"], ascending=False)
    return out


def main():
    os.makedirs(Settings.OUTPUT_PATH, exist_ok=True)
    df = pd.read_parquet(Settings.FACTOR_FILE)
    constraints = Constraints()
    res = adaptive_search(df=df, constraints=constraints)

    out_all = os.path.join(Settings.OUTPUT_PATH, "moo_walkforward_all.csv")
    out_top = os.path.join(Settings.OUTPUT_PATH, "moo_walkforward_top20.csv")
    out_both = os.path.join(Settings.OUTPUT_PATH, "moo_walkforward_both_positive.csv")
    out_feasible = os.path.join(Settings.OUTPUT_PATH, "moo_walkforward_feasible.csv")

    res.to_csv(out_all, index=False)
    res.head(20).to_csv(out_top, index=False)
    res[(res["alpha24"] > 0) & (res["alpha25"] > 0)].to_csv(out_both, index=False)
    res[res["feasible"]].to_csv(out_feasible, index=False)

    print(f"saved: {out_all}")
    print(f"saved: {out_top}")
    print(f"saved: {out_both}")
    print(f"saved: {out_feasible}")


if __name__ == "__main__":
    main()
