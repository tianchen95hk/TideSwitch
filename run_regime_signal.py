import argparse
import os

import pandas as pd

from config.settings import Settings
from core_strategy.regime_signaler import RegimeSignaler


def _consistency_report(signal_df):
    out = signal_df.copy()
    close = out["close"]
    out["fwd_7d"] = close.shift(-42) / close - 1.0
    out["fwd_14d"] = close.shift(-84) / close - 1.0
    out["fwd_30d"] = close.shift(-180) / close - 1.0

    rows = []
    for state, d in out.groupby("state"):
        if d.empty:
            continue
        rows.append(
            {
                "state": state,
                "bars": int(len(d)),
                "mean_fwd_7d": float(d["fwd_7d"].mean()),
                "mean_fwd_14d": float(d["fwd_14d"].mean()),
                "mean_fwd_30d": float(d["fwd_30d"].mean()),
                "p_fwd_7d_pos": float((d["fwd_7d"] > 0).mean()),
                "p_fwd_14d_pos": float((d["fwd_14d"] > 0).mean()),
                "p_fwd_30d_pos": float((d["fwd_30d"] > 0).mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("state")


def main():
    parser = argparse.ArgumentParser(description="Build 4-state BTC regime signal.")
    parser.add_argument("--start-date", default="2023-01-01")
    parser.add_argument("--end-date", default="2026-03-16")
    parser.add_argument("--input", default=Settings.FACTOR_FILE)
    parser.add_argument("--output-prefix", default="regime_signal_v3")
    args = parser.parse_args()

    df = pd.read_parquet(args.input).sort_index()
    signaler = RegimeSignaler()
    sig = signaler.build(df, start_date=args.start_date, end_date=args.end_date)
    annual = signaler.annual_summary(sig)
    consistency = _consistency_report(sig)

    os.makedirs(Settings.OUTPUT_PATH, exist_ok=True)
    s_tag = args.start_date.replace("-", "")
    e_tag = args.end_date.replace("-", "")
    main_path = os.path.join(Settings.OUTPUT_PATH, f"{args.output_prefix}_{s_tag}_{e_tag}.csv")
    annual_path = os.path.join(
        Settings.OUTPUT_PATH, f"{args.output_prefix}_annual_summary_{s_tag}_{e_tag}.csv"
    )
    cons_path = os.path.join(
        Settings.OUTPUT_PATH, f"{args.output_prefix}_consistency_{s_tag}_{e_tag}.csv"
    )

    sig.to_csv(main_path)
    annual.to_csv(annual_path, index=False)
    consistency.to_csv(cons_path, index=False)

    last_ts = sig.index.max()
    last = sig.loc[last_ts]
    print("saved:", main_path)
    print("saved:", annual_path)
    print("saved:", cons_path)
    print("last_ts:", last_ts)
    print("last_state:", last["state"])
    print("last_mode:", last["mode"])
    print("last_regime_score:", round(float(last["regime_score"]), 4))
    print("last_target_beta_hint:", round(float(last["target_beta_hint"]), 4))
    print("last_confidence:", round(float(last["confidence"]), 4))


if __name__ == "__main__":
    main()
