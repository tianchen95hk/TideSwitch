import argparse
import json

from config.settings import Settings
from core_live.trader import LiveTrader


def parse_args():
    parser = argparse.ArgumentParser(description="4Alpha TideSwitch Live Trader")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="真实下单模式（默认仅 dry-run 模拟）",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # 双保险：必须同时满足 --execute + LIVE_ENABLED=True 才会真实下单
    dry_run = not args.execute
    if args.execute and not Settings.LIVE_ENABLED:
        raise RuntimeError("检测到 --execute，但 LIVE_ENABLED=False。请先在 .env 开启 LIVE_ENABLED=True。")

    if Settings.LIVE_DRY_RUN:
        dry_run = True

    trader = LiveTrader(dry_run=dry_run)
    result = trader.run_once()

    print("\n=== Run Summary ===")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
