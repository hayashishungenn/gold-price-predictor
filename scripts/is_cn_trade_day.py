from __future__ import annotations

import argparse
from datetime import date

import pandas as pd


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check whether a given date is a China trading day.")
    parser.add_argument("--date", dest="target_date", type=str, default=date.today().isoformat(), help="Target date in YYYY-MM-DD format.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    target = pd.to_datetime(args.target_date).normalize()

    import akshare as ak

    calendar = ak.tool_trade_date_hist_sina()
    trade_dates = pd.to_datetime(calendar["trade_date"]).dt.normalize()
    is_trade_day = bool((trade_dates == target).any())
    print(f"is_trade_day={'true' if is_trade_day else 'false'}")
    print(f"target_date={target.date()}")


if __name__ == "__main__":
    main()
