"""產生三腳策略（開盤上衝／午盤放空／盤中翻多）K線進出場視覺化用的資料
（data/three_leg_candlestick.json）：最近 N 個交易日的日盤 1 分鐘K棒，加上
每天各腳的進場/出場時間、價格、損益。

用法：
    python scripts/build_three_leg_candlestick_data.py --days 60
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tw_quant.lunch_reversal_strategy import backtest as backtest_lunch  # noqa: E402
from tw_quant.opening_rally_strategy import backtest as backtest_opening  # noqa: E402
from tw_quant.technical_indicators import filter_trades_by_volume  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = REPO_ROOT / "data" / "txf_1min.parquet"
OUT_PATH = REPO_ROOT / "data" / "three_leg_candlestick.json"

VOL_THRESHOLD = 1.1  # 樣本內(2001-2020)算出的固定門檻，見 hammer/lunch reversal 研究


def build(days: int) -> dict:
    full_df = pd.read_parquet(DATA_PATH)

    t = full_df["datetime"].dt.time
    day_df = full_df[(t >= time(8, 45)) & (t <= time(13, 45))].copy()
    day_df["date"] = day_df["datetime"].dt.date
    all_dates = sorted(day_df["date"].unique())
    selected_dates = set(all_dates[-days:])

    sel_full = full_df[full_df["datetime"].dt.date.isin(selected_dates)]
    sel_day_df = day_df[day_df["date"].isin(selected_dates)].copy()
    sel_day_df["date_str"] = sel_day_df["date"].astype(str)
    sel_day_df["minute"] = sel_day_df["datetime"].dt.strftime("%H:%M")

    open_trades = backtest_opening(sel_full)
    lunch_trades = backtest_lunch(sel_full)
    short_leg = lunch_trades[lunch_trades["leg"] == "short_lunch_dip"]
    long_leg = lunch_trades[lunch_trades["leg"] == "long_afternoon_rebound"]
    long_leg_filtered = filter_trades_by_volume(long_leg, full_df, VOL_THRESHOLD)

    leg_specs = [
        ("開盤上衝", "long", open_trades, "leg1"),
        ("午盤放空", "short", short_leg, "leg2"),
        ("盤中翻多", "long", long_leg_filtered, "leg3"),
    ]

    days_payload: dict[str, dict] = {}
    for d, g in sel_day_df.groupby("date_str"):
        g = g.sort_values("datetime")
        candles = [[row["minute"], round(row["open"], 1), round(row["high"], 1),
                    round(row["low"], 1), round(row["close"], 1)] for _, row in g.iterrows()]

        legs = []
        for leg_name, direction, trades_df, color_key in leg_specs:
            row = trades_df[trades_df["trading_date"].astype(str) == d]
            if row.empty:
                continue
            r = row.iloc[0]
            legs.append(dict(
                name=leg_name, direction=direction, color=color_key,
                entry_time=pd.Timestamp(r["entry_dt"]).strftime("%H:%M"),
                entry_price=float(r["entry_price"]),
                exit_time=pd.Timestamp(r["exit_dt"]).strftime("%H:%M"),
                exit_price=float(r["exit_price"]),
                pnl_points=float(r["pnl_points"]),
                pnl_twd=float(r["pnl_twd"]),
            ))
        days_payload[d] = dict(candles=candles, legs=legs)

    return {"dates": sorted(days_payload.keys()), "days": days_payload}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=60)
    args = parser.parse_args()

    payload = build(args.days)
    OUT_PATH.write_text(json.dumps(payload))
    print(f"saved: {OUT_PATH} ({OUT_PATH.stat().st_size / 1e6:.2f} MB), {len(payload['dates'])} 個交易日")


if __name__ == "__main__":
    main()
