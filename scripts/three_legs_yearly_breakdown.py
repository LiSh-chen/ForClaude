"""三腿策略逐年損益拆解（IS 2001-2020），用來檢查「樣本內子區間穩健性」
背後是不是有更細的衰減型態（子區間 5 年一格可能蓋掉逐年的趨勢）。

跟 revalidate_three_legs_after_data_fix.py 用同一組門檻/邏輯，只是輸出
粒度從 5 年一格改成逐年，方便看出哪幾年在賺、哪幾年在虧。

用法：
    python scripts/three_legs_yearly_breakdown.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tw_quant.hammer_signal_backtest import COST_SCENARIOS, apply_costs  # noqa: E402
from tw_quant.lunch_reversal_strategy import backtest as backtest_lunch  # noqa: E402
from tw_quant.opening_rally_strategy import backtest as backtest_opening  # noqa: E402
from tw_quant.technical_indicators import daily_indicators, filter_trades_by_volume  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = REPO_ROOT / "data" / "txf_1min.parquet"
OUT_PATH = REPO_ROOT / "data" / "three_legs_yearly_breakdown.parquet"
IS_CUTOFF = "2021-01-01"


def yearly(trades: pd.DataFrame, leg: str) -> pd.DataFrame:
    t = trades.copy()
    t["year"] = pd.to_datetime(t["trading_date"]).dt.year
    costed = apply_costs(t, COST_SCENARIOS[1])
    t["net_twd"] = costed["net_twd"]
    g = t.groupby("year").agg(n=("pnl_points", "size"), net_twd=("net_twd", "sum")).reset_index()
    g["leg"] = leg
    return g


def main() -> None:
    df = pd.read_parquet(DATA_PATH)
    is_df = df[df["datetime"] < IS_CUTOFF]
    vol_threshold = daily_indicators(is_df)["vol_ratio_lag1"].quantile(2 / 3)

    open_trades = backtest_opening(is_df)
    lunch_trades = backtest_lunch(is_df)
    short_leg = lunch_trades[lunch_trades["leg"] == "short_lunch_dip"]
    long_leg = filter_trades_by_volume(lunch_trades[lunch_trades["leg"] == "long_afternoon_rebound"], is_df, vol_threshold)

    parts = [yearly(open_trades, "opening"), yearly(short_leg, "short_lunch"), yearly(long_leg, "long_rebound")]
    all_years = pd.concat(parts, ignore_index=True)
    pivot = all_years.pivot(index="year", columns="leg", values="net_twd").fillna(0)
    pivot["combined"] = pivot.sum(axis=1)
    pivot["combined_cum"] = pivot["combined"].cumsum()

    pd.set_option("display.width", 150)
    pd.set_option("display.float_format", lambda x: f"{x:,.0f}")
    print(pivot.to_string())

    all_years.to_parquet(OUT_PATH, index=False)
    print(f"\nsaved: {OUT_PATH}")


if __name__ == "__main__":
    main()
