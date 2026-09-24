"""探索台指期的日曆效應——完全不需要任何價格指標或連續運算，只需要知道
「今天是幾月幾號、星期幾、當月第幾個交易日、是不是結算日」，可以提前
排好行事曆，不用盯盤。這是回應「放棄需要動態水準的趨勢日策略家族，
找一個結構上就不依賴動態水準的全新方向」的請求。

檢查的日曆效應（全部用日盤 08:45開盤->13:45收盤 的報酬當基本單位）：
1. 星期效應：週一~週五分別的平均報酬/t值。
2. 當月交易日位置效應：每月第1~3個交易日（月初）、最後1~3個交易日
   （月底），常見的「作帳行情」候選。
3. 結算日效應：台指期結算日是每月第三個星期三（近似算法，沒有精確的
   台灣期交所假日行事曆，如果那個星期三剛好是假日會有誤差，這裡先用
   近似值探索方向，抓到候選後再用實際日期核對）。結算日前1天、結算日
   當天、結算日後1天，分別檢查。

用法：
    python scripts/scan_calendar_effects.py
"""

from __future__ import annotations

import sys
from datetime import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = REPO_ROOT / "data" / "txf_1min.parquet"
IS_CUTOFF = "2021-01-01"


def build_daily_session_returns(df: pd.DataFrame) -> pd.DataFrame:
    t = df["datetime"].dt.time
    day_df = df[(t >= time(8, 45)) & (t <= time(13, 45))].copy()
    day_df["trading_date"] = day_df["datetime"].dt.date

    daily = day_df.groupby("trading_date").agg(
        open=("open", "first"), close=("close", "last"), high=("high", "max"), low=("low", "min"),
    ).reset_index()
    daily["trading_date"] = pd.to_datetime(daily["trading_date"])
    daily = daily.sort_values("trading_date").reset_index(drop=True)
    daily["day_session_return_pts"] = daily["close"] - daily["open"]

    daily["day_of_week"] = daily["trading_date"].dt.day_name()
    daily["year_month"] = daily["trading_date"].dt.to_period("M")
    daily["trading_day_of_month"] = daily.groupby("year_month").cumcount() + 1
    daily["trading_days_in_month"] = daily.groupby("year_month")["trading_date"].transform("count")
    daily["trading_day_from_end"] = daily["trading_days_in_month"] - daily["trading_day_of_month"] + 1

    # 結算日近似：每月第三個星期三（沒有精確的期交所假日行事曆，先用這個近似）
    def third_wednesday(period: pd.Period) -> pd.Timestamp:
        month_start = period.start_time
        first_wed_offset = (2 - month_start.dayofweek) % 7  # 2=Wednesday
        first_wed = month_start + pd.Timedelta(days=first_wed_offset)
        return first_wed + pd.Timedelta(weeks=2)

    daily["settlement_date_approx"] = daily["year_month"].apply(third_wednesday)
    daily["days_to_settlement"] = (daily["settlement_date_approx"] - daily["trading_date"]).dt.days
    return daily


def tstat(x: pd.Series) -> float:
    n = len(x)
    if n < 2 or x.std(ddof=1) == 0:
        return np.nan
    return x.mean() / (x.std(ddof=1) / np.sqrt(n))


def report_group(daily: pd.DataFrame, group_col: str, label: str) -> None:
    print(f"\n=== {label} ===")
    g = daily.groupby(group_col)["day_session_return_pts"]
    for name, s in g:
        print(f"  {name}: n={len(s)} mean={s.mean():.2f}pt t={tstat(s):.2f}")


def main() -> None:
    df = pd.read_parquet(DATA_PATH)
    daily = build_daily_session_returns(df)
    is_daily = daily[daily["trading_date"] < IS_CUTOFF]

    print(f"IS 交易日數: {len(is_daily)}")

    report_group(is_daily, "day_of_week", "1) 星期效應（IS）")

    is_daily = is_daily.copy()
    is_daily["month_position"] = "中段"
    is_daily.loc[is_daily["trading_day_of_month"] <= 3, "month_position"] = "月初(前3個交易日)"
    is_daily.loc[is_daily["trading_day_from_end"] <= 3, "month_position"] = "月底(後3個交易日)"
    report_group(is_daily, "month_position", "2) 當月交易日位置效應（IS）")

    is_daily["settlement_position"] = "非結算相關"
    is_daily.loc[is_daily["days_to_settlement"] == 1, "settlement_position"] = "結算前1天"
    is_daily.loc[is_daily["days_to_settlement"] == 0, "settlement_position"] = "結算當天"
    is_daily.loc[is_daily["days_to_settlement"] == -1, "settlement_position"] = "結算後1天"
    report_group(is_daily, "settlement_position", "3) 結算日效應（IS，近似結算日）")

    is_daily.to_parquet(REPO_ROOT / "data" / "calendar_effects_daily_is.parquet", index=False)
    print(f"\nsaved: data/calendar_effects_daily_is.parquet")


if __name__ == "__main__":
    main()
