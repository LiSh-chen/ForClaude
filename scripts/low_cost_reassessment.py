"""用「高成本情境不切實際（現在都電子下單）」這個前提重新檢視兩個
關鍵候選——三腿策略、VWAP趨勢日——只看低成本(30元)/中成本(60元)情境，
拿掉高成本(100元，傳統營業員)這個過時假設，逐子區間、OOS重新攤開來看。

不重新調整策略參數，只是換一個更務實的成本假設重新檢視已經算好的
交易序列。

用法：
    python scripts/low_cost_reassessment.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tw_quant.hammer_signal_backtest import COST_SCENARIOS, apply_costs  # noqa: E402
from tw_quant.lunch_reversal_strategy import backtest as backtest_lunch  # noqa: E402
from tw_quant.opening_rally_strategy import backtest as backtest_opening  # noqa: E402
from tw_quant.technical_indicators import daily_indicators, filter_trades_by_volume  # noqa: E402
from tw_quant.trend_day_strategy import TrendDayConfig, backtest as backtest_trend_day  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = REPO_ROOT / "data" / "txf_1min.parquet"

LOW_COST, MID_COST, HIGH_COST = COST_SCENARIOS
IS_CUTOFF = "2021-01-01"
PERIODS = [
    ("2001-2010", "2001-01-01", "2011-01-01"),
    ("2011-2015", "2011-01-01", "2016-01-01"),
    ("2016-2020", "2016-01-01", "2021-01-01"),
    ("2011-2023(近13年合計)", "2011-01-01", "2024-01-01"),
    ("OOS 2021-2023", "2021-01-01", "2024-01-01"),
]


def tstat(x: pd.Series) -> float:
    n = len(x)
    if n < 2 or x.std(ddof=1) == 0:
        return np.nan
    return x.mean() / (x.std(ddof=1) / np.sqrt(n))


def report(trades: pd.DataFrame, label: str) -> None:
    if trades.empty:
        print(f"  {label}: n=0")
        return
    low = apply_costs(trades, LOW_COST)["net_twd"].sum()
    mid = apply_costs(trades, MID_COST)["net_twd"].sum()
    t = tstat(trades["pnl_points"])
    print(f"  {label}: n={len(trades)}, t={t:.2f}, 低成本淨損益={low:,.0f}, 中成本淨損益={mid:,.0f}")


def three_legs_trades(data_df: pd.DataFrame, vol_threshold: float) -> pd.DataFrame:
    open_trades = backtest_opening(data_df)
    lunch_trades = backtest_lunch(data_df)
    short_leg = lunch_trades[lunch_trades["leg"] == "short_lunch_dip"]
    long_leg_raw = lunch_trades[lunch_trades["leg"] == "long_afternoon_rebound"]
    long_leg = filter_trades_by_volume(long_leg_raw, data_df, vol_threshold)
    return pd.concat([open_trades, short_leg, long_leg], ignore_index=True)


def main() -> None:
    df = pd.read_parquet(DATA_PATH)
    is_df = df[df["datetime"] < IS_CUTOFF]
    is_indicators = daily_indicators(is_df)
    vol_threshold = is_indicators["vol_ratio_lag1"].quantile(2 / 3)

    print("=" * 70)
    print("1) 三腿策略合併：低成本/中成本情境，逐區間對照")
    print("=" * 70)
    for name, start, end in PERIODS:
        sub_df = df[(df["datetime"] >= start) & (df["datetime"] < end)]
        trades = three_legs_trades(sub_df, vol_threshold)
        report(trades, name)

    print("\n" + "=" * 70)
    print("2) VWAP趨勢日(frac=0.90)：低成本/中成本情境，逐區間對照")
    print("=" * 70)
    cfg = TrendDayConfig(min_dominant_side_fraction=0.90)
    for name, start, end in PERIODS:
        sub_df = df[(df["datetime"] >= start) & (df["datetime"] < end)]
        trades = backtest_trend_day(sub_df, cfg)
        report(trades, name)


if __name__ == "__main__":
    main()
