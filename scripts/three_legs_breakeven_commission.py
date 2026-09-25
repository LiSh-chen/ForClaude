"""三腿策略可接受的手續費區間：淨損益對「來回手續費」是線性關係
（成本 = 稅金(固定，跟手續費無關) + 來回手續費常數 x 筆數），所以每個
區間/每一腳的損益兩平手續費可以直接解代數，不用網格掃描：

    breakeven_commission = (毛利點數換算TWD - 稅金總和) / 筆數
                          = 「零手續費情境淨損益」/ 筆數

用法：
    python scripts/three_legs_breakeven_commission.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tw_quant.hammer_signal_backtest import TradeCost, apply_costs  # noqa: E402
from tw_quant.lunch_reversal_strategy import backtest as backtest_lunch  # noqa: E402
from tw_quant.opening_rally_strategy import backtest as backtest_opening  # noqa: E402
from tw_quant.technical_indicators import daily_indicators, filter_trades_by_volume  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = REPO_ROOT / "data" / "txf_1min.parquet"

ZERO_COMM = TradeCost("零手續費(僅稅金)", commission_round_trip=0.0)
IS_CUTOFF = "2021-01-01"
PERIODS = [
    ("2001-2010", "2001-01-01", "2011-01-01"),
    ("2011-2015", "2011-01-01", "2016-01-01"),
    ("2016-2020", "2016-01-01", "2021-01-01"),
    ("2011-2023(近13年合計)", "2011-01-01", "2024-01-01"),
    ("全歷史 2001-2023", "2001-01-01", "2024-01-01"),
    ("OOS 2021-2023", "2021-01-01", "2024-01-01"),
]


def breakeven(trades: pd.DataFrame) -> tuple[int, float, float]:
    if trades.empty:
        return 0, float("nan"), float("nan")
    zero = apply_costs(trades, ZERO_COMM)["net_twd"]
    n = len(trades)
    return n, zero.sum(), zero.sum() / n


def three_legs_trades(data_df: pd.DataFrame, vol_threshold: float) -> dict[str, pd.DataFrame]:
    open_trades = backtest_opening(data_df)
    lunch_trades = backtest_lunch(data_df)
    short_leg = lunch_trades[lunch_trades["leg"] == "short_lunch_dip"]
    long_leg_raw = lunch_trades[lunch_trades["leg"] == "long_afternoon_rebound"]
    long_leg = filter_trades_by_volume(long_leg_raw, data_df, vol_threshold)
    return dict(opening=open_trades, short_lunch=short_leg, long_rebound=long_leg)


def main() -> None:
    df = pd.read_parquet(DATA_PATH)
    is_df = df[df["datetime"] < IS_CUTOFF]
    is_indicators = daily_indicators(is_df)
    vol_threshold = is_indicators["vol_ratio_lag1"].quantile(2 / 3)

    print("=" * 78)
    print("三腿合併：每個區間的損益兩平來回手續費（超過這個數字，該區間淨損益轉負）")
    print("=" * 78)
    print(f"{'區間':<24}{'筆數':>8}{'零手續費淨損益':>18}{'損益兩平手續費(元)':>20}")
    for name, start, end in PERIODS:
        sub_df = df[(df["datetime"] >= start) & (df["datetime"] < end)]
        legs = three_legs_trades(sub_df, vol_threshold)
        combined = pd.concat(legs.values(), ignore_index=True)
        n, zero_sum, be = breakeven(combined)
        print(f"{name:<24}{n:>8}{zero_sum:>18,.0f}{be:>20.1f}")

    print("\n" + "=" * 78)
    print("逐腳拆解（全歷史 2001-2023）")
    print("=" * 78)
    print(f"{'腳':<14}{'筆數':>8}{'零手續費淨損益':>18}{'損益兩平手續費(元)':>20}")
    legs = three_legs_trades(df, vol_threshold)
    for name, trades in legs.items():
        n, zero_sum, be = breakeven(trades)
        print(f"{name:<14}{n:>8}{zero_sum:>18,.0f}{be:>20.1f}")

    print("\n" + "=" * 78)
    print("逐腳拆解（OOS 2021-2023，唯一驗證期）")
    print("=" * 78)
    print(f"{'腳':<14}{'筆數':>8}{'零手續費淨損益':>18}{'損益兩平手續費(元)':>20}")
    oos_df = df[df["datetime"] >= IS_CUTOFF]
    oos_legs = three_legs_trades(oos_df, vol_threshold)
    for name, trades in oos_legs.items():
        n, zero_sum, be = breakeven(trades)
        print(f"{name:<14}{n:>8}{zero_sum:>18,.0f}{be:>20.1f}")

    print("\n" + "=" * 78)
    print("最保守的可接受手續費：取上面所有「應該要獲利」區間中最低的損益兩平點")
    print("（若實際手續費低於這個數字，所有已檢視的區間才會全部維持淨損益為正）")
    print("=" * 78)


if __name__ == "__main__":
    main()
