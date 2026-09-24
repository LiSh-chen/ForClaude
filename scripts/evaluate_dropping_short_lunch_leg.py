"""評估把「午盤放空」腳從實盤組合中剔除、只保留「開盤上衝」+「盤中翻多
（量能濾網）」兩腳，對整體策略的影響。

背景（見 revalidate_three_legs_after_data_fix.py 的逐年拆解）：午盤放空
這一腳原始點數 t 值不低（IS t=5.70），但 IS 20 年累積淨損益幾乎剛好打平
（-806元）——完全是靠 2001-2009 賺的錢撐住，2010 年之後幾乎年年小虧，
是「統計顯著但經濟上不顯著」的典型例子，且 2012-2020 這 9 年裡有 6 年
淨虧損，不符合本次會話一貫採用的子區間穩健性標準。

用法：
    python scripts/evaluate_dropping_short_lunch_leg.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tw_quant.hammer_signal_backtest import COST_SCENARIOS, TradeCost, apply_costs  # noqa: E402
from tw_quant.lunch_reversal_strategy import backtest as backtest_lunch  # noqa: E402
from tw_quant.opening_rally_strategy import backtest as backtest_opening  # noqa: E402
from tw_quant.technical_indicators import daily_indicators, filter_trades_by_volume  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = REPO_ROOT / "data" / "txf_1min.parquet"
OUT_PATH = REPO_ROOT / "data" / "two_legs_vs_three_legs_comparison.parquet"
IS_CUTOFF = "2021-01-01"
SUB_PERIODS = [
    ("2001-2005", "2001-01-01", "2006-01-01"),
    ("2006-2010", "2006-01-01", "2011-01-01"),
    ("2011-2015", "2011-01-01", "2016-01-01"),
    ("2016-2020", "2016-01-01", "2021-01-01"),
]


def tstat(points: pd.Series) -> float:
    n = len(points)
    if n < 2 or points.std(ddof=1) == 0:
        return np.nan
    return points.mean() / (points.std(ddof=1) / np.sqrt(n))


def two_legs(data_df: pd.DataFrame, vol_threshold: float) -> pd.DataFrame:
    open_t = backtest_opening(data_df)
    long_t = filter_trades_by_volume(
        backtest_lunch(data_df).pipe(lambda x: x[x["leg"] == "long_afternoon_rebound"]), data_df, vol_threshold,
    )
    return pd.concat([open_t, long_t], ignore_index=True)


def three_legs(data_df: pd.DataFrame, vol_threshold: float) -> pd.DataFrame:
    lunch_t = backtest_lunch(data_df)
    short_t = lunch_t[lunch_t["leg"] == "short_lunch_dip"]
    return pd.concat([two_legs(data_df, vol_threshold), short_t], ignore_index=True)


def main() -> None:
    df = pd.read_parquet(DATA_PATH)
    is_df = df[df["datetime"] < IS_CUTOFF]
    oos_df = df[df["datetime"] >= IS_CUTOFF]
    vol_threshold = daily_indicators(is_df)["vol_ratio_lag1"].quantile(2 / 3)

    rows = []
    for period_label, data_df in [("IS 2001-2020", is_df), ("OOS 2021-2023", oos_df)]:
        for combo_label, fn in [("三腿(含午盤放空)", three_legs), ("兩腿(去掉午盤放空)", two_legs)]:
            trades = fn(data_df, vol_threshold)
            row = dict(period=period_label, combo=combo_label, n=len(trades), t_stat=tstat(trades["pnl_points"]))
            for cost in COST_SCENARIOS:
                row[cost.label] = apply_costs(trades, cost)["net_twd"].sum()
            rows.append(row)
    summary = pd.DataFrame(rows)
    pd.set_option("display.width", 160)
    pd.set_option("display.float_format", lambda x: f"{x:,.0f}" if isinstance(x, float) else str(x))
    print("=== IS/OOS、三腿 vs 兩腿、三種成本情境 ===")
    print(summary.to_string(index=False))

    print("\n=== 兩腿版本 IS 子區間穩健性 ===")
    for name, start, end in SUB_PERIODS:
        sub_df = df[(df["datetime"] >= start) & (df["datetime"] < end)]
        trades = two_legs(sub_df, vol_threshold)
        mid_net = apply_costs(trades, COST_SCENARIOS[1])["net_twd"].sum()
        print(f"  {name}: n={len(trades)} t={tstat(trades['pnl_points']):.2f} 中檔成本淨損益={mid_net:,.0f}")

    print("\n=== 兩腿版本滑價敏感度（中檔手續費，OOS） ===")
    oos_two = two_legs(oos_df, vol_threshold)
    for slip in [0, 0.5, 1.0, 1.5, 2.0]:
        cost = TradeCost("中", commission_round_trip=60.0, slippage_points_round_trip=slip)
        net = apply_costs(oos_two, cost)["net_twd"].sum()
        print(f"  滑價{slip}點: OOS淨損益={net:,.0f}")

    summary.to_parquet(OUT_PATH, index=False)
    print(f"\nsaved: {OUT_PATH}")


if __name__ == "__main__":
    main()
