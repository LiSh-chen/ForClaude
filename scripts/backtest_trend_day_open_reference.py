"""趨勢日策略 reference="open" 版本驗證：用固定開盤價取代連續VWAP運算，
其餘參數完全沿用已經鎖定的版本（decision=11:00, min_move=20pt,
min_dominant_side_fraction=0.90），不重新調參，直接跑 IS/子區間/OOS/
滾動窗口，看這個「實務上更好執行」的簡化版本，統計證據會不會明顯輸給
VWAP版本。

用法：
    python scripts/backtest_trend_day_open_reference.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tw_quant.hammer_signal_backtest import COST_SCENARIOS, TradeCost, apply_costs  # noqa: E402
from tw_quant.trend_day_strategy import TrendDayConfig, backtest  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = REPO_ROOT / "data" / "txf_1min.parquet"
OUT_DIR = REPO_ROOT / "data"

IS_CUTOFF = "2021-01-01"
LOCKED_CFG = TrendDayConfig(min_dominant_side_fraction=0.90, reference="open")
SUB_PERIODS = [
    ("2001-2005", "2001-01-01", "2006-01-01"),
    ("2006-2010", "2006-01-01", "2011-01-01"),
    ("2011-2015", "2011-01-01", "2016-01-01"),
    ("2016-2020", "2016-01-01", "2021-01-01"),
]
WINDOWS = [
    ("2001-2003", "2001-01-01", "2004-01-01"),
    ("2004-2006", "2004-01-01", "2007-01-01"),
    ("2007-2009", "2007-01-01", "2010-01-01"),
    ("2010-2012", "2010-01-01", "2013-01-01"),
    ("2013-2015", "2013-01-01", "2016-01-01"),
    ("2016-2018", "2016-01-01", "2019-01-01"),
    ("2019-2021", "2019-01-01", "2022-01-01"),
    ("2021-2023(原OOS)", "2021-01-01", "2024-01-01"),
]
ZERO_SLIP_COST = TradeCost("零額外成本(僅稅+手續費)", commission_round_trip=60.0)


def tstat(points: pd.Series) -> float:
    n = len(points)
    if n < 2 or points.std(ddof=1) == 0:
        return np.nan
    return points.mean() / (points.std(ddof=1) / np.sqrt(n))


def summarize(trades: pd.DataFrame, cost: TradeCost, label: str) -> dict:
    if trades.empty:
        return dict(label=label, n=0)
    t = apply_costs(trades, cost)
    net = t["net_twd"].to_numpy()
    n = len(t)
    gross_win = t.loc[net > 0, "net_twd"].sum()
    gross_loss = -t.loc[net <= 0, "net_twd"].sum()
    pf = gross_win / gross_loss if gross_loss > 0 else np.inf
    return dict(
        label=label, n=n, win_rate=(net > 0).mean(), mean_net_twd=net.mean(), sum_net_twd=net.sum(),
        profit_factor=pf, t_stat=tstat(t["pnl_points"]),
    )


def main() -> None:
    df = pd.read_parquet(DATA_PATH)
    is_df = df[df["datetime"] < IS_CUTOFF]
    oos_df = df[df["datetime"] >= IS_CUTOFF]

    print("=" * 70)
    print("1) IS 整體（reference=open, 其餘參數沿用已鎖定版本）")
    print("=" * 70)
    is_trades = backtest(is_df, LOCKED_CFG)
    print(f"IS 交易筆數: {len(is_trades)}")
    for cost in [ZERO_SLIP_COST, *COST_SCENARIOS]:
        print(summarize(is_trades, cost, cost.label))
    if not is_trades.empty:
        print(f"出場原因分布:\n{is_trades['exit_reason'].value_counts()}")

    print("\n" + "=" * 70)
    print("2) IS 子區間穩健性")
    print("=" * 70)
    for name, start, end in SUB_PERIODS:
        sub_df = df[(df["datetime"] >= start) & (df["datetime"] < end)]
        sub_trades = backtest(sub_df, LOCKED_CFG)
        print(summarize(sub_trades, COST_SCENARIOS[1], name))

    print("\n" + "=" * 70)
    print("3) OOS 唯一一次驗證")
    print("=" * 70)
    oos_trades = backtest(oos_df, LOCKED_CFG)
    print(f"OOS 交易筆數: {len(oos_trades)}")
    for cost in [ZERO_SLIP_COST, *COST_SCENARIOS]:
        print(summarize(oos_trades, cost, cost.label))

    print("\n" + "=" * 70)
    print("4) 滾動窗口（同一組固定參數，8個約3年窗口）")
    print("=" * 70)
    window_rows = []
    for name, start, end in WINDOWS:
        sub_df = df[(df["datetime"] >= start) & (df["datetime"] < end)]
        trades = backtest(sub_df, LOCKED_CFG)
        row = summarize(trades, COST_SCENARIOS[1], name)
        window_rows.append(row)
        print(row)
    positive = sum(1 for r in window_rows if r.get("sum_net_twd", 0) > 0)
    print(f"\n{positive}/{len(window_rows)} 個窗口淨損益為正")

    all_trades = pd.concat([backtest(df[(df["datetime"] >= s) & (df["datetime"] < e)], LOCKED_CFG)
                             for _, s, e in WINDOWS], ignore_index=True)
    overall_t = tstat(all_trades["pnl_points"])
    overall_net = apply_costs(all_trades, COST_SCENARIOS[1])["net_twd"].sum()
    print(f"\n全部23年合併: n={len(all_trades)}, t={overall_t:.2f}, 中檔成本淨損益={overall_net:,.0f}")

    is_trades.to_parquet(OUT_DIR / "trend_day_open_ref_is_trades.parquet", index=False)
    oos_trades.to_parquet(OUT_DIR / "trend_day_open_ref_oos_trades.parquet", index=False)
    pd.DataFrame(window_rows).to_parquet(OUT_DIR / "trend_day_open_ref_windows.parquet", index=False)
    print("\nsaved: trend_day_open_ref_is_trades / trend_day_open_ref_oos_trades / trend_day_open_ref_windows .parquet")


if __name__ == "__main__":
    main()
