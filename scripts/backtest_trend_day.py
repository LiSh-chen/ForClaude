"""趨勢日偵測（tw_quant/trend_day_strategy.py）完整驗證。

用法：
    python scripts/backtest_trend_day.py
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
DEFAULT_CFG = TrendDayConfig()  # decision=11:00, min_move=20pt, session_end=13:25, slippage=1pt
SUB_PERIODS = [
    ("2001-2005", "2001-01-01", "2006-01-01"),
    ("2006-2010", "2006-01-01", "2011-01-01"),
    ("2011-2015", "2011-01-01", "2016-01-01"),
    ("2016-2020", "2016-01-01", "2021-01-01"),
]
DECISION_GRID = ["10:00", "10:30", "11:00", "11:30"]
MIN_MOVE_GRID = [10.0, 20.0, 30.0, 40.0]
ZERO_SLIP_COST = TradeCost("零額外成本(僅稅+手續費)", commission_round_trip=60.0)


def _time(s: str):
    from datetime import time
    h, m = s.split(":")
    return time(int(h), int(m))


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
    print("1) IS 預設參數整體結果（decision=11:00, min_move=20pt, session_end=13:25, slippage=1pt）")
    print("=" * 70)
    is_trades = backtest(is_df, DEFAULT_CFG)
    print(f"IS 交易筆數: {len(is_trades)}")
    for cost in [ZERO_SLIP_COST, *COST_SCENARIOS]:
        print(summarize(is_trades, cost, cost.label))
    if not is_trades.empty:
        print(f"\n多空分布:\n{is_trades['direction'].value_counts()}")
        print(f"出場原因分布:\n{is_trades['exit_reason'].value_counts()}")
        print(f"觸發率: {len(is_trades)} / {is_df['datetime'].dt.date.nunique()} 個IS交易日")

    print("\n" + "=" * 70)
    print("2) IS 子區間穩健性檢查")
    print("=" * 70)
    sub_rows = []
    for name, start, end in SUB_PERIODS:
        sub_df = df[(df["datetime"] >= start) & (df["datetime"] < end)]
        sub_trades = backtest(sub_df, DEFAULT_CFG)
        row = summarize(sub_trades, COST_SCENARIOS[1], name)
        sub_rows.append(row)
        print(row)

    print("\n" + "=" * 70)
    print("3) IS 參數網格（決策時點 x 最小幅度門檻），中檔成本")
    print("=" * 70)
    grid_rows = []
    for dt_str in DECISION_GRID:
        for mm in MIN_MOVE_GRID:
            cfg = TrendDayConfig(decision_time=_time(dt_str), min_move_points=mm)
            trades = backtest(is_df, cfg)
            row = summarize(trades, COST_SCENARIOS[1], f"decision={dt_str},min_move={mm}")
            row["decision_time"] = dt_str
            row["min_move_points"] = mm
            grid_rows.append(row)
    grid_df = pd.DataFrame(grid_rows)
    pd.set_option("display.width", 160)
    print(grid_df[["decision_time", "min_move_points", "n", "win_rate", "mean_net_twd", "sum_net_twd", "profit_factor"]]
          .to_string(index=False))
    n_profitable = (grid_df["sum_net_twd"] > 0).sum()
    print(f"\n{n_profitable}/{len(grid_df)} 組參數組合中檔成本下為正淨損益")

    print("\n" + "=" * 70)
    print("4) OOS 唯一一次驗證（沿用 IS 預設參數，不回頭調參）")
    print("=" * 70)
    oos_trades = backtest(oos_df, DEFAULT_CFG)
    print(f"OOS 交易筆數: {len(oos_trades)}")
    for cost in [ZERO_SLIP_COST, *COST_SCENARIOS]:
        print(summarize(oos_trades, cost, cost.label))

    is_trades.to_parquet(OUT_DIR / "trend_day_is_trades.parquet", index=False)
    oos_trades.to_parquet(OUT_DIR / "trend_day_oos_trades.parquet", index=False)
    grid_df.to_parquet(OUT_DIR / "trend_day_param_grid.parquet", index=False)
    pd.DataFrame(sub_rows).to_parquet(OUT_DIR / "trend_day_subperiods.parquet", index=False)
    print("\nsaved: trend_day_is_trades / trend_day_oos_trades / trend_day_param_grid / trend_day_subperiods .parquet")


if __name__ == "__main__":
    main()
