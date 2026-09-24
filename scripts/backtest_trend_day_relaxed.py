"""趨勢日偵測放寬版：掃 min_dominant_side_fraction，在IS上選一個「樣本數
有明顯增加、又沒有把t值犧牲太多」的門檻，鎖定後對OOS只驗證一次。

用法：
    python scripts/backtest_trend_day_relaxed.py
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
FRACTION_GRID = [1.0, 0.98, 0.95, 0.92, 0.90, 0.85, 0.80]
SUB_PERIODS = [
    ("2001-2005", "2001-01-01", "2006-01-01"),
    ("2006-2010", "2006-01-01", "2011-01-01"),
    ("2011-2015", "2011-01-01", "2016-01-01"),
    ("2016-2020", "2016-01-01", "2021-01-01"),
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
    print("1) IS：min_dominant_side_fraction 掃描（decision=11:00, min_move=20pt 固定）")
    print("=" * 70)
    rows = []
    for frac in FRACTION_GRID:
        cfg = TrendDayConfig(min_dominant_side_fraction=frac)
        trades = backtest(is_df, cfg)
        row = summarize(trades, COST_SCENARIOS[1], f"frac={frac}")
        row["fraction"] = frac
        rows.append(row)
        print(row)

    grid_df = pd.DataFrame(rows)

    print("\n" + "=" * 70)
    print("2) 選定門檻的子區間穩健性")
    print("=" * 70)
    SELECTED_FRACTION = 0.90
    print(f"選定 min_dominant_side_fraction = {SELECTED_FRACTION}"
          f"（樣本數比嚴格版本明顯增加、IS整體t值沒有明顯惡化）")
    selected_cfg = TrendDayConfig(min_dominant_side_fraction=SELECTED_FRACTION)
    sub_rows = []
    for name, start, end in SUB_PERIODS:
        sub_df = df[(df["datetime"] >= start) & (df["datetime"] < end)]
        sub_trades = backtest(sub_df, selected_cfg)
        row = summarize(sub_trades, COST_SCENARIOS[1], name)
        sub_rows.append(row)
        print(row)

    print("\n" + "=" * 70)
    print(f"3) OOS 唯一一次驗證（min_dominant_side_fraction={SELECTED_FRACTION}，不回頭調參）")
    print("=" * 70)
    oos_trades = backtest(oos_df, selected_cfg)
    print(f"OOS 交易筆數: {len(oos_trades)}")
    for cost in [ZERO_SLIP_COST, *COST_SCENARIOS]:
        print(summarize(oos_trades, cost, cost.label))
    if not oos_trades.empty:
        print(f"\nOOS 逐筆明細:\n{oos_trades.to_string(index=False)}")

    grid_df.to_parquet(OUT_DIR / "trend_day_relaxed_fraction_grid.parquet", index=False)
    pd.DataFrame(sub_rows).to_parquet(OUT_DIR / "trend_day_relaxed_subperiods.parquet", index=False)
    oos_trades.to_parquet(OUT_DIR / "trend_day_relaxed_oos_trades.parquet", index=False)
    print("\nsaved: trend_day_relaxed_fraction_grid / trend_day_relaxed_subperiods / trend_day_relaxed_oos_trades .parquet")


if __name__ == "__main__":
    main()
