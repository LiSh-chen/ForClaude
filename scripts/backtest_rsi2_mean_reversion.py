"""RSI(2) 短期均值回歸策略（tw_quant/rsi2_mean_reversion_strategy.py）完整驗證。

跟 donchian 系列一樣的四步流程：IS 整體 -> IS 子區間穩健性 -> IS 參數網格
-> OOS 唯一一次驗證，成本模型用稅+手續費三檔（進出場價格已經是隔天開盤
+/-滑價的真實成交價，不重複疊加滑價成本）。

用法：
    python scripts/backtest_rsi2_mean_reversion.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tw_quant.hammer_signal_backtest import COST_SCENARIOS, TradeCost, apply_costs  # noqa: E402
from tw_quant.rsi2_mean_reversion_strategy import Rsi2Config, backtest  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = REPO_ROOT / "data" / "txf_1min.parquet"
OUT_DIR = REPO_ROOT / "data"

IS_CUTOFF = "2021-01-01"
DEFAULT_CFG = Rsi2Config()  # rsi_period=2, oversold=10, overbought=70, trend_ma_window=200, max_hold=10, slip=1
SUB_PERIODS = [
    ("2001-2005", "2001-01-01", "2006-01-01"),
    ("2006-2010", "2006-01-01", "2011-01-01"),
    ("2011-2015", "2011-01-01", "2016-01-01"),
    ("2016-2020", "2016-01-01", "2021-01-01"),
]
OVERSOLD_GRID = [5, 10, 15, 20]
MAX_HOLD_GRID = [5, 10, 15, 20]
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
    print("1) IS 預設參數整體結果（rsi_period=2, oversold=10, trend_ma=200, max_hold=10, slippage=1pt）")
    print("=" * 70)
    is_trades = backtest(is_df, DEFAULT_CFG)
    print(f"IS 交易筆數: {len(is_trades)}")
    for cost in [ZERO_SLIP_COST, *COST_SCENARIOS]:
        print(summarize(is_trades, cost, cost.label))
    if not is_trades.empty:
        print(f"\n多空分布:\n{is_trades['direction'].value_counts()}")
        print(f"出場原因分布:\n{is_trades['exit_reason'].value_counts()}")
        print(f"平均持有天數約: {(pd.to_datetime(is_trades['exit_date']) - pd.to_datetime(is_trades['entry_date'])).dt.days.mean():.1f}")

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
    print("3) IS 參數網格（oversold x max_hold_days），中檔成本")
    print("=" * 70)
    grid_rows = []
    for os_ in OVERSOLD_GRID:
        for mh in MAX_HOLD_GRID:
            cfg = Rsi2Config(oversold=os_, overbought=100 - os_, max_hold_days=mh)
            trades = backtest(is_df, cfg)
            row = summarize(trades, COST_SCENARIOS[1], f"oversold={os_},max_hold={mh}")
            row["oversold"] = os_
            row["max_hold_days"] = mh
            grid_rows.append(row)
    grid_df = pd.DataFrame(grid_rows)
    pd.set_option("display.width", 160)
    print(grid_df[["oversold", "max_hold_days", "n", "win_rate", "mean_net_twd", "sum_net_twd", "profit_factor"]]
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
    if not oos_trades.empty:
        print(f"\nOOS 逐筆明細:\n{oos_trades.to_string(index=False)}")

    is_trades.to_parquet(OUT_DIR / "rsi2_is_trades.parquet", index=False)
    oos_trades.to_parquet(OUT_DIR / "rsi2_oos_trades.parquet", index=False)
    grid_df.to_parquet(OUT_DIR / "rsi2_param_grid.parquet", index=False)
    pd.DataFrame(sub_rows).to_parquet(OUT_DIR / "rsi2_subperiods.parquet", index=False)
    print("\nsaved: rsi2_is_trades / rsi2_oos_trades / rsi2_param_grid / rsi2_subperiods .parquet")


if __name__ == "__main__":
    main()
