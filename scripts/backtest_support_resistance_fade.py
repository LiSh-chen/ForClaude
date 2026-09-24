"""壓力支撐區間逆勢操作（tw_quant/support_resistance_fade_strategy.py）完整驗證。

進場/停利用限價單模擬（沒有滑價，甚至跳空時還有機會用更好的價格成交），
只有停損（跌破支撐/漲破壓力）跟到期強制出場才會真正吃到滑價——這是
這次特別針對「滑價不可控」的設計，用法跟 donchian/rsi2 一樣走 IS整體
-> 子區間 -> 參數網格 -> OOS 單次驗證的四步流程。

用法：
    python scripts/backtest_support_resistance_fade.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tw_quant.hammer_signal_backtest import COST_SCENARIOS, TradeCost, apply_costs  # noqa: E402
from tw_quant.support_resistance_fade_strategy import SupportResistanceFadeConfig, backtest  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = REPO_ROOT / "data" / "txf_1min.parquet"
OUT_DIR = REPO_ROOT / "data"

IS_CUTOFF = "2021-01-01"
DEFAULT_CFG = SupportResistanceFadeConfig()
SUB_PERIODS = [
    ("2001-2005", "2001-01-01", "2006-01-01"),
    ("2006-2010", "2006-01-01", "2011-01-01"),
    ("2011-2015", "2011-01-01", "2016-01-01"),
    ("2016-2020", "2016-01-01", "2021-01-01"),
]
CHANNEL_GRID = [10, 15, 20, 25]
STOP_MULT_GRID = [0.5, 1.0, 1.5, 2.0]
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
    print("1) IS 預設參數整體結果（channel=20, min_range=3%, trend_ma=60/20天, "
          "slope門檻=3%, stop=1*ATR14, max_hold=20, slippage=1pt）")
    print("=" * 70)
    is_trades = backtest(is_df, DEFAULT_CFG)
    print(f"IS 交易筆數: {len(is_trades)}")
    for cost in [ZERO_SLIP_COST, *COST_SCENARIOS]:
        print(summarize(is_trades, cost, cost.label))
    if not is_trades.empty:
        print(f"\n多空分布:\n{is_trades['direction'].value_counts()}")
        print(f"出場原因分布:\n{is_trades['exit_reason'].value_counts()}")
        print(f"限價出場（target，無滑價）筆數: {(is_trades['exit_reason']=='target').sum()} / {len(is_trades)}")

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
    print("3) IS 參數網格（channel_window x stop_atr_mult），中檔成本")
    print("=" * 70)
    grid_rows = []
    for cw in CHANNEL_GRID:
        for sm in STOP_MULT_GRID:
            cfg = SupportResistanceFadeConfig(channel_window=cw, stop_atr_mult=sm)
            trades = backtest(is_df, cfg)
            row = summarize(trades, COST_SCENARIOS[1], f"cw={cw},stop={sm}")
            row["channel_window"] = cw
            row["stop_atr_mult"] = sm
            grid_rows.append(row)
    grid_df = pd.DataFrame(grid_rows)
    pd.set_option("display.width", 160)
    print(grid_df[["channel_window", "stop_atr_mult", "n", "win_rate", "mean_net_twd", "sum_net_twd", "profit_factor"]]
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

    is_trades.to_parquet(OUT_DIR / "sr_fade_is_trades.parquet", index=False)
    oos_trades.to_parquet(OUT_DIR / "sr_fade_oos_trades.parquet", index=False)
    grid_df.to_parquet(OUT_DIR / "sr_fade_param_grid.parquet", index=False)
    pd.DataFrame(sub_rows).to_parquet(OUT_DIR / "sr_fade_subperiods.parquet", index=False)
    print("\nsaved: sr_fade_is_trades / sr_fade_oos_trades / sr_fade_param_grid / sr_fade_subperiods .parquet")


if __name__ == "__main__":
    main()
