"""唐奇安通道突破 + ATR 移動停損（tw_quant/donchian_breakout_strategy.py）完整驗證。

流程（跟本次會話其餘策略一致）：
1. IS（2001-2020）用預設參數跑一次，看基本統計、逐年損益。
2. IS 子區間穩健性檢查（四個 5 年區間）。
3. IS 參數網格（entry_window x atr_stop_mult）檢查高原/懸崖。
4. 用 IS 選出的（預設）參數，對 OOS（2021-2023）跑「唯一一次」驗證，
   不因為結果不理想而回頭調參。

成本模型：期交稅（精確）＋手續費三檔（低中高）＋滑價（策略本身已經在
_fill_price 內模擬滑價，這裡的 slippage_points_round_trip 額外疊加
「非停損單本身滑價」以外的隱含成本則設為 0，避免重複計算——真正的
下單滑價已經反映在 entry_price/exit_price 裡了）。

用法：
    python scripts/backtest_donchian_breakout.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tw_quant.donchian_breakout_strategy import DonchianConfig, backtest  # noqa: E402
from tw_quant.hammer_signal_backtest import COST_SCENARIOS, TradeCost, apply_costs  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = REPO_ROOT / "data" / "txf_1min.parquet"
OUT_DIR = REPO_ROOT / "data"

IS_CUTOFF = "2021-01-01"
DEFAULT_CFG = DonchianConfig()  # entry=20, exit=10, atr_window=20, atr_stop_mult=2.0, slippage=1.0
SUB_PERIODS = [
    ("2001-2005", "2001-01-01", "2006-01-01"),
    ("2006-2010", "2006-01-01", "2011-01-01"),
    ("2011-2015", "2011-01-01", "2016-01-01"),
    ("2016-2020", "2016-01-01", "2021-01-01"),
]
ENTRY_WINDOWS = [15, 20, 25, 30]
ATR_STOP_MULTS = [1.5, 2.0, 2.5, 3.0]
ZERO_SLIP_COST = TradeCost("零滑價成本(僅稅+手續費)", commission_round_trip=60.0)


def summarize(trades: pd.DataFrame, cost: TradeCost, label: str) -> dict:
    if trades.empty:
        return dict(label=label, n=0)
    t = apply_costs(trades, cost)
    net = t["net_twd"].to_numpy()
    wins = (net > 0).sum()
    n = len(t)
    gross_win = t.loc[net > 0, "net_twd"].sum()
    gross_loss = -t.loc[net <= 0, "net_twd"].sum()
    pf = gross_win / gross_loss if gross_loss > 0 else np.inf
    tstat = (net.mean() / (net.std(ddof=1) / np.sqrt(n))) if n > 1 and net.std(ddof=1) > 0 else np.nan
    return dict(
        label=label, n=n, win_rate=wins / n, mean_net_twd=net.mean(), sum_net_twd=net.sum(),
        profit_factor=pf, t_stat=tstat,
    )


def main() -> None:
    df = pd.read_parquet(DATA_PATH)
    is_df = df[df["datetime"] < IS_CUTOFF]
    oos_df = df[df["datetime"] >= IS_CUTOFF]

    print("=" * 70)
    print("1) IS 預設參數整體結果（entry=20, exit=10, atr_window=20, atr_stop_mult=2.0, slippage=1pt）")
    print("=" * 70)
    is_trades = backtest(is_df, DEFAULT_CFG)
    print(f"IS 交易筆數: {len(is_trades)}")
    for cost in [ZERO_SLIP_COST, *COST_SCENARIOS]:
        row = summarize(is_trades, cost, cost.label)
        print(row)

    print(f"\n進場原因分布:\n{is_trades['entry_reason'].value_counts()}")
    print(f"出場原因分布:\n{is_trades['exit_reason'].value_counts()}")
    print(f"多空分布:\n{is_trades['direction'].value_counts()}")

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
    print("3) IS 參數網格（entry_window x atr_stop_mult），中檔成本，看高原/懸崖")
    print("=" * 70)
    grid_rows = []
    for ew in ENTRY_WINDOWS:
        for mult in ATR_STOP_MULTS:
            cfg = DonchianConfig(entry_window=ew, atr_stop_mult=mult)
            trades = backtest(is_df, cfg)
            row = summarize(trades, COST_SCENARIOS[1], f"ew={ew},mult={mult}")
            row["entry_window"] = ew
            row["atr_stop_mult"] = mult
            grid_rows.append(row)
    grid_df = pd.DataFrame(grid_rows)
    pd.set_option("display.width", 160)
    print(grid_df[["entry_window", "atr_stop_mult", "n", "win_rate", "mean_net_twd", "sum_net_twd", "profit_factor"]]
          .to_string(index=False))
    n_profitable = (grid_df["sum_net_twd"] > 0).sum()
    print(f"\n{n_profitable}/{len(grid_df)} 組參數組合中檔成本下為正淨損益")

    print("\n" + "=" * 70)
    print("4) OOS 唯一一次驗證（沿用 IS 預設參數，不回頭調參）")
    print("=" * 70)
    oos_trades = backtest(oos_df, DEFAULT_CFG)
    print(f"OOS 交易筆數: {len(oos_trades)}")
    for cost in [ZERO_SLIP_COST, *COST_SCENARIOS]:
        row = summarize(oos_trades, cost, cost.label)
        print(row)
    if not oos_trades.empty:
        print(f"\nOOS 逐筆明細:\n{oos_trades.to_string(index=False)}")

    is_trades.to_parquet(OUT_DIR / "donchian_is_trades.parquet", index=False)
    oos_trades.to_parquet(OUT_DIR / "donchian_oos_trades.parquet", index=False)
    grid_df.to_parquet(OUT_DIR / "donchian_param_grid.parquet", index=False)
    pd.DataFrame(sub_rows).to_parquet(OUT_DIR / "donchian_subperiods.parquet", index=False)
    print(f"\nsaved: donchian_is_trades.parquet / donchian_oos_trades.parquet / "
          f"donchian_param_grid.parquet / donchian_subperiods.parquet")


if __name__ == "__main__":
    main()
