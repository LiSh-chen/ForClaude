"""午盤效應策略（tw_quant/lunch_reversal_strategy.py）的完整 IS/OOS 驗證。

規則已經在樣本內（2001-2020）資料上定案，這裡只做兩件事：
1. 印出樣本內完整績效（含逐年、含停損 vs 不停損比較——不停損版本更好，見
   模組 docstring 的解釋）。
2. 用完全相同、沒有再調整過的設定，跑「只跑一次」的樣本外（2021-2023）驗證。

用法：
    python scripts/backtest_lunch_reversal.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tw_quant.hammer_signal_backtest import COST_SCENARIOS, apply_costs  # noqa: E402
from tw_quant.lunch_reversal_strategy import LunchReversalConfig, backtest  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = REPO_ROOT / "data" / "txf_1min.parquet"
IS_TRADES_OUT = REPO_ROOT / "data" / "lunch_reversal_is_trades.parquet"
OOS_TRADES_OUT = REPO_ROOT / "data" / "lunch_reversal_oos_trades.parquet"

IS_CUTOFF = "2021-01-01"


def report(trades: pd.DataFrame, label: str) -> None:
    print(f"\n=== {label}：n={len(trades)} ===")
    for leg in trades["leg"].unique():
        t = trades[trades["leg"] == leg]
        wins = t[t["pnl_points"] > 0]
        se = t["pnl_points"].std(ddof=1) / np.sqrt(len(t))
        tstat = t["pnl_points"].mean() / se if se > 0 else np.nan
        print(f"  {leg}: n={len(t)} win_rate={len(wins)/len(t):.3f} mean_pts={t['pnl_points'].mean():.3f} "
              f"tstat={tstat:.2f} gross_twd={t['pnl_twd'].sum():,.0f}")
    print(f"  合計毛損益: {trades['pnl_twd'].sum():,.0f}")
    for cost in COST_SCENARIOS:
        priced = apply_costs(trades, cost)
        print(f"    {cost.label}: net_total={priced['net_twd'].sum():,.0f}")


def main() -> None:
    df = pd.read_parquet(DATA_PATH)
    is_df = df[df["datetime"] < IS_CUTOFF]
    oos_df = df[df["datetime"] >= IS_CUTOFF]

    cfg = LunchReversalConfig()  # 定案設定：不設保護性停損

    is_trades = backtest(is_df, cfg)
    is_trades.to_parquet(IS_TRADES_OUT, index=False)
    report(is_trades, "樣本內 2001-2020")

    oos_trades = backtest(oos_df, cfg)
    oos_trades.to_parquet(OOS_TRADES_OUT, index=False)
    report(oos_trades, "樣本外 2021-2023（只跑一次，設定跟樣本內完全相同）")

    print("\n=== 樣本內 vs 樣本外，逐年毛損益 ===")
    all_trades = pd.concat([is_trades, oos_trades])
    all_trades["year"] = pd.to_datetime(all_trades["entry_dt"]).dt.year
    yearly = all_trades.groupby("year")["pnl_twd"].sum()
    pd.set_option("display.float_format", lambda x: f"{x:,.0f}")
    print(yearly.to_string())


if __name__ == "__main__":
    main()
