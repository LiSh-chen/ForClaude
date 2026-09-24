"""「大做小」策略回測：較大週期 MA 多頭排列 + 較小週期 MA 空頭排列（拉回）
做多；較大週期空頭排列 + 較小週期多頭排列（反彈）做空，固定風報比停損停利，
套用真實成本模型（沿用 tw_quant.hammer_signal_backtest 的 TradeCost）。

掃「大週期/小週期」組合 × 風報比，均線週期（5/10/20）與 swing lookback（20）
先固定，找到有希望的區域後再細掃、並比照上一輪的方法論做鄰近參數 + 切半樣本
外驗證，避免重蹈「孤立的好數字其實是雜訊」的覆轍。

用法：
    python scripts/backtest_ma_alignment.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tw_quant.hammer_signal_backtest import COST_SCENARIOS, apply_costs  # noqa: E402
from tw_quant.ma_alignment_backtest import build_base_arrays, build_timeframe_state, simulate  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = REPO_ROOT / "data" / "txf_1min.parquet"
SUMMARY_OUT = REPO_ROOT / "data" / "ma_alignment_summary.parquet"

TF_PAIRS = [("60min", "15min"), ("60min", "5min"), ("30min", "5min"), ("30min", "15min")]
R_MULTIPLES = [1.0, 1.5, 2.0, 3.0]
SWING_LOOKBACK = 20
MID_COST = COST_SCENARIOS[1]


def main() -> None:
    df = pd.read_parquet(DATA_PATH)
    arrays = build_base_arrays(df, swing_lookback=SWING_LOOKBACK)
    print(f"資料範圍: {df['datetime'].min()} ~ {df['datetime'].max()}, rows={len(df):,}\n")

    rows = []
    t0 = time.time()
    for large_freq, small_freq in TF_PAIRS:
        large_state = build_timeframe_state(df, freq=large_freq)
        small_state = build_timeframe_state(df, freq=small_freq)
        for r_mult in R_MULTIPLES:
            trades = simulate(arrays, large_state, small_state, r_multiple=r_mult)
            if trades.empty:
                continue

            for cost in COST_SCENARIOS:
                priced = apply_costs(trades, cost)
                n = len(priced)
                wins = priced[priced["pnl_points"] > 0]
                losses = priced[priced["pnl_points"] <= 0]
                gw = wins["pnl_points"].sum()
                gl = -losses["pnl_points"].sum()
                row = dict(
                    large_freq=large_freq, small_freq=small_freq, r_multiple=r_mult,
                    cost_scenario=cost.label, trades=n, win_rate=len(wins) / n,
                    gross_profit_factor=(gw / gl) if gl > 0 else np.nan,
                    net_total_twd=priced["net_twd"].sum(),
                    net_avg_twd=priced["net_twd"].mean(),
                )
                rows.append(row)

    summary = pd.DataFrame(rows)
    summary.to_parquet(SUMMARY_OUT, index=False)
    print(f"跑完，耗時 {time.time()-t0:.1f}s\n")

    pd.set_option("display.width", 240)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.float_format", lambda x: f"{x:,.1f}")

    mid_only = summary[summary["cost_scenario"] == MID_COST.label].sort_values("net_total_twd", ascending=False)
    print("=== 中成本情境（來回60元）排序 ===")
    print(mid_only[["large_freq", "small_freq", "r_multiple", "trades", "win_rate",
                     "gross_profit_factor", "net_total_twd", "net_avg_twd"]].to_string(index=False))

    print("\n=== 每組 (大週期,小週期,風報比) 三種成本情境的淨損益 ===")
    for (lf, sf, r), g in summary.groupby(["large_freq", "small_freq", "r_multiple"]):
        vals = {row["cost_scenario"]: row["net_total_twd"] for _, row in g.iterrows()}
        print(f"  {lf}/{sf} R=1:{r}  trades={g['trades'].iloc[0]:>6}  " +
              "  ".join(f"{k}={v:,.0f}" for k, v in vals.items()))


if __name__ == "__main__":
    main()
