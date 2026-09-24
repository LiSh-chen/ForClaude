"""在「長下影線 + 次根紅K + 固定風報比停損停利」規則上，疊加一個更高時間週期
（預設 60 分鐘 SMA20）的趨勢過濾器，比較三種模式：

- none    ：不過濾（跟 backtest_hammer_ratio_rr.py 的基準一致，作對照）
- with    ：順higher-TF趨勢——做多只在高週期多頭、做空只在高週期空頭
- against ：逆higher-TF趨勢——跟 with 相反

目的：檢驗「疊加更高時間週期過濾條件」是否真的能透過減少交易筆數（進而降低
稅金總額）讓策略在扣完真實成本後轉為淨正，而不是繼續在原本規則的參數
（比例門檻/風報比）上打轉。

用法：
    python scripts/backtest_hammer_ratio_htf_filter.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tw_quant.hammer_signal_backtest import (  # noqa: E402
    COST_SCENARIOS, apply_costs, build_base_arrays, build_htf_trend, simulate,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = REPO_ROOT / "data" / "txf_1min.parquet"
SUMMARY_OUT = REPO_ROOT / "data" / "hammer_ratio_htf_filter_summary.parquet"

RATIO_THRESHOLDS = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
R_MULTIPLES = [1.0, 1.5, 2.0, 3.0]
DIRECTIONS = ["long", "short"]
TREND_MODES = ["none", "with", "against"]

HTF_FREQ = "60min"
HTF_SMA_WINDOW = 20


def main() -> None:
    df = pd.read_parquet(DATA_PATH)
    arrays = build_base_arrays(df)
    htf_trend = build_htf_trend(df, freq=HTF_FREQ, sma_window=HTF_SMA_WINDOW)
    print(f"資料範圍: {df['datetime'].min()} ~ {df['datetime'].max()}, rows={len(df):,}")
    print(f"高週期過濾器: {HTF_FREQ} SMA{HTF_SMA_WINDOW}\n")

    rows = []
    t0 = time.time()

    for trend_mode in TREND_MODES:
        for direction in DIRECTIONS:
            for ratio_th in RATIO_THRESHOLDS:
                for r_mult in R_MULTIPLES:
                    trades = simulate(
                        arrays, ratio_threshold=ratio_th, r_multiple=r_mult, direction=direction,
                        htf_trend=htf_trend if trend_mode != "none" else None, trend_mode=trend_mode,
                    )
                    if trades.empty:
                        continue

                    priced = apply_costs(trades, COST_SCENARIOS[0])
                    n = len(priced)
                    wins = priced[priced["pnl_points"] > 0]
                    losses = priced[priced["pnl_points"] <= 0]
                    gross_win_pts = wins["pnl_points"].sum()
                    gross_loss_pts = -losses["pnl_points"].sum()
                    gross_total_twd = priced["gross_twd"].sum()
                    total_tax_twd = (priced["cost_twd"] - COST_SCENARIOS[0].commission_round_trip).sum()

                    row = dict(
                        trend_mode=trend_mode, direction=direction, ratio_ge=ratio_th, r_multiple=r_mult,
                        trades=n, win_rate=len(wins) / n,
                        gross_profit_factor=(gross_win_pts / gross_loss_pts) if gross_loss_pts > 0 else np.nan,
                        gross_total_twd=gross_total_twd,
                        total_tax_twd=total_tax_twd,
                        net_after_tax_twd=gross_total_twd - total_tax_twd,
                        breakeven_commission_twd=(gross_total_twd - total_tax_twd) / n,
                    )
                    for cost in COST_SCENARIOS:
                        net = gross_total_twd - total_tax_twd - n * cost.commission_round_trip
                        row[f"net_twd[{cost.label}]"] = net
                    rows.append(row)

    summary = pd.DataFrame(rows)
    summary.to_parquet(SUMMARY_OUT, index=False)
    print(f"跑完 {len(summary)} 組參數組合，耗時 {time.time()-t0:.1f}s\n")

    pd.set_option("display.width", 260)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.float_format", lambda x: f"{x:,.1f}")

    print("=== 各 trend_mode 的整體樣貌（trades 加總、net_after_tax 加總、正值筆數）===")
    agg = summary.groupby("trend_mode").agg(
        total_trades=("trades", "sum"),
        total_net_after_tax=("net_after_tax_twd", "sum"),
        configs_net_positive=("net_after_tax_twd", lambda s: (s > 0).sum()),
        best_net_after_tax=("net_after_tax_twd", "max"),
        best_breakeven_commission=("breakeven_commission_twd", "max"),
    )
    print(agg.to_string())

    print("\n=== 每個 trend_mode 下最好的 5 組 ===")
    for trend_mode in TREND_MODES:
        sub = summary[summary["trend_mode"] == trend_mode].sort_values("net_after_tax_twd", ascending=False).head(5)
        print(f"\n-- {trend_mode} --")
        cols = ["direction", "ratio_ge", "r_multiple", "trades", "win_rate", "gross_profit_factor",
                "net_after_tax_twd", "breakeven_commission_twd"]
        print(sub[cols].to_string(index=False))


if __name__ == "__main__":
    main()
