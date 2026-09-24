"""「長下影線 + 次根紅K」訊號，接固定風報比停損停利，套真實成本模型，
比較正向（做多）vs 反向（做空）在不同比例門檻 × 風報比下的績效。

用法：
    python scripts/backtest_hammer_ratio_rr.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tw_quant.hammer_signal_backtest import COST_SCENARIOS, apply_costs, build_base_arrays, simulate  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = REPO_ROOT / "data" / "txf_1min.parquet"
SUMMARY_OUT = REPO_ROOT / "data" / "hammer_ratio_rr_summary.parquet"
TRADES_OUT_DIR = REPO_ROOT / "data" / "hammer_ratio_rr_trades"

RATIO_THRESHOLDS = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
R_MULTIPLES = [1.0, 1.5, 2.0, 3.0]
DIRECTIONS = ["long", "short"]


def main() -> None:
    df = pd.read_parquet(DATA_PATH)
    arrays = build_base_arrays(df)
    print(f"資料範圍: {df['datetime'].min()} ~ {df['datetime'].max()}, rows={len(df):,}\n")

    TRADES_OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_summary_rows = []
    t0 = time.time()

    for direction in DIRECTIONS:
        for ratio_th in RATIO_THRESHOLDS:
            for r_mult in R_MULTIPLES:
                trades = simulate(arrays, ratio_threshold=ratio_th, r_multiple=r_mult, direction=direction)
                if trades.empty:
                    continue

                # 三個成本情境只差手續費，稅率共用；用第一個情境算一次稅金/毛損益即可，
                # 其餘情境的淨損益 = 這次的 net_twd 再扣「情境間的手續費差額 * 筆數」。
                priced = apply_costs(trades, COST_SCENARIOS[0])
                n = len(priced)
                wins = priced[priced["pnl_points"] > 0]
                losses = priced[priced["pnl_points"] <= 0]
                gross_win_pts = wins["pnl_points"].sum()
                gross_loss_pts = -losses["pnl_points"].sum()
                gross_total_twd = priced["gross_twd"].sum()
                total_tax_twd = (priced["cost_twd"] - COST_SCENARIOS[0].commission_round_trip).sum()

                row = dict(
                    direction=direction, ratio_ge=ratio_th, r_multiple=r_mult,
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

                all_summary_rows.append(row)

                trades_path = TRADES_OUT_DIR / f"{direction}_ratio{ratio_th}_r{r_mult}.parquet"
                priced.to_parquet(trades_path, index=False)

    summary = pd.DataFrame(all_summary_rows)
    summary.to_parquet(SUMMARY_OUT, index=False)
    print(f"跑完 {len(summary)} 組參數組合，耗時 {time.time()-t0:.1f}s\n")

    pd.set_option("display.width", 260)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.float_format", lambda x: f"{x:,.1f}")

    for direction in DIRECTIONS:
        print(f"\n=== {direction} ===")
        sub = summary[summary["direction"] == direction].sort_values(["ratio_ge", "r_multiple"])
        cols = ["ratio_ge", "r_multiple", "trades", "win_rate", "gross_profit_factor",
                "gross_total_twd", "total_tax_twd", "net_after_tax_twd", "breakeven_commission_twd"] + \
               [c for c in sub.columns if c.startswith("net_twd[")]
        print(sub[cols].to_string(index=False))


if __name__ == "__main__":
    main()
