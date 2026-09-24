"""策略一（MTX 夜盤震盪低谷防禦）風報比敏感度：掃 range_r_multiple，套用真實
成本模型（沿用 tw_quant.hammer_signal_backtest 的 TradeCost）。

策略二（突破回測 10MA 跟隨）沒有固定風報比可調（出場是移動停利，不是固定
TP），這裡改成印出它的「實際勝率 / 平均賺賠比」描述性統計作對照，而不是假裝
它有一個可以掃的風報比參數。

用法：
    python scripts/backtest_txf_night_pinbar_rr_sweep.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tw_quant.hammer_signal_backtest import COST_SCENARIOS, apply_costs  # noqa: E402
from tw_quant.txf_night_pinbar import StrategyConfig, backtest  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = REPO_ROOT / "data" / "txf_1min.parquet"
OUT_PATH = REPO_ROOT / "data" / "txf_night_pinbar_rr_sweep.parquet"

R_MULTIPLES = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0]
MID_COST = COST_SCENARIOS[1]


def summarize_trades(trades: pd.DataFrame) -> dict:
    n = len(trades)
    if n == 0:
        return dict(trades=0)
    wins = trades[trades["pnl_points"] > 0]
    losses = trades[trades["pnl_points"] <= 0]
    gw, gl = wins["pnl_points"].sum(), -losses["pnl_points"].sum()
    return dict(
        trades=n, win_rate=len(wins) / n,
        avg_win_pts=wins["pnl_points"].mean() if len(wins) else np.nan,
        avg_loss_pts=losses["pnl_points"].mean() if len(losses) else np.nan,
        win_loss_ratio=(wins["pnl_points"].mean() / -losses["pnl_points"].mean()) if len(losses) and len(wins) else np.nan,
        gross_profit_factor=(gw / gl) if gl > 0 else np.nan,
    )


def main() -> None:
    df = pd.read_parquet(DATA_PATH)
    print(f"資料範圍: {df['datetime'].min()} ~ {df['datetime'].max()}\n")

    rows = []
    for r in R_MULTIPLES:
        cfg = StrategyConfig(range_r_multiple=r)
        trades = backtest(df, cfg)
        range_trades = trades[trades["strategy"] == "range_hammer"]
        s = summarize_trades(range_trades)
        if s["trades"] == 0:
            continue
        for cost in COST_SCENARIOS:
            priced = apply_costs(range_trades, cost)
            row = dict(r_multiple=r, cost_scenario=cost.label, **s,
                       net_total_twd=priced["net_twd"].sum(), net_avg_twd=priced["net_twd"].mean())
            rows.append(row)

    summary = pd.DataFrame(rows)
    summary.to_parquet(OUT_PATH, index=False)

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.float_format", lambda x: f"{x:,.2f}")

    print("=== range_hammer（策略一）風報比敏感度 ===")
    mid = summary[summary["cost_scenario"] == MID_COST.label]
    print(mid[["r_multiple", "trades", "win_rate", "win_loss_ratio", "gross_profit_factor",
               "net_total_twd", "net_avg_twd"]].to_string(index=False))

    print("\n=== 三種成本情境對照 ===")
    for r in R_MULTIPLES:
        sub = summary[summary["r_multiple"] == r]
        if sub.empty:
            continue
        vals = {row["cost_scenario"]: row["net_total_twd"] for _, row in sub.iterrows()}
        print(f"  R=1:{r}  " + "  ".join(f"{k}={v:,.0f}" for k, v in vals.items()))

    print("\n=== breakout_retest（策略二，移動停利、無固定風報比）描述性統計 ===")
    trades_all = backtest(df, StrategyConfig())
    breakout_trades = trades_all[trades_all["strategy"] == "breakout_retest"]
    s2 = summarize_trades(breakout_trades)
    for k, v in s2.items():
        print(f"  {k}: {v}")
    priced2 = apply_costs(breakout_trades, MID_COST)
    print(f"  net_total_twd(中成本): {priced2['net_twd'].sum():,.0f}")


if __name__ == "__main__":
    main()
