"""跑小台指夜盤長下影線策略（震盪低谷防禦 / 突破回測 10MA 跟隨）回測，
印出績效摘要，並把交易明細、權益曲線存成 parquet。

用法：
    python scripts/backtest_txf_night_pinbar.py
    python scripts/backtest_txf_night_pinbar.py --swing-lookback 10
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tw_quant.txf_night_pinbar import StrategyConfig, backtest  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = REPO_ROOT / "data" / "txf_1min.parquet"
TRADES_OUT = REPO_ROOT / "data" / "txf_night_pinbar_trades.parquet"
EQUITY_OUT = REPO_ROOT / "data" / "txf_night_pinbar_equity.parquet"


def summarize(trades: pd.DataFrame, strategy: str) -> dict:
    t = trades[trades["strategy"] == strategy]
    if t.empty:
        return dict(strategy=strategy, trades=0)

    wins = t[t["pnl_points"] > 0]
    losses = t[t["pnl_points"] <= 0]
    gross_win = wins["pnl_twd"].sum()
    gross_loss = -losses["pnl_twd"].sum()

    equity = t["pnl_twd"].cumsum()
    running_max = equity.cummax()
    drawdown = equity - running_max
    max_dd = drawdown.min()

    return dict(
        strategy=strategy,
        trades=len(t),
        win_rate=len(wins) / len(t),
        total_pnl_twd=t["pnl_twd"].sum(),
        avg_pnl_twd=t["pnl_twd"].mean(),
        avg_win_twd=wins["pnl_twd"].mean() if len(wins) else np.nan,
        avg_loss_twd=losses["pnl_twd"].mean() if len(losses) else np.nan,
        profit_factor=(gross_win / gross_loss) if gross_loss > 0 else np.nan,
        max_drawdown_twd=max_dd,
        first_trade=t["entry_dt"].min(),
        last_trade=t["entry_dt"].max(),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--swing-lookback", type=int, default=20)
    parser.add_argument("--sensitivity", action="store_true", help="額外跑 N=10/20/30 的敏感度對照")
    args = parser.parse_args()

    df = pd.read_parquet(DATA_PATH)
    cfg = StrategyConfig(swing_lookback=args.swing_lookback)

    trades = backtest(df, cfg)
    TRADES_OUT.parent.mkdir(parents=True, exist_ok=True)
    trades.to_parquet(TRADES_OUT, index=False)

    print(f"資料範圍: {df['datetime'].min()} ~ {df['datetime'].max()}")
    print(f"swing_lookback (N) = {cfg.swing_lookback}")
    print(f"總交易筆數: {len(trades):,}\n")

    summaries = []
    for strat in ["range_hammer", "breakout_retest"]:
        s = summarize(trades, strat)
        summaries.append(s)
        print(f"=== {strat} ===")
        for k, v in s.items():
            if k == "strategy":
                continue
            print(f"  {k}: {v}")
        print()

    equity_rows = []
    for strat, sub in trades.groupby("strategy"):
        sub = sub.sort_values("entry_dt")
        eq = sub["pnl_twd"].cumsum()
        equity_rows.append(pd.DataFrame({"strategy": strat, "entry_dt": sub["entry_dt"], "equity_twd": eq}))
    if equity_rows:
        pd.concat(equity_rows).to_parquet(EQUITY_OUT, index=False)

    if args.sensitivity:
        print("=== N 敏感度對照 (10 / 20 / 30) ===")
        for n in (10, 20, 30):
            sens_trades = backtest(df, StrategyConfig(swing_lookback=n))
            for strat in ["range_hammer", "breakout_retest"]:
                s = summarize(sens_trades, strat)
                print(f"N={n:>2} {strat:<16} trades={s.get('trades',0):>5} "
                      f"win_rate={s.get('win_rate', float('nan')):.2%} "
                      f"total_pnl={s.get('total_pnl_twd', float('nan')):,.0f} "
                      f"max_dd={s.get('max_drawdown_twd', float('nan')):,.0f}")


if __name__ == "__main__":
    main()
