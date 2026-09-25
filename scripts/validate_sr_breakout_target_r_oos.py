"""支撐壓力順勢突破：對IS篩選出的兩個新候選（加上target_r_multiple停利
目標）做OOS驗證。

**重要方法論揭露**：這次同時對兩個候選（target_r=1.0 跟 target_r=3.0）
做OOS驗證，不是嚴格的單一候選單次驗證——是使用者在看過IS比較後明確
要求兩個都測。這代表多重比較的風險比原本的方法論紀律更高：如果剛好
其中一個在OOS恰巧顯著，可信度要打折扣（大略估計，两次近似獨立的檢定
下，真正的型一誤率會從5%上升到接近10%，不是嚴格的5%）。看報告的人
應該把這次的OOS結果當作「比原本單次驗證更需要保留」的證據，不是
「兩個都通過就等於兩倍確定」。

不管哪個候選OOS結果如何，這是這個策略分支最後一次看OOS——不會再回頭
微調參數重測。

用法：
    python scripts/validate_sr_breakout_target_r_oos.py
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
BASE_KW = dict(direction_mode="breakout", channel_window=20, stop_atr_mult=1.0,
               min_range_pct=2.0, trend_slope_threshold_pct=5.0)
CANDIDATES = {
    "target_r=1.0": SupportResistanceFadeConfig(**BASE_KW, target_r_multiple=1.0),
    "target_r=3.0": SupportResistanceFadeConfig(**BASE_KW, target_r_multiple=3.0),
}
ZERO_SLIP_COST = TradeCost("零額外成本(僅稅+手續費)", commission_round_trip=60.0)


def tstat(points: pd.Series) -> float:
    n = len(points)
    if n < 2 or points.std(ddof=1) == 0:
        return np.nan
    return points.mean() / (points.std(ddof=1) / np.sqrt(n))


def summarize(trades: pd.DataFrame, cost: TradeCost) -> dict:
    if trades.empty:
        return dict(n=0)
    t = apply_costs(trades, cost)
    net = t["net_twd"].to_numpy()
    n = len(t)
    win = t[net > 0]
    loss = t[net <= 0]
    gross_win = win["net_twd"].sum()
    gross_loss = -loss["net_twd"].sum()
    pf = gross_win / gross_loss if gross_loss > 0 else np.inf
    return dict(
        n=n, win_rate=(net > 0).mean(), mean_net_twd=net.mean(), sum_net_twd=net.sum(),
        profit_factor=pf, t_stat=tstat(t["pnl_points"]),
    )


def main() -> None:
    df = pd.read_parquet(DATA_PATH)
    is_df = df[df["datetime"] < IS_CUTOFF]
    oos_df = df[df["datetime"] >= IS_CUTOFF]

    for label, cfg in CANDIDATES.items():
        print("=" * 70)
        print(f"候選：{label}")
        print("=" * 70)

        is_trades = backtest(is_df, cfg)
        print(f"IS(2001-2020) n={len(is_trades)}")
        for cost in [ZERO_SLIP_COST, *COST_SCENARIOS]:
            print(f"  {cost.label}: {summarize(is_trades, cost)}")

        oos_trades = backtest(oos_df, cfg)
        print(f"\nOOS(2021-2023) n={len(oos_trades)}")
        for cost in [ZERO_SLIP_COST, *COST_SCENARIOS]:
            print(f"  {cost.label}: {summarize(oos_trades, cost)}")
        if not oos_trades.empty:
            print(f"出場原因分布: {dict(oos_trades['exit_reason'].value_counts())}")
            win_trades = oos_trades[oos_trades["pnl_points"] > 0]
            loss_trades = oos_trades[oos_trades["pnl_points"] <= 0]
            print(f"勝負分布：{len(win_trades)}勝/{len(loss_trades)}負，"
                  f"平均獲利{win_trades['pnl_points'].mean():.0f}點 vs "
                  f"平均虧損{loss_trades['pnl_points'].mean() if len(loss_trades) else float('nan'):.0f}點")

        safe_label = label.replace("=", "_").replace(".", "p")
        is_trades.to_parquet(OUT_DIR / f"sr_breakout_{safe_label}_is_trades.parquet", index=False)
        oos_trades.to_parquet(OUT_DIR / f"sr_breakout_{safe_label}_oos_trades.parquet", index=False)
        print()

    print("=" * 70)
    print("完成。這是sr_breakout(順勢突破)分支最後一次OOS驗證，不會再回頭調整重測。")
    print("=" * 70)


if __name__ == "__main__":
    main()
