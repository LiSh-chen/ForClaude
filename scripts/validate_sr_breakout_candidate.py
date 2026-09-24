"""通道策略順逆勢全面交叉網格（scripts/comprehensive_channel_strategy_grid.py）
篩出的唯一「順勢」候選（channel_window=20, stop_atr_mult=1.0, min_range_pct=2.0,
trend_slope_threshold_pct=5.0），子區間 t 值 [1.81, 1.66, 1.12, -0.10]（3/4
同方向，第4個接近零不是真的反向），對 OOS 只驗證一次。

用法：
    python scripts/validate_sr_breakout_candidate.py
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
LOCKED_CFG = SupportResistanceFadeConfig(
    direction_mode="breakout", channel_window=20, stop_atr_mult=1.0,
    min_range_pct=2.0, trend_slope_threshold_pct=5.0,
)
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
    print("1) IS 整體（候選鎖定參數）")
    print("=" * 70)
    is_trades = backtest(is_df, LOCKED_CFG)
    for cost in [ZERO_SLIP_COST, *COST_SCENARIOS]:
        print(summarize(is_trades, cost, cost.label))
    if not is_trades.empty:
        print(f"出場原因分布:\n{is_trades['exit_reason'].value_counts()}")

    print("\n" + "=" * 70)
    print("2) OOS 唯一一次驗證")
    print("=" * 70)
    oos_trades = backtest(oos_df, LOCKED_CFG)
    print(f"OOS 交易筆數: {len(oos_trades)}")
    for cost in [ZERO_SLIP_COST, *COST_SCENARIOS]:
        print(summarize(oos_trades, cost, cost.label))
    if not oos_trades.empty:
        print(f"出場原因分布:\n{oos_trades['exit_reason'].value_counts()}")
        win_trades = oos_trades[oos_trades["pnl_points"] > 0]
        loss_trades = oos_trades[oos_trades["pnl_points"] <= 0]
        print(f"\n勝負分布：{len(win_trades)}勝/{len(loss_trades)}負，"
              f"平均獲利{win_trades['pnl_points'].mean():.0f}點 vs 平均虧損{loss_trades['pnl_points'].mean():.0f}點")

    is_trades.to_parquet(OUT_DIR / "sr_breakout_candidate_is_trades.parquet", index=False)
    oos_trades.to_parquet(OUT_DIR / "sr_breakout_candidate_oos_trades.parquet", index=False)
    print("\nsaved: sr_breakout_candidate_is_trades / sr_breakout_candidate_oos_trades .parquet")


if __name__ == "__main__":
    main()
