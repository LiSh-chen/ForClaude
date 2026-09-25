"""支撐壓力順勢突破候選：檢查「調鬆參數換取更多交易次數」划不划算。

只用IS(2001-2020)資料，不碰OOS、不產生新候選、不重新驗證——純粹回答
「這次會話探討過的4個參數裡，哪個對交易次數影響最大，調整後犧牲多少
統計強度跟交易品質」這個問題，供使用者評估要不要選一個「更多交易但
稍微弱一點」的版本。

用法：
    python scripts/sr_breakout_n_tradeoff_check.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tw_quant.hammer_signal_backtest import COST_SCENARIOS, apply_costs  # noqa: E402
from tw_quant.support_resistance_fade_strategy import SupportResistanceFadeConfig, backtest as bt_sr  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
LOW_COST = COST_SCENARIOS[0]


def detail(label: str, is_df: pd.DataFrame, cfg: SupportResistanceFadeConfig) -> None:
    t = bt_sr(is_df, cfg)
    t = apply_costs(t, LOW_COST)
    n = len(t)
    win = t[t["pnl_points"] > 0]
    loss = t[t["pnl_points"] <= 0]
    hold = (pd.to_datetime(t["exit_date"]) - pd.to_datetime(t["entry_date"])).dt.days
    mean, std = t["pnl_points"].mean(), t["pnl_points"].std(ddof=1)
    tstat = mean / (std / n ** 0.5)
    print(f"--- {label} ---")
    print(f"  n={n}, t={tstat:.3f}, 勝率={len(win)/n:.1%}, 賺賠比={abs(win['pnl_points'].mean()/loss['pnl_points'].mean()):.2f}")
    print(f"  平均持有天數={hold.mean():.1f}, 出場原因={dict(t['exit_reason'].value_counts())}")
    print(f"  跳空滑價(gap_through)佔比={t['exit_reason'].eq('gap_through').mean():.1%}"
          f"（這種出場不受滑價假設保護，實際滑價可能更差）")
    print(f"  淨損益總和={t['net_twd'].sum():,.0f}")


def main() -> None:
    df = pd.read_parquet(REPO_ROOT / "data" / "txf_1min.parquet")
    is_df = df[df["datetime"] < "2021-01-01"]

    print("=" * 70)
    print("1) 網格裡各參數維度對交易次數(n)的影響力（144組breakout組合平均）")
    print("=" * 70)
    sr = pd.read_parquet(REPO_ROOT / "data" / "channel_grid_sr_is.parquet")
    breakout = sr[sr["direction_mode"] == "breakout"]
    for col in ["channel_window", "stop_atr_mult", "min_range_pct", "trend_slope_threshold_pct"]:
        g = breakout.groupby(col)["n"].mean()
        print(f"  {col}: " + ", ".join(f"{k}={v:.0f}筆" for k, v in g.items()))

    print("\n" + "=" * 70)
    print("2) 逐項調鬆鎖定候選(cw=20,sam=1.0,mrp=2.0,tst=5.0, n=262, t=2.08)看實際影響")
    print("=" * 70)
    detail("原始鎖定候選", is_df, SupportResistanceFadeConfig(
        direction_mode="breakout", channel_window=20, stop_atr_mult=1.0, min_range_pct=2.0, trend_slope_threshold_pct=5.0))
    detail("縮短通道(cw 20->10)", is_df, SupportResistanceFadeConfig(
        direction_mode="breakout", channel_window=10, stop_atr_mult=1.0, min_range_pct=2.0, trend_slope_threshold_pct=5.0))
    detail("縮緊停損(sam 1.0->0.5)", is_df, SupportResistanceFadeConfig(
        direction_mode="breakout", channel_window=20, stop_atr_mult=0.5, min_range_pct=2.0, trend_slope_threshold_pct=5.0))
    detail("兩者都調(cw=10, sam=0.5)", is_df, SupportResistanceFadeConfig(
        direction_mode="breakout", channel_window=10, stop_atr_mult=0.5, min_range_pct=2.0, trend_slope_threshold_pct=5.0))

    print("\n完成。min_range_pct/trend_slope_threshold_pct已經在網格測試範圍內最寬鬆的"
          "一端，無法再調鬆增加交易次數（要更寬鬆需要跑網格沒測過的新值，等於開一個"
          "新的探索，不在這次只用既有資料回答問題的範圍內）。")


if __name__ == "__main__":
    main()
