"""VWAP趨勢日候選（min_dominant_side_fraction=0.90）深入分析：資金曲線、
最大回撤、Sharpe/Calmar、連續虧損分析、逐年拆解。

這次會話目前為止只看過t值跟淨損益總和，從沒算過「最大回撤」這個實戰
最關鍵的風險指標——策略有沒有正期望值是一回事，實際能不能撐過中間的
連續虧損（心理面+保證金面）是另一回事，這一步補上這個缺口。

用全歷史(2001-2023)連續交易序列做分析，不是重新調參數或重新做OOS
驗證——IS(2001-2020)跟OOS(2021-2023)的t值/顯著性結論早就鎖定過了，
這裡只是把同一組已鎖定參數(frac=0.90)在完整歷史上的交易序列攤開來看
風險輪廓，是分析角度的延伸，不影響先前的統計結論。

用法：
    python scripts/vwap_trend_day_risk_analysis.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tw_quant.hammer_signal_backtest import COST_SCENARIOS, apply_costs  # noqa: E402
from tw_quant.trend_day_strategy import TrendDayConfig, backtest  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = REPO_ROOT / "data" / "txf_1min.parquet"
OUT_DIR = REPO_ROOT / "data"
MID_COST = COST_SCENARIOS[1]
LOCKED_CFG = TrendDayConfig(min_dominant_side_fraction=0.90)


def tstat(x: pd.Series) -> float:
    n = len(x)
    if n < 2 or x.std(ddof=1) == 0:
        return np.nan
    return x.mean() / (x.std(ddof=1) / np.sqrt(n))


def max_drawdown(equity: pd.Series) -> tuple[float, pd.Timestamp, pd.Timestamp]:
    running_max = equity.cummax()
    drawdown = equity - running_max
    trough_idx = drawdown.idxmin()
    mdd = drawdown.loc[trough_idx]
    peak_idx = equity.loc[:trough_idx].idxmax()
    return mdd, peak_idx, trough_idx


def longest_streak(is_win: pd.Series) -> tuple[int, int]:
    max_win_streak = max_loss_streak = cur_win = cur_loss = 0
    for w in is_win:
        if w:
            cur_win += 1
            cur_loss = 0
        else:
            cur_loss += 1
            cur_win = 0
        max_win_streak = max(max_win_streak, cur_win)
        max_loss_streak = max(max_loss_streak, cur_loss)
    return max_win_streak, max_loss_streak


def main() -> None:
    df = pd.read_parquet(DATA_PATH)
    trades = backtest(df, LOCKED_CFG)
    trades = apply_costs(trades, MID_COST)
    trades["trading_date"] = pd.to_datetime(trades["trading_date"])
    trades = trades.sort_values("trading_date").reset_index(drop=True)

    print("=" * 70)
    print("1) 全歷史(2001-2023)整體概況（中成本，frac=0.90鎖定參數）")
    print("=" * 70)
    print(f"n={len(trades)}, t={tstat(trades['pnl_points']):.3f}, win_rate={(trades['net_twd']>0).mean():.3f}, "
          f"sum_net={trades['net_twd'].sum():,.0f}, mean_net={trades['net_twd'].mean():.1f}")

    print("\n" + "=" * 70)
    print("2) 資金曲線與最大回撤（單口，累積TWD）")
    print("=" * 70)
    trades["equity"] = trades["net_twd"].cumsum()
    equity = trades.set_index("trading_date")["equity"]
    mdd, peak_dt, trough_dt = max_drawdown(equity)
    print(f"期末累積淨損益: {equity.iloc[-1]:,.0f} TWD")
    print(f"最大回撤: {mdd:,.0f} TWD（從高點 {peak_dt.date()} 到低點 {trough_dt.date()}）")
    recovery = equity.loc[trough_dt:]
    recovered = recovery[recovery >= equity.loc[peak_dt]]
    recovery_dt = recovered.index[0] if len(recovered) > 0 else None
    print(f"回撤後回本日: {recovery_dt.date() if recovery_dt is not None else '尚未回本（到資料結尾為止）'}")
    print(f"最大回撤 / 期末權益 比例: {abs(mdd)/equity.iloc[-1]:.1%}" if equity.iloc[-1] > 0 else "N/A")

    print("\n" + "=" * 70)
    print("3) 風險調整後報酬（年化，簡化用交易筆數估算，非日曆天數）")
    print("=" * 70)
    years_span = (trades["trading_date"].max() - trades["trading_date"].min()).days / 365.25
    annual_net = trades["net_twd"].sum() / years_span
    annual_std = trades["net_twd"].std() * np.sqrt(len(trades) / years_span)
    sharpe_like = annual_net / annual_std if annual_std > 0 else np.nan
    calmar = annual_net / abs(mdd) if mdd != 0 else np.nan
    print(f"年數: {years_span:.1f}, 年均淨損益: {annual_net:,.0f} TWD, 年化標準差(近似): {annual_std:,.0f} TWD")
    print(f"類Sharpe比率: {sharpe_like:.3f}")
    print(f"Calmar比率(年均報酬/最大回撤): {calmar:.3f}")

    print("\n" + "=" * 70)
    print("4) 連續虧損分析")
    print("=" * 70)
    is_win = trades["net_twd"] > 0
    max_win_streak, max_loss_streak = longest_streak(is_win)
    print(f"最長連勝: {max_win_streak} 筆, 最長連敗: {max_loss_streak} 筆")
    avg_loss = trades.loc[~is_win, "net_twd"].mean()
    print(f"平均每筆虧損: {avg_loss:,.0f} TWD -> 最長連敗估計最大連續虧損: {avg_loss*max_loss_streak:,.0f} TWD（粗估，實際連敗金額會因單筆大小不同而有出入）")

    print("\n" + "=" * 70)
    print("5) 逐年拆解")
    print("=" * 70)
    trades["year"] = trades["trading_date"].dt.year
    yearly = trades.groupby("year").agg(
        n=("net_twd", "size"), win_rate=("net_twd", lambda x: (x > 0).mean()),
        sum_net=("net_twd", "sum"), t_stat=("pnl_points", tstat),
    )
    pd.set_option("display.width", 200)
    print(yearly.to_string())
    print(f"\n淨損益為正的年數: {(yearly['sum_net']>0).sum()} / {len(yearly)}")

    trades.to_parquet(OUT_DIR / "trend_day_relaxed_full_history_trades.parquet", index=False)
    print("\nsaved: trend_day_relaxed_full_history_trades.parquet")


if __name__ == "__main__":
    main()
