"""三腿策略（開盤上衝/午盤放空/盤中翻多+量能濾網）合併資金曲線、最大
回撤、Sharpe/Calmar、連續虧損、逐年拆解——跟 vwap_trend_day_risk_
analysis.py 同一套方法，這次套用在三腿策略上，因為 low_cost_
reassessment.py 已經顯示這個策略在低成本情境下比VWAP趨勢日更有希望，
需要同樣檢查資金曲線品質（statistical significance不代表能撐過中間
的回撤）。

主要用低成本(30元)情境呈現（這次分析的起點就是「高成本不切實際」這個
前提），中成本(60元)一併附上當對照。

用法：
    python scripts/three_legs_risk_analysis.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tw_quant.hammer_signal_backtest import COST_SCENARIOS, apply_costs  # noqa: E402
from tw_quant.lunch_reversal_strategy import backtest as backtest_lunch  # noqa: E402
from tw_quant.opening_rally_strategy import backtest as backtest_opening  # noqa: E402
from tw_quant.technical_indicators import daily_indicators, filter_trades_by_volume  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = REPO_ROOT / "data" / "txf_1min.parquet"
OUT_DIR = REPO_ROOT / "data"
LOW_COST, MID_COST, _ = COST_SCENARIOS
IS_CUTOFF = "2021-01-01"


def tstat(x: pd.Series) -> float:
    n = len(x)
    if n < 2 or x.std(ddof=1) == 0:
        return np.nan
    return x.mean() / (x.std(ddof=1) / np.sqrt(n))


def max_drawdown(equity: pd.Series) -> tuple[float, int, int]:
    """equity 必須是位置(0..n-1)索引的序列（同一天可能有多筆交易，日期
    索引會重複，不能拿來做 idxmax/idxmin），回傳的 peak/trough 是位置。"""
    running_max = equity.cummax()
    drawdown = equity - running_max
    trough_pos = int(np.argmin(drawdown.to_numpy()))
    mdd = drawdown.iloc[trough_pos]
    peak_pos = int(np.argmax(equity.to_numpy()[: trough_pos + 1]))
    return mdd, peak_pos, trough_pos


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
    is_df = df[df["datetime"] < IS_CUTOFF]
    is_indicators = daily_indicators(is_df)
    vol_threshold = is_indicators["vol_ratio_lag1"].quantile(2 / 3)

    open_trades = backtest_opening(df)
    lunch_trades = backtest_lunch(df)
    short_leg = lunch_trades[lunch_trades["leg"] == "short_lunch_dip"]
    long_leg_raw = lunch_trades[lunch_trades["leg"] == "long_afternoon_rebound"]
    long_leg = filter_trades_by_volume(long_leg_raw, df, vol_threshold)

    trades = pd.concat([open_trades, short_leg, long_leg], ignore_index=True)
    trades = apply_costs(trades, LOW_COST).rename(columns={"net_twd": "net_twd_low"})
    trades["net_twd_mid"] = apply_costs(trades, MID_COST)["net_twd"]
    trades["trading_date"] = pd.to_datetime(trades["trading_date"])
    trades = trades.sort_values(["trading_date", "entry_dt"]).reset_index(drop=True)

    print("=" * 70)
    print("1) 全歷史(2001-2023)整體概況（三腿合併）")
    print("=" * 70)
    print(f"n={len(trades)}, t={tstat(trades['pnl_points']):.3f}, "
          f"win_rate={(trades['net_twd_low']>0).mean():.3f}")
    print(f"低成本 sum_net={trades['net_twd_low'].sum():,.0f}, mean_net={trades['net_twd_low'].mean():.1f}")
    print(f"中成本 sum_net={trades['net_twd_mid'].sum():,.0f}, mean_net={trades['net_twd_mid'].mean():.1f}")

    for cost_label, col in [("低成本", "net_twd_low"), ("中成本", "net_twd_mid")]:
        print("\n" + "=" * 70)
        print(f"2) 資金曲線與最大回撤（{cost_label}，單口，累積TWD，每日多筆交易依entry_dt排序後逐筆累加）")
        print("=" * 70)
        equity = trades[col].cumsum().reset_index(drop=True)
        mdd, peak_pos, trough_pos = max_drawdown(equity)
        peak_dt = trades["trading_date"].iloc[peak_pos]
        trough_dt = trades["trading_date"].iloc[trough_pos]
        print(f"期末累積淨損益: {equity.iloc[-1]:,.0f} TWD")
        print(f"最大回撤: {mdd:,.0f} TWD（從高點 {peak_dt.date()} 到低點 {trough_dt.date()}）")
        peak_equity = equity.iloc[peak_pos]
        recovery = equity.iloc[trough_pos:]
        recovered = recovery[recovery >= peak_equity]
        recovery_dt = trades["trading_date"].iloc[recovered.index[0]] if len(recovered) > 0 else None
        print(f"回撤後回本日: {recovery_dt.date() if recovery_dt is not None else '尚未回本（到資料結尾為止）'}")

        years_span = (trades["trading_date"].max() - trades["trading_date"].min()).days / 365.25
        annual_net = trades[col].sum() / years_span
        annual_std = trades[col].std() * np.sqrt(len(trades) / years_span)
        sharpe_like = annual_net / annual_std if annual_std > 0 else np.nan
        calmar = annual_net / abs(mdd) if mdd != 0 else np.nan
        print(f"年均淨損益: {annual_net:,.0f} TWD, 類Sharpe比率: {sharpe_like:.3f}, Calmar比率: {calmar:.3f}")

        is_win = trades[col] > 0
        max_win_streak, max_loss_streak = longest_streak(is_win)
        print(f"最長連勝: {max_win_streak} 筆, 最長連敗: {max_loss_streak} 筆")

    print("\n" + "=" * 70)
    print("3) 逐年拆解（低成本 vs 中成本 並列）")
    print("=" * 70)
    trades["year"] = trades["trading_date"].dt.year
    yearly = trades.groupby("year").agg(
        n=("net_twd_low", "size"),
        sum_net_low=("net_twd_low", "sum"), sum_net_mid=("net_twd_mid", "sum"),
    )
    pd.set_option("display.width", 200)
    print(yearly.to_string())
    print(f"\n低成本淨損益為正的年數: {(yearly['sum_net_low']>0).sum()} / {len(yearly)}")
    print(f"中成本淨損益為正的年數: {(yearly['sum_net_mid']>0).sum()} / {len(yearly)}")

    trades.to_parquet(OUT_DIR / "three_legs_full_history_trades.parquet", index=False)
    print("\nsaved: three_legs_full_history_trades.parquet")


if __name__ == "__main__":
    main()
