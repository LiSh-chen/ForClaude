"""支撐壓力順勢突破（sr_breakout）候選的深入分析，接續
scripts/validate_sr_breakout_candidate.py 跟 scripts/comprehensive_channel_strategy_grid.py
已經做過的IS篩選+OOS單次驗證，這裡只用「已經產生過的資料」做進一步
刻畫，不重新看OOS、不調整參數（維持這次會話一貫的紀律）：

1. 參數鄰域穩健性：整個breakout網格（144組，來自
   data/channel_grid_sr_is.parquet）是不是只有鎖定的那一組顯著，還是
   附近參數也普遍偏正——區分「真的有訊號」跟「288組亂猜挑到一組」。
2. 全歷史（IS+OOS合併，2001-2023）風險指標：權益曲線、最大回撤、
   Calmar比率、連勝連敗，比照三腿策略風險分析的方法。
3. 跟已驗證三腿策略的每日損益相關性：分散配置價值。
4. 逐年損益：獲利有沒有集中在少數年份。
5. IS vs OOS 交易特徵比較（出場原因分布、勝率、賺賠比）：機制有沒有
   質變，還是單純樣本數小、運氣不好。

用法：
    python scripts/sr_breakout_deep_dive.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tw_quant.hammer_signal_backtest import COST_SCENARIOS, apply_costs  # noqa: E402
from tw_quant.support_resistance_fade_strategy import SupportResistanceFadeConfig, backtest as bt_sr  # noqa: E402
from tw_quant.opening_rally_strategy import OpeningRallyConfig, backtest as bt_og  # noqa: E402
from tw_quant.lunch_reversal_strategy import LunchReversalConfig, backtest as bt_lunch  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = REPO_ROOT / "data" / "txf_1min.parquet"
LOW_COST = COST_SCENARIOS[0]
IS_CUTOFF = "2021-01-01"
LOCKED_CFG = SupportResistanceFadeConfig(
    direction_mode="breakout", channel_window=20, stop_atr_mult=1.0,
    min_range_pct=2.0, trend_slope_threshold_pct=5.0,
)


def max_drawdown(equity: pd.Series) -> tuple[float, int, int]:
    """equity 必須是位置(0..n-1)索引，回傳 (mdd, peak_pos, trough_pos)。"""
    running_max = equity.cummax()
    drawdown = equity - running_max
    trough_pos = int(np.argmin(drawdown.to_numpy()))
    mdd = drawdown.iloc[trough_pos]
    peak_pos = int(np.argmax(equity.to_numpy()[: trough_pos + 1]))
    return mdd, peak_pos, trough_pos


def streaks(net: pd.Series) -> tuple[int, int]:
    win = (net > 0).astype(int)
    max_win, max_loss, cur_win, cur_loss = 0, 0, 0, 0
    for w in win:
        if w:
            cur_win += 1; cur_loss = 0
        else:
            cur_loss += 1; cur_win = 0
        max_win = max(max_win, cur_win)
        max_loss = max(max_loss, cur_loss)
    return max_win, max_loss


def main() -> None:
    df = pd.read_parquet(DATA_PATH)

    print("=" * 70)
    print("1) 參數鄰域穩健性（breakout網格144組，不含fade）")
    print("=" * 70)
    sr = pd.read_parquet(REPO_ROOT / "data" / "channel_grid_sr_is.parquet")
    breakout = sr[sr["direction_mode"] == "breakout"]
    print(f"144組全部t_stat為正：{(breakout['t_stat'] > 0).all()}（min={breakout['t_stat'].min():.2f}, "
          f"mean={breakout['t_stat'].mean():.2f}, max={breakout['t_stat'].max():.2f}）")
    print("\n分維度平均t值（找出真正在驅動績效的參數）：")
    for col in ["channel_window", "stop_atr_mult", "min_range_pct", "trend_slope_threshold_pct"]:
        g = breakout.groupby(col)["t_stat"].mean()
        print(f"  {col}: " + ", ".join(f"{k}={v:.2f}" for k, v in g.items()))
    print("\n鎖定的候選(cw=20,sam=1.0,mrp=2.0,tst=5.0)在144組裡排名："
          f"{(breakout['t_stat'] > 2.082037).sum() + 1} / 144（t=2.08是網格裡的全域最大值）")

    print("\n" + "=" * 70)
    print("2) 全歷史（2001-2023）風險指標")
    print("=" * 70)
    trades = bt_sr(df, LOCKED_CFG)
    trades = apply_costs(trades, LOW_COST)
    daily_net = trades.groupby("exit_date")["net_twd"].sum().sort_index()
    equity = daily_net.cumsum().reset_index(drop=True)
    mdd, peak_pos, trough_pos = max_drawdown(equity)
    peak_date = trades.groupby("exit_date")["net_twd"].sum().sort_index().index[peak_pos]
    trough_date = trades.groupby("exit_date")["net_twd"].sum().sort_index().index[trough_pos]
    recovery_date = None
    peak_val = equity.iloc[peak_pos]
    for i in range(trough_pos, len(equity)):
        if equity.iloc[i] >= peak_val:
            recovery_date = trades.groupby("exit_date")["net_twd"].sum().sort_index().index[i]
            break
    total_net = daily_net.sum()
    calmar = total_net / abs(mdd) if mdd != 0 else np.nan
    mean_d, std_d = daily_net.mean(), daily_net.std(ddof=1)
    sharpe_like = mean_d / std_d * np.sqrt(252) if std_d else np.nan
    max_win_streak, max_loss_streak = streaks(daily_net)
    print(f"交易筆數: {len(trades)}, 總淨損益: {total_net:,.0f} TWD")
    print(f"最大回撤(MDD): {mdd:,.0f} TWD，高點 {peak_date} -> 低點 {trough_date}"
          + (f"，回復於 {recovery_date}" if recovery_date else "，尚未回復"))
    print(f"Calmar(總淨損益/|MDD|): {calmar:.3f}")
    print(f"年化Sharpe-like: {sharpe_like:.3f}")
    print(f"最長連勝(交易日): {max_win_streak}，最長連虧(交易日): {max_loss_streak}")

    print("\n逐年淨損益（檢查集中度）：")
    yearly = trades.assign(year=pd.to_datetime(trades["exit_date"]).dt.year).groupby("year")["net_twd"].sum()
    print(yearly.apply(lambda x: f"{x:,.0f}"))
    top2_share = yearly.abs().nlargest(2).sum() / yearly.abs().sum()
    print(f"最大兩年(絕對值)佔全歷史總絕對淨損益的比例: {top2_share:.1%}")

    print("\n" + "=" * 70)
    print("3) 跟三腿策略(已驗證)的每日損益相關性")
    print("=" * 70)
    og_trades = apply_costs(bt_og(df, OpeningRallyConfig()), LOW_COST)
    lunch_trades = apply_costs(bt_lunch(df, LunchReversalConfig()), LOW_COST)
    three_legs_daily = pd.concat([
        og_trades.assign(date=og_trades["trading_date"])[["date", "net_twd"]],
        lunch_trades.assign(date=lunch_trades["trading_date"])[["date", "net_twd"]],
    ]).groupby("date")["net_twd"].sum()
    three_legs_daily.index = pd.to_datetime(three_legs_daily.index)

    sr_daily = daily_net.copy()
    sr_daily.index = pd.to_datetime(sr_daily.index)

    merged = pd.DataFrame({"three_legs": three_legs_daily, "sr_breakout": sr_daily}).fillna(0.0)
    corr_daily = merged["three_legs"].corr(merged["sr_breakout"])
    print(f"每日損益相關係數（三腿 vs sr_breakout，兩邊都補0，n={len(merged)}）: {corr_daily:.4f}")

    yearly_tl = three_legs_daily.groupby(three_legs_daily.index.year).sum()
    yearly_sr = sr_daily.groupby(sr_daily.index.year).sum()
    yearly_merged = pd.DataFrame({"three_legs": yearly_tl, "sr_breakout": yearly_sr}).fillna(0.0)
    corr_yearly = yearly_merged["three_legs"].corr(yearly_merged["sr_breakout"])
    print(f"年度損益相關係數: {corr_yearly:.4f}")

    combined_daily = merged["three_legs"] + merged["sr_breakout"]
    combined_std = combined_daily.std(ddof=1)
    tl_std = merged["three_legs"].std(ddof=1)
    print(f"加入sr_breakout後，合併每日損益標準差變化: {tl_std:,.0f} -> {combined_std:,.0f} TWD"
          f"（{'降低' if combined_std < tl_std else '提高'} {abs(combined_std - tl_std) / tl_std:.1%}）")

    print("\n" + "=" * 70)
    print("4) IS vs OOS 交易特徵比較（機制有沒有質變）")
    print("=" * 70)
    is_trades = trades[pd.to_datetime(trades["entry_date"]) < IS_CUTOFF]
    oos_trades = trades[pd.to_datetime(trades["entry_date"]) >= IS_CUTOFF]
    for label, t in [("IS(2001-2020)", is_trades), ("OOS(2021-2023)", oos_trades)]:
        win = t[t["pnl_points"] > 0]
        loss = t[t["pnl_points"] <= 0]
        hold_days = (pd.to_datetime(t["exit_date"]) - pd.to_datetime(t["entry_date"])).dt.days
        print(f"\n{label}: n={len(t)}, 勝率={len(win)/len(t):.1%}, "
              f"平均獲利={win['pnl_points'].mean():.0f}點, 平均虧損={loss['pnl_points'].mean():.0f}點, "
              f"賺賠比={abs(win['pnl_points'].mean()/loss['pnl_points'].mean()):.2f}, "
              f"平均持有天數={hold_days.mean():.1f}天")
        print(f"  出場原因分布: {dict(t['exit_reason'].value_counts())}")

    print("\n完成。")


if __name__ == "__main__":
    main()
