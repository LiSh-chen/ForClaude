"""把 test_night_session_gap_follow.py 找到的「大跳空延續」線索做最後一輪
穩健性檢查，判斷夠不夠格進入實戰模擬（不是判斷夠不夠格直接上真實資金）。

延續前一版的疑慮：最大3筆獲利集中在2020年3月COVID崩盤週，去掉後t值
從2.61掉到1.65。這裡做更完整的穩健性檢查，而不是只看「去掉最大3筆」
這種容易被質疑的手法：
1. 逐年損益：是不是只有2020年賺、其他年都虧？
2. 完全排除2020整年：還剩不剩得住正報酬？
3. 進一步排除2022年（另一個高波動年）：只用相對平靜的年份還work嗎？
4. 多空分開看：不是單一方向的巧合？
5. 風險指標：最大單筆虧損、每年交易頻率、平均每筆淨損益——這些數字是
   實際要不要進場模擬、用多大部位模擬的依據。

用法：
    python scripts/finalize_night_gap_follow_for_simulation.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import scripts.test_night_session_gap_fade as gap_fade  # noqa: E402
import scripts.test_night_session_gap_follow as gap_follow  # noqa: E402
from tw_quant.hammer_signal_backtest import TradeCost, apply_costs  # noqa: E402


def main() -> None:
    df = pd.read_parquet(gap_fade.DATA_PATH)
    day_close = gap_fade.build_day_close(df)
    trades = gap_follow.simulate_follow(df, day_close, 1.0)
    trades = apply_costs(trades, TradeCost("夜盤", commission_round_trip=60.0, slippage_points_round_trip=3.0))
    trades["year"] = pd.to_datetime(trades["trading_date"]).dt.year

    print("=" * 78)
    print("1) 逐年損益（檢查是不是只有2020年在賺）")
    print("=" * 78)
    yearly = trades.groupby("year").agg(n=("pnl_points", "size"), sum_pts=("pnl_points", "sum"),
                                          win_rate=("pnl_points", lambda x: (x > 0).mean() * 100))
    print(yearly)
    print(f"\n正貢獻年數: {(yearly['sum_pts'] > 0).sum()}/{len(yearly)}")

    print("\n" + "=" * 78)
    print("2) 排除2020整年後還剩不剩得住")
    print("=" * 78)
    no2020 = trades[trades["year"] != 2020]
    t, n, m = gap_fade.tstat(no2020["pnl_points"])
    print(f"  n={n} t={t:.2f} mean={m:.2f}點 sum={no2020['pnl_points'].sum():.0f}點 "
          f"勝率={(no2020['pnl_points']>0).mean()*100:.1f}%")

    print("\n" + "=" * 78)
    print("3) 再排除2022年（另一個高波動年），只用相對平靜年份")
    print("=" * 78)
    calm = trades[~trades["year"].isin([2020, 2022])]
    t, n, m = gap_fade.tstat(calm["pnl_points"])
    print(f"  n={n} t={t:.2f} mean={m:.2f}點 sum={calm['pnl_points'].sum():.0f}點 "
          f"勝率={(calm['pnl_points']>0).mean()*100:.1f}%")

    print("\n" + "=" * 78)
    print("4) 多空分開看")
    print("=" * 78)
    for d in ["long", "short"]:
        sub = trades[trades["direction"] == d]
        t, n, m = gap_fade.tstat(sub["pnl_points"])
        print(f"  {d}: n={n} t={t:.2f} mean={m:.2f}點 勝率={(sub['pnl_points']>0).mean()*100:.1f}%")

    print("\n" + "=" * 78)
    print("5) 風險/頻率指標（模擬部位規劃用）")
    print("=" * 78)
    years_span = (pd.to_datetime(trades["trading_date"]).max() - pd.to_datetime(trades["trading_date"]).min()).days / 365
    print(f"  總筆數: {len(trades)}，年均交易次數: {len(trades)/years_span:.1f}")
    print(f"  最大單筆虧損: {trades['pnl_points'].min():.1f}點 ({trades['net_twd'].min():,.0f}元)")
    print(f"  最大單筆獲利: {trades['pnl_points'].max():.1f}點 ({trades['net_twd'].max():,.0f}元)")
    print(f"  平均每筆淨損益: {trades['net_twd'].mean():,.0f}元")
    print(f"  出場原因分布:\n{trades['exit_reason'].value_counts().to_string()}")
    print(f"  停損觸發時最大虧損: {trades[trades['exit_reason']=='stop']['pnl_points'].min():.1f}點")


if __name__ == "__main__":
    main()
