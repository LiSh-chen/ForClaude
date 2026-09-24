"""檢驗兩個「證據不夠強、無法單獨進場」的候選策略——VWAP趨勢日
(trend_day_relaxed_oos_trades.parquet, min_dominant_side_fraction=0.90)
和壓力支撐順勢突破 (sr_breakout_candidate_oos_trades.parquet)——在OOS期間
(2021-2023) 是否低相關，組成投資組合是否有分散風險的價值。

方法：
1. VWAP是當沖策略（同一天進出），SR突破是波段策略（最長持有20個交易日），
   兩者頻率、持有期完全不同，不能直接逐筆配對比較，改成：
   a) 用「已實現損益歸屬到出場日」的方式，把兩個策略都轉成同一條日期軸上
      的每日淨損益序列（含未交易日=0），再做月度加總後算相關係數
      （逐日序列大多是0，月度加總更能反映策略之間是否會同增同減）。
   b) 檢查SR突破部位持有期間，VWAP策略當天進場方向是否與SR突破方向一致
      （同向=風險集中，反向或不重疊=分散）。
   c) 比較兩策略分別的OOS t值 vs 等權重組合後的OOS t值，看合併後統計顯著性
      是否提升（分散化的實際效果指標，不是只看相關係數高低）。

用法：
    python scripts/candidate_correlation_check.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tw_quant.hammer_signal_backtest import COST_SCENARIOS, apply_costs  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
MID_COST = COST_SCENARIOS[1]


def tstat(x: pd.Series) -> float:
    n = len(x)
    if n < 2 or x.std(ddof=1) == 0:
        return np.nan
    return x.mean() / (x.std(ddof=1) / np.sqrt(n))


def main() -> None:
    vwap = pd.read_parquet(DATA_DIR / "trend_day_relaxed_oos_trades.parquet")
    sr = pd.read_parquet(DATA_DIR / "sr_breakout_candidate_oos_trades.parquet")

    vwap = apply_costs(vwap, MID_COST)
    sr = apply_costs(sr, MID_COST)

    vwap["date"] = pd.to_datetime(vwap["trading_date"])
    sr["entry_date"] = pd.to_datetime(sr["entry_date"])
    sr["exit_date"] = pd.to_datetime(sr["exit_date"])

    print("=" * 70)
    print("1) 兩策略各自OOS概況（中成本）")
    print("=" * 70)
    print(f"VWAP趨勢日：n={len(vwap)}, t={tstat(vwap['pnl_points']):.3f}, "
          f"net_sum={vwap['net_twd'].sum():,.0f}, 期間 {vwap['date'].min().date()}~{vwap['date'].max().date()}")
    print(f"SR順勢突破：n={len(sr)}, t={tstat(sr['pnl_points']):.3f}, "
          f"net_sum={sr['net_twd'].sum():,.0f}, 期間 {sr['entry_date'].min().date()}~{sr['exit_date'].max().date()}")

    # ---------- 2a) 日期軸上的每日淨損益序列 -> 月度相關 ----------
    print("\n" + "=" * 70)
    print("2) 月度淨損益相關係數（已實現損益歸屬到出場日）")
    print("=" * 70)
    full_range = pd.date_range("2021-01-01", "2023-12-31", freq="D")
    vwap_daily = pd.Series(0.0, index=full_range)
    for _, r in vwap.iterrows():
        vwap_daily[r["date"]] += r["net_twd"]
    sr_daily = pd.Series(0.0, index=full_range)
    for _, r in sr.iterrows():
        sr_daily[r["exit_date"]] += r["net_twd"]

    vwap_m = vwap_daily.resample("ME").sum()
    sr_m = sr_daily.resample("ME").sum()
    combined_m = pd.DataFrame({"vwap": vwap_m, "sr_breakout": sr_m})
    corr = combined_m["vwap"].corr(combined_m["sr_breakout"])
    print(f"月度淨損益相關係數: {corr:.3f}")
    n_months_both_nonzero = ((combined_m["vwap"] != 0) & (combined_m["sr_breakout"] != 0)).sum()
    print(f"兩策略同月都有損益發生的月數: {n_months_both_nonzero} / {len(combined_m)}")

    # ---------- 2b) SR突破持倉期間，VWAP同期表現/方向一致性 ----------
    print("\n" + "=" * 70)
    print("3) SR突破持倉期間，VWAP策略同期表現與方向一致性")
    print("=" * 70)
    overlap_rows = []
    for _, r in sr.iterrows():
        mask = (vwap["date"] >= r["entry_date"]) & (vwap["date"] <= r["exit_date"])
        overlapping_vwap = vwap[mask]
        same_dir = (overlapping_vwap["direction"] == r["direction"]).sum()
        opp_dir = (overlapping_vwap["direction"] != r["direction"]).sum()
        overlap_rows.append(dict(
            sr_direction=r["direction"], sr_entry=r["entry_date"].date(), sr_exit=r["exit_date"].date(),
            sr_net=r["net_twd"], n_vwap_trades_during_hold=len(overlapping_vwap),
            vwap_same_dir=same_dir, vwap_opp_dir=opp_dir,
            vwap_net_during_hold=overlapping_vwap["net_twd"].sum(),
        ))
    overlap_df = pd.DataFrame(overlap_rows)
    pd.set_option("display.width", 200)
    print(overlap_df.to_string(index=False))
    print(f"\n總計：SR突破44筆交易的持倉期間內，VWAP策略共發生 {overlap_df['n_vwap_trades_during_hold'].sum()} 筆交易"
          f"（同向 {overlap_df['vwap_same_dir'].sum()} / 反向 {overlap_df['vwap_opp_dir'].sum()}）")

    # ---------- 2c) 等權重組合 vs 個別策略的統計顯著性 ----------
    print("\n" + "=" * 70)
    print("4) 等權重組合（1口VWAP + 1口SR突破）vs 個別策略：月度報酬t值比較")
    print("=" * 70)
    combined_m["combined"] = combined_m["vwap"] + combined_m["sr_breakout"]
    print(combined_m)
    print(f"\nVWAP單獨：月度t={tstat(vwap_m):.3f}, 月度std={vwap_m.std():.0f}")
    print(f"SR突破單獨：月度t={tstat(sr_m):.3f}, 月度std={sr_m.std():.0f}")
    print(f"等權重組合：月度t={tstat(combined_m['combined']):.3f}, 月度std={combined_m['combined'].std():.0f}")
    vol_reduction = 1 - combined_m["combined"].std() / (vwap_m.std() + sr_m.std())
    print(f"組合波動 vs 兩者波動直接相加的縮減比例: {vol_reduction:.1%}")


if __name__ == "__main__":
    main()
