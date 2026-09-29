"""解決「8%百分比停損 vs 固定保證金緩衝(815點)」的結構性錯位問題。

背景（上一步發現）：MTX原始保證金175,250元、維持保證金134,500元，換算
緩衝 = (175,250-134,500)/50 = 815點。這是「現在」的保證金金額（TAIFEX
會隨時間調整保證金，這裡沒有逐年重建歷史保證金表，只能用現在的數字
去檢視「如果現在／未來遇到跟過去相似的低波動期間，這個停損規則還撐得住嗎」）。

過去27段低波動週期用回測時，8%停損換算成點數，隨著大盤點位從2002年
約6,000點漲到2025年約29,000點，點數距離從483點(0.59x緩衝)一路長大到
2,321點(2.85x緩衝)——代表照現在的保證金水準，會先斷頭才等到停損動作，
停損規則形同虛設。

修正方案（三個都測，跟原始8%百分比停損比較）：
1. 純點數停損：固定N點，不管價位多高都一樣（N分別測400/500/600/700點，
   對應現在815點緩衝的49%/61%/74%/86%）
2. 混合式：百分比停損但用點數上限封頂 —— min(8%*峰值價格換算點數, 點數上限)
   這樣低價位時沿用原本8%的行為，高價位時自動收斂到安全的點數距離
3. 保證金比例停損：直接定義成「緩衝的X%」，等於方案1但用比例表示，
   本質相同、只是參數化方式不同，這裡不重複列，用方案1代表

用法：
    python scripts/test_margin_aware_stoploss.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
TX_PATH = REPO_ROOT / "data" / "taifex_tx_multi_contract.parquet"
REGIME_PATH = REPO_ROOT / "data" / "regime_classification_daily.parquet"
POINT_VALUE = 50.0
MARGIN_BUFFER_POINTS = 815.0

POINT_STOPS = [400, 500, 600, 700]
HYBRID_CAPS = [500, 600, 700]  # 8%百分比停損但用點數上限封頂
PCT_STOP = 0.08


def find_low_vol_episodes(regime: pd.Series, dates: pd.Series) -> pd.DataFrame:
    valid = regime.notna()
    r = (regime[valid] == 0.0)
    d = dates[valid]
    group_id = r.ne(r.shift()).cumsum()
    tmp = pd.DataFrame({"date": d.values, "is_low_vol": r.values, "group": group_id.values})
    episodes = []
    for gid, sub in tmp.groupby("group"):
        if sub["is_low_vol"].iloc[0]:
            episodes.append(dict(start=sub["date"].min(), end=sub["date"].max(), n_days=len(sub)))
    return pd.DataFrame(episodes)


def simulate(df: pd.DataFrame, episodes: pd.DataFrame, stop_type: str, param: float | None) -> pd.DataFrame:
    """stop_type: 'pct' | 'points' | 'hybrid' | 'none'
    param: pct用小數(0.08), points用點數(600), hybrid用點數上限(600)"""
    daily_pnl = []
    max_ratio_seen = 0.0
    for ep in episodes.itertuples():
        ep_df = df[(df["date"] >= ep.start) & (df["date"] <= ep.end)].reset_index(drop=True)
        if len(ep_df) < 2:
            continue
        entry_price = ep_df["near_price"].iloc[0]
        peak = entry_price
        stopped = False

        if stop_type == "points":
            stop_dist = param
        elif stop_type == "pct":
            stop_dist = entry_price * param  # 用進場價估計，跟原始腳本一致（用peak也差不多）
        elif stop_type == "hybrid":
            stop_dist = min(entry_price * PCT_STOP, param)
        else:
            stop_dist = None

        if stop_dist is not None:
            max_ratio_seen = max(max_ratio_seen, stop_dist / MARGIN_BUFFER_POINTS)

        for i in range(1, len(ep_df)):
            prev_price = ep_df["near_price"].iloc[i - 1]
            price = ep_df["near_price"].iloc[i]
            if not stopped:
                pnl = (price - prev_price) * POINT_VALUE
                peak = max(peak, price)
                if stop_type == "hybrid":
                    stop_dist = min(peak * PCT_STOP, param)
                    max_ratio_seen = max(max_ratio_seen, stop_dist / MARGIN_BUFFER_POINTS)
                elif stop_type == "pct":
                    stop_dist = peak * param
                    max_ratio_seen = max(max_ratio_seen, stop_dist / MARGIN_BUFFER_POINTS)
                if stop_dist is not None and (peak - price) >= stop_dist:
                    stopped = True
            else:
                pnl = 0.0
            daily_pnl.append(dict(date=ep_df["date"].iloc[i], pnl=pnl))
    result = pd.DataFrame(daily_pnl)
    return result, max_ratio_seen


def main() -> None:
    tx = pd.read_parquet(TX_PATH)[["date", "near_price"]].sort_values("date").reset_index(drop=True)
    regime = pd.read_parquet(REGIME_PATH)[["date", "high_vol_regime"]]
    df = tx.merge(regime, on="date", how="left")

    episodes = find_low_vol_episodes(df["high_vol_regime"], df["date"])
    print(f"低波動週期數: {len(episodes)}")
    print(f"保證金緩衝: {MARGIN_BUFFER_POINTS:.0f}點\n")

    print("=" * 78)
    print("各停損設計比較（全樣本2001-2026）")
    print("=" * 78)
    rows = []

    sim_none, _ = simulate(df, episodes, "none", None)
    cum = sim_none.set_index("date")["pnl"].cumsum()
    total_none = sim_none["pnl"].sum()
    dd_none = (cum - cum.cummax()).min()
    rows.append(("無停損(對照)", total_none, dd_none, np.nan))
    print(f"  無停損(對照): 總損益={total_none:,.0f} 最大回撤={dd_none:,.0f} 停損最大點距/緩衝比=N/A")

    sim_pct, ratio_pct = simulate(df, episodes, "pct", PCT_STOP)
    cum = sim_pct.set_index("date")["pnl"].cumsum()
    total_pct = sim_pct["pnl"].sum()
    dd_pct = (cum - cum.cummax()).min()
    rows.append(("8%百分比停損(原方案)", total_pct, dd_pct, ratio_pct))
    print(f"  8%百分比停損(原方案): 總損益={total_pct:,.0f} 最大回撤={dd_pct:,.0f} 停損最大點距/緩衝比={ratio_pct:.2f}x")

    print("\n  -- 純點數停損 --")
    for pts in POINT_STOPS:
        sim, ratio = simulate(df, episodes, "points", pts)
        cum = sim.set_index("date")["pnl"].cumsum()
        total = sim["pnl"].sum()
        dd = (cum - cum.cummax()).min()
        rows.append((f"{pts}點固定停損", total, dd, ratio))
        print(f"  {pts}點固定停損: 總損益={total:,.0f} 最大回撤={dd:,.0f} 點距/緩衝比={ratio:.2f}x(恆定)")

    print("\n  -- 混合式(8%百分比,點數封頂) --")
    for cap in HYBRID_CAPS:
        sim, ratio = simulate(df, episodes, "hybrid", cap)
        cum = sim.set_index("date")["pnl"].cumsum()
        total = sim["pnl"].sum()
        dd = (cum - cum.cummax()).min()
        rows.append((f"8%百分比+{cap}點封頂", total, dd, ratio))
        print(f"  8%百分比+{cap}點封頂: 總損益={total:,.0f} 最大回撤={dd:,.0f} 實際最大點距/緩衝比={ratio:.2f}x")

    print("\n" + "=" * 78)
    print("相對無停損總報酬保留比例")
    print("=" * 78)
    for label, total, dd, ratio in rows:
        pct_of_none = total / total_none * 100 if total_none != 0 else np.nan
        print(f"  {label}: {pct_of_none:.1f}% (差額{total-total_none:,.0f})")

    print("\n" + "=" * 78)
    print("2025年最近兩段低波動週期：用600點封頂 vs 原8%百分比 分別觸發狀況")
    print("=" * 78)
    recent_eps = episodes[episodes["start"] >= "2024-01-01"]
    for stop_type, param, label in [("pct", PCT_STOP, "8%百分比"), ("hybrid", 600, "8%+600點封頂")]:
        for ep in recent_eps.itertuples():
            ep_df = df[(df["date"] >= ep.start) & (df["date"] <= ep.end)].reset_index(drop=True)
            entry_price = ep_df["near_price"].iloc[0]
            if stop_type == "pct":
                dist = entry_price * PCT_STOP
            else:
                dist = min(entry_price * PCT_STOP, param)
            print(f"  {label} | 進場{ep.start.date()} 進場價{entry_price:.0f} 停損點距{dist:.0f}點 "
                  f"對緩衝比={dist/MARGIN_BUFFER_POINTS:.2f}x")


if __name__ == "__main__":
    main()
