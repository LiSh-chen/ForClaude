"""給低波動做多部分加停損：解決2008/2022那種「體制濾網只能事後確認、
不能提前預警崩盤」的問題。

做法：在每一段低波動體制週期內，追蹤「進場後至今的最高收盤價」，一旦
收盤價從最高點回落超過停損百分比，當天就出場、直到這段低波動週期
結束前都維持空手（不會在同一段週期內重新進場——如果體制還沒真的轉換，
重新進場等於還在賭同一個崩盤中繼)。等下一段低波動週期開始才重新啟動。

停損百分比用敏感度掃描（不是挑一個看起來最好的），並且特別檢視
2008年、2022年這兩個原本受傷最重的年份改善了多少，以及总報酬/最大回撤
的取捨。

用法：
    python scripts/test_low_vol_stoploss.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
TX_PATH = REPO_ROOT / "data" / "taifex_tx_multi_contract.parquet"
REGIME_PATH = REPO_ROOT / "data" / "regime_classification_daily.parquet"
POINT_VALUE = 50.0

STOP_PCTS = [0.02, 0.03, 0.05, 0.08, 0.10, None]  # None = 無停損(對照)


def find_low_vol_episodes(regime: pd.Series, dates: pd.Series) -> pd.DataFrame:
    valid = regime.notna()
    r = (regime[valid] == 0.0)  # True=低波動
    d = dates[valid]
    group_id = r.ne(r.shift()).cumsum()
    tmp = pd.DataFrame({"date": d.values, "is_low_vol": r.values, "group": group_id.values})
    episodes = []
    for gid, sub in tmp.groupby("group"):
        if sub["is_low_vol"].iloc[0]:
            episodes.append(dict(start=sub["date"].min(), end=sub["date"].max(), n_days=len(sub)))
    return pd.DataFrame(episodes)


def simulate_with_stop(df: pd.DataFrame, episodes: pd.DataFrame, stop_pct: float | None) -> pd.DataFrame:
    daily_pnl = []
    for ep in episodes.itertuples():
        ep_df = df[(df["date"] >= ep.start) & (df["date"] <= ep.end)].reset_index(drop=True)
        if len(ep_df) < 2:
            continue
        peak = ep_df["near_price"].iloc[0]
        stopped = False
        for i in range(1, len(ep_df)):
            prev_price = ep_df["near_price"].iloc[i - 1]
            price = ep_df["near_price"].iloc[i]
            if not stopped:
                pnl = (price - prev_price) * POINT_VALUE
                peak = max(peak, price)
                if stop_pct is not None and price <= peak * (1 - stop_pct):
                    stopped = True
            else:
                pnl = 0.0
            daily_pnl.append(dict(date=ep_df["date"].iloc[i], pnl=pnl))
    return pd.DataFrame(daily_pnl)


def main() -> None:
    tx = pd.read_parquet(TX_PATH)[["date", "near_price"]].sort_values("date").reset_index(drop=True)
    regime = pd.read_parquet(REGIME_PATH)[["date", "high_vol_regime"]]
    df = tx.merge(regime, on="date", how="left")

    episodes = find_low_vol_episodes(df["high_vol_regime"], df["date"])
    print(f"低波動週期數: {len(episodes)}\n")

    print("=" * 70)
    print("停損百分比敏感度掃描")
    print("=" * 70)
    results = {}
    for stop_pct in STOP_PCTS:
        sim = simulate_with_stop(df, episodes, stop_pct)
        sim = sim.set_index("date").sort_index()
        cum = sim["pnl"].cumsum()
        dd = (cum - cum.cummax()).min()
        total = sim["pnl"].sum()
        label = f"{stop_pct*100:.0f}%停損" if stop_pct is not None else "無停損(對照)"
        print(f"  {label}: 總損益={total:,.0f} 最大回撤={dd:,.0f}")
        results[label] = sim

    print("\n" + "=" * 70)
    print("2008年、2022年逐年比較（原本最糟的兩年）")
    print("=" * 70)
    for label, sim in results.items():
        sim2 = sim.copy()
        sim2["year"] = sim2.index.year
        yearly = sim2.groupby("year")["pnl"].sum()
        y2008 = yearly.get(2008, 0)
        y2022 = yearly.get(2022, 0)
        print(f"  {label}: 2008={y2008:,.0f}  2022={y2022:,.0f}")

    print("\n" + "=" * 70)
    print("選定 5% 停損版本：逐年損益 + IS/OOS 拆解")
    print("=" * 70)
    sim5 = simulate_with_stop(df, episodes, 0.05).set_index("date").sort_index()
    sim5["year"] = sim5.index.year
    print(sim5.groupby("year")["pnl"].sum().to_string())

    is_total = sim5[sim5.index < "2021-01-01"]["pnl"].sum()
    oos_total = sim5[(sim5.index >= "2021-01-01") & (sim5.index < "2024-01-01")]["pnl"].sum()
    print(f"\nIS(2001-2020)總損益: {is_total:,.0f}")
    print(f"OOS(2021-2023)總損益: {oos_total:,.0f}")


if __name__ == "__main__":
    main()
