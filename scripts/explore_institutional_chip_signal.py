"""探索性檢視：三大法人期貨未平倉籌碼（外資及陸資/投信/自營商）跟次日
台指期報酬的關係。**明確標記為探索性，不是已驗證候選**——資料只有
2023-10-02~2026-09-24(725個交易日，約3年)，遠不足以比照這次會話一貫的
IS(2001-2020)/OOS(2021-2023)驗證紀律，這裡只做初步觀察，不會宣稱「發現
可用策略」。

測兩個最常見的籌碼假設：
1. 淨未平倉水位本身（外資淨多/淨空）跟次日報酬的關係——「外資看多/看空」
2. 淨未平倉的日變動量（今天比昨天加碼多單還是空單）跟次日報酬的關係——
   「外資今天在做什麼」，這個在台灣期貨社群討論度更高

用法：
    python scripts/explore_institutional_chip_signal.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
INST_PATH = REPO_ROOT / "data" / "taifex_txf_institutional_daily.parquet"
TX_PATH = REPO_ROOT / "data" / "taifex_tx_multi_contract.parquet"


def tstat(x: pd.Series) -> tuple[float, int, float]:
    x = x.dropna()
    n = len(x)
    if n < 2 or x.std(ddof=1) == 0:
        return np.nan, n, np.nan
    return x.mean() / (x.std(ddof=1) / np.sqrt(n)), n, x.mean()


def main() -> None:
    inst = pd.read_parquet(INST_PATH)
    tx = pd.read_parquet(TX_PATH)[["date", "near_price"]].sort_values("date")

    wide = inst.pivot(index="date", columns="identity", values="net_oi").reset_index()
    wide.columns = ["date", "foreign_net_oi", "trust_net_oi", "dealer_net_oi"]

    df = wide.merge(tx, on="date", how="inner").sort_values("date").reset_index(drop=True)
    df["ret_next"] = (df["near_price"].shift(-1) / df["near_price"] - 1) * 100
    for col in ["foreign_net_oi", "trust_net_oi", "dealer_net_oi"]:
        df[f"{col}_chg"] = df[col].diff()

    print(f"樣本範圍: {df['date'].min().date()} ~ {df['date'].max().date()}，n={len(df)}")
    print("\n" + "=" * 70)
    print("1) 淨未平倉水位（本身）vs 次日報酬（z-score高分位 vs 低分位）")
    print("=" * 70)
    for col in ["foreign_net_oi", "trust_net_oi", "dealer_net_oi"]:
        z = (df[col] - df[col].rolling(60).mean().shift(1)) / df[col].rolling(60).std(ddof=1).shift(1)
        high = df.loc[z >= 1.0, "ret_next"]
        low = df.loc[z <= -1.0, "ret_next"]
        t_h, n_h, m_h = tstat(high)
        t_l, n_l, m_l = tstat(low)
        print(f"  {col}: z>=1(偏多) n={n_h} mean_next_ret={m_h:.3f}% t={t_h:.2f} | "
              f"z<=-1(偏空) n={n_l} mean_next_ret={m_l:.3f}% t={t_l:.2f}")

    print("\n" + "=" * 70)
    print("2) 淨未平倉日變動量 vs 次日報酬（加碼多單/空單）")
    print("=" * 70)
    for col in ["foreign_net_oi_chg", "trust_net_oi_chg", "dealer_net_oi_chg"]:
        add_long = df.loc[df[col] > 0, "ret_next"]
        add_short = df.loc[df[col] < 0, "ret_next"]
        t_l, n_l, m_l = tstat(add_long)
        t_s, n_s, m_s = tstat(add_short)
        corr = df[col].corr(df["ret_next"])
        print(f"  {col}: 加碼多單 n={n_l} mean={m_l:.3f}% t={t_l:.2f} | "
              f"加碼空單 n={n_s} mean={m_s:.3f}% t={t_s:.2f} | corr={corr:.3f}")

    print("\n" + "=" * 70)
    print("3) 前後各半期穩健性檢查（資料太短，只能切兩半，不是完整子區間）")
    print("=" * 70)
    mid = df["date"].quantile(0.5)
    for half_name, half_df in [("前半", df[df["date"] < mid]), ("後半", df[df["date"] >= mid])]:
        col = "foreign_net_oi_chg"
        add_long = half_df.loc[half_df[col] > 0, "ret_next"]
        add_short = half_df.loc[half_df[col] < 0, "ret_next"]
        t_l, n_l, m_l = tstat(add_long)
        t_s, n_s, m_s = tstat(add_short)
        print(f"  外資加碼多單({half_name}): n={n_l} mean={m_l:.3f}% t={t_l:.2f} | "
              f"外資加碼空單({half_name}): n={n_s} mean={m_s:.3f}% t={t_s:.2f}")


if __name__ == "__main__":
    main()
