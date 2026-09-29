"""測試低波動期間的兩個互補方案：
1. 簡單做多（買進持有）：低波動期間近月期貨收盤對收盤報酬
2. 賣選擇權勒式收權利金（用IS/子區間/OOS同一套紀律）

方案1不需要新資料，直接用已有的近月期貨收盤價+體制標籤。

用法：
    python scripts/test_low_vol_complements.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
TX_PATH = REPO_ROOT / "data" / "taifex_tx_multi_contract.parquet"
REGIME_PATH = REPO_ROOT / "data" / "regime_classification_daily.parquet"

SUB_PERIODS = [
    ("2001-2005", "2001-01-01", "2006-01-01"),
    ("2006-2010", "2006-01-01", "2011-01-01"),
    ("2011-2015", "2011-01-01", "2016-01-01"),
    ("2016-2020", "2016-01-01", "2021-01-01"),
]


def tstat(x: pd.Series) -> tuple[float, int, float]:
    x = x.dropna()
    n = len(x)
    if n < 2 or x.std(ddof=1) == 0:
        return np.nan, n, np.nan
    return x.mean() / (x.std(ddof=1) / np.sqrt(n)), n, x.mean()


def main() -> None:
    tx = pd.read_parquet(TX_PATH)[["date", "near_price"]].sort_values("date").reset_index(drop=True)
    regime = pd.read_parquet(REGIME_PATH)[["date", "bull_regime", "high_vol_regime"]]
    df = tx.merge(regime, on="date", how="left")
    df["ret_1d"] = df["near_price"].pct_change() * 100

    print("=" * 70)
    print("方案1：低波動期間做多（買進持有），收盤對收盤日報酬")
    print("=" * 70)

    is_df = df[df["date"] < "2021-01-01"]
    print("\nIS(2001-2020)整體：")
    low_vol = is_df[is_df["high_vol_regime"] == 0.0]["ret_1d"]
    high_vol = is_df[is_df["high_vol_regime"] == 1.0]["ret_1d"]
    all_ret = is_df["ret_1d"]
    t_l, n_l, m_l = tstat(low_vol)
    t_h, n_h, m_h = tstat(high_vol)
    t_a, n_a, m_a = tstat(all_ret)
    print(f"  低波動做多: n={n_l} mean={m_l:.4f}%/天 t={t_l:.2f} 年化約{m_l*252:.1f}%")
    print(f"  高波動做多(對照): n={n_h} mean={m_h:.4f}%/天 t={t_h:.2f} 年化約{m_h*252:.1f}%")
    print(f"  全樣本做多(對照): n={n_a} mean={m_a:.4f}%/天 t={t_a:.2f} 年化約{m_a*252:.1f}%")

    print("\n子區間穩健性（低波動做多）：")
    for name, start, end in SUB_PERIODS:
        sub = df[(df["date"] >= start) & (df["date"] < end) & (df["high_vol_regime"] == 0.0)]
        t, n, m = tstat(sub["ret_1d"])
        print(f"  {name}: n={n} mean={m:.4f}%/天 t={t:.2f} 年化約{m*252:.1f}%")

    print("\nOOS(2021-2023)：")
    oos = df[(df["date"] >= "2021-01-01") & (df["date"] < "2024-01-01") & (df["high_vol_regime"] == 0.0)]
    t, n, m = tstat(oos["ret_1d"])
    print(f"  低波動做多: n={n} mean={m:.4f}%/天 t={t:.2f} 年化約{m*252:.1f}%")

    print("\n2024-2026補充窗格：")
    fresh = df[(df["date"] >= "2024-01-01") & (df["high_vol_regime"] == 0.0)]
    t, n, m = tstat(fresh["ret_1d"])
    print(f"  低波動做多: n={n} mean={m:.4f}%/天 t={t:.2f} 年化約{m*252:.1f}%")

    print("\n" + "=" * 70)
    print("互補性檢查：低波動做多 vs 三腿策略（高波動限定版）合併起來的覆蓋率")
    print("=" * 70)
    total_days = df["high_vol_regime"].notna().sum()
    low_days = (df["high_vol_regime"] == 0.0).sum()
    high_days = (df["high_vol_regime"] == 1.0).sum()
    print(f"總交易日數: {total_days}，低波動天數: {low_days}({low_days/total_days*100:.1f}%)，"
          f"高波動天數: {high_days}({high_days/total_days*100:.1f}%)")
    print("兩個策略合併後理論上每天都有部位（低波動做多期貨、高波動跑三腿），"
          "覆蓋率100%——但要注意這不是「兩個報酬直接相加」，持倉方向/風險特性完全不同，"
          "只是時間軸上互斥切換。")


if __name__ == "__main__":
    main()
