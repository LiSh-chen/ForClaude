"""OOS驗證：put/call未平倉量比率(pc_oi_ratio_z)高分位 -> 接下來10天近月
期貨報酬看漲。這是這次會話「選擇權/期貨/現貨關聯性」探索裡目前最一致的
候選（見scripts/scan_putcall_ratio_signal.py：10個不重疊offset全部同向、
4個子區間全部同向）。

驗證設計：
1. 主要驗證：OOS(2021-2023)，跟這次會話一貫的IS(2001-2020)/OOS(2021-2023)
   切分一致，只跑一次、不管結果如何都不回頭調整。z分數的滾動窗格(60日)
   本身用shift(1)，不需要用IS資料重新估計任何門檔——這個候選沒有像
   sr_breakout量能濾網那樣需要從IS資料估計的固定參數，z分數是純粹的
   滾動統計量，OOS期間自然延續使用。
2. 補充驗證：2024-2026這段全新資料（這次會話任何候選都還沒真正驗證過
   這段，只有calendar spread candidate因為樣本不足而不能下定論）——
   N=10天，662個交易日夠切出約66個不重疊區塊，檢定能力比calendar spread
   的N=20好。

兩種驗證都同時報告重疊窗格版本(較容易解讀但t值可能虛高)跟不重疊抽樣
校正版本(較保守但更誠實)。

用法：
    python scripts/validate_putcall_ratio_oos.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
TXO_PATH = REPO_ROOT / "data" / "taifex_txo_daily_putcall.parquet"
TX_PATH = REPO_ROOT / "data" / "taifex_tx_multi_contract.parquet"

IS_CUTOFF = "2021-01-01"
OOS_END = "2024-01-01"
FRESH_START = "2024-01-01"
W = 60
N = 10


def tstat(x: pd.Series) -> tuple[float, int, float]:
    x = x.dropna()
    n = len(x)
    if n < 2 or x.std(ddof=1) == 0:
        return np.nan, n, np.nan
    return x.mean() / (x.std(ddof=1) / np.sqrt(n)), n, x.mean()


def main() -> None:
    txo = pd.read_parquet(TXO_PATH)
    tx = pd.read_parquet(TX_PATH)[["date", "near_price"]]
    df = txo.merge(tx, on="date", how="inner").sort_values("date").reset_index(drop=True)

    df["pc_oi_ratio"] = df["put_oi"] / df["call_oi"]
    roll_mean = df["pc_oi_ratio"].rolling(W).mean().shift(1)
    roll_std = df["pc_oi_ratio"].rolling(W).std(ddof=1).shift(1)
    df["z"] = (df["pc_oi_ratio"].shift(1) - roll_mean) / roll_std
    df["fwd_ret"] = (df["near_price"].shift(-N) / df["near_price"] - 1) * 100

    print("=" * 70)
    print("1) 主要OOS驗證：2021-2023（跟這次會話一貫的IS/OOS切分一致，單次驗證）")
    print("=" * 70)
    oos = df[(df["date"] >= IS_CUTOFF) & (df["date"] < OOS_END)].reset_index(drop=True)
    print(f"OOS交易日數: {len(oos)}")

    high = oos[oos["z"] >= 1.0]["fwd_ret"]
    low = oos[oos["z"] <= -1.0]["fwd_ret"]
    t_h, n_h, m_h = tstat(high)
    t_l, n_l, m_l = tstat(low)
    print(f"\n[重疊窗格版本]")
    print(f"  高分位(z>=1，候選本身): n={n_h} mean={m_h:.3f}% t={t_h:.2f}")
    print(f"  低分位(z<=-1，對照): n={n_l} mean={m_l:.3f}% t={t_l:.2f}")

    print(f"\n[不重疊抽樣校正]（每{N}個交易日取1筆，{N}種offset）:")
    ts = []
    for offset in range(N):
        sub = oos.iloc[offset::N]
        h = sub[sub["z"] >= 1.0]["fwd_ret"]
        t, n, m = tstat(h)
        ts.append(t)
        print(f"  offset={offset}: n={n} mean={m if not np.isnan(m) else float('nan'):.3f}% t={t:.2f}")
    valid_ts = [t for t in ts if not np.isnan(t)]
    pos_count = sum(1 for t in valid_ts if t > 0) if valid_ts else 0
    print(f"\n  {len(valid_ts)}個有效offset中，{pos_count}個方向為正")

    print("\n" + "=" * 70)
    print("2) 補充驗證：2024-2026全新資料（此候選首次接觸這段期間）")
    print("=" * 70)
    fresh = df[df["date"] >= FRESH_START].reset_index(drop=True)
    print(f"交易日數: {len(fresh)} ({fresh['date'].min()} ~ {fresh['date'].max()})")

    high_f = fresh[fresh["z"] >= 1.0]["fwd_ret"]
    t_hf, n_hf, m_hf = tstat(high_f)
    print(f"\n[重疊窗格版本] 高分位(z>=1): n={n_hf} mean={m_hf:.3f}% t={t_hf:.2f}")

    print(f"\n[不重疊抽樣校正]（每{N}個交易日取1筆，{N}種offset）:")
    ts_f = []
    for offset in range(N):
        sub = fresh.iloc[offset::N]
        h = sub[sub["z"] >= 1.0]["fwd_ret"]
        t, n, m = tstat(h)
        ts_f.append(t)
        print(f"  offset={offset}: n={n} mean={m if not np.isnan(m) else float('nan'):.3f}% t={t:.2f}")
    valid_ts_f = [t for t in ts_f if not np.isnan(t)]
    pos_count_f = sum(1 for t in valid_ts_f if t > 0) if valid_ts_f else 0
    print(f"\n  {len(valid_ts_f)}個有效offset中，{pos_count_f}個方向為正")

    print("\n" + "=" * 70)
    print("完成。這是這個候選唯一一次OOS驗證，不論結果如何都不回頭調整。")
    print("=" * 70)


if __name__ == "__main__":
    main()
