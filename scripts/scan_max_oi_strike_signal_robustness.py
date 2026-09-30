"""max_oi_strike訊號初步IS掃描的後續穩健性檢查——跟這次會話一貫的流程
一樣：初步掃描出來的效果都偏弱、邊緣，先做子區間穩健性+不重疊抽樣校正，
再決定值不值得花掉單次OOS驗證機會。

四個訊號裡，初步掃描(scan_max_oi_strike_signal.py)的結果：
- 壓力牆(max_call_oi_wall)：方向大致對，但相關係數極小(<0.06)，t值多半<2
- 支撐牆(max_put_oi_wall)：方向大致對，但N=5那組方向反了，相關係數也很小
- max pain磁吸(全樣本)：四個N(3/5/10/20)都通過|t|>=2(above組)，方向一致
  ，是四個裡最有戲的，這裡重點做穩健性檢查
- max pain磁吸(結算前5日/pin risk)：樣本小(n=216/332)、方向不一致，
  初步看起來站不住腳，這裡不繼續深究

用法：
    python scripts/scan_max_oi_strike_signal_robustness.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
MAXOI_PATH = REPO_ROOT / "data" / "taifex_txo_max_oi_strike.parquet"
TX_PATH = REPO_ROOT / "data" / "taifex_tx_multi_contract.parquet"

IS_CUTOFF = "2021-01-01"
N_FOR_ROBUSTNESS = [5, 10]


def tstat(x: pd.Series) -> tuple[float, int, float]:
    x = x.dropna()
    n = len(x)
    if n < 2 or x.std(ddof=1) == 0:
        return np.nan, n, np.nan
    return x.mean() / (x.std(ddof=1) / np.sqrt(n)), n, x.mean()


def main() -> None:
    maxoi = pd.read_parquet(MAXOI_PATH)
    tx = pd.read_parquet(TX_PATH)[["date", "near_price"]]
    df = maxoi.merge(tx, on="date", how="inner").sort_values("date").reset_index(drop=True)

    price_lag = df["near_price"].shift(1)
    df["dist_pain_pct"] = (df["max_pain_strike"].shift(1) - price_lag) / price_lag * 100
    df["dist_put_pct"] = (price_lag - df["max_put_oi_strike"].shift(1)) / price_lag * 100

    is_mask = df["date"] < IS_CUTOFF

    print("=" * 70)
    print("1) max pain磁吸：子區間穩健性 (above vs below, N=5天)")
    print("=" * 70)
    N = 5
    fwd_ret = (df["near_price"].shift(-N) / df["near_price"] - 1) * 100
    periods = [("2001-2005", "2001-01-01", "2006-01-01"), ("2006-2010", "2006-01-01", "2011-01-01"),
               ("2011-2015", "2011-01-01", "2016-01-01"), ("2016-2020", "2016-01-01", "2021-01-01")]
    for label, start, end in periods:
        mask = (df["date"] >= start) & (df["date"] < end) & df["dist_pain_pct"].notna() & fwd_ret.notna()
        above = fwd_ret[mask & (df["dist_pain_pct"] >= 1.0)]
        below = fwd_ret[mask & (df["dist_pain_pct"] <= -1.0)]
        t_a, n_a, m_a = tstat(above)
        t_b, n_b, m_b = tstat(below)
        print(f"  {label}: above n={n_a} mean={m_a:.3f} t={t_a:.2f}  |  below n={n_b} mean={m_b:.3f} t={t_b:.2f}")

    print()
    print("=" * 70)
    print("2) max pain磁吸：不重疊抽樣校正 (N=5天, 5種offset, above組)")
    print("=" * 70)
    valid_full = is_mask & df["dist_pain_pct"].notna() & fwd_ret.notna()
    sub = df.loc[valid_full, ["dist_pain_pct"]].copy()
    sub["fwd_ret"] = fwd_ret[valid_full]
    sub = sub.reset_index(drop=True)
    offset_results = []
    for offset in range(N):
        idx = sub.index[offset::N]
        s = sub.loc[idx]
        above = s.loc[s["dist_pain_pct"] >= 1.0, "fwd_ret"]
        t_a, n_a, m_a = tstat(above)
        offset_results.append((offset, n_a, m_a, t_a))
        print(f"  offset={offset}: n={n_a} mean={m_a:.3f} t={t_a:.2f}")
    means = [r[2] for r in offset_results if not np.isnan(r[2])]
    pos_count = sum(1 for m in means if m > 0)
    print(f"  {pos_count}/{len(means)} 個offset方向為正 (mean_above > 0)")

    print()
    print("=" * 70)
    print("3) max pain磁吸：不重疊抽樣校正 (N=10天, 10種offset, above組)")
    print("=" * 70)
    N10 = 10
    fwd_ret10 = (df["near_price"].shift(-N10) / df["near_price"] - 1) * 100
    valid10 = is_mask & df["dist_pain_pct"].notna() & fwd_ret10.notna()
    sub10 = df.loc[valid10, ["dist_pain_pct"]].copy()
    sub10["fwd_ret"] = fwd_ret10[valid10]
    sub10 = sub10.reset_index(drop=True)
    offset_results10 = []
    for offset in range(N10):
        idx = sub10.index[offset::N10]
        s = sub10.loc[idx]
        above = s.loc[s["dist_pain_pct"] >= 1.0, "fwd_ret"]
        t_a, n_a, m_a = tstat(above)
        offset_results10.append((offset, n_a, m_a, t_a))
    for r in offset_results10:
        print(f"  offset={r[0]}: n={r[1]} mean={r[2]:.3f} t={r[3]:.2f}")
    means10 = [r[2] for r in offset_results10 if not np.isnan(r[2])]
    pos_count10 = sum(1 for m in means10 if m > 0)
    tvals10 = [r[3] for r in offset_results10 if not np.isnan(r[3])]
    sig_count10 = sum(1 for t in tvals10 if abs(t) >= 2)
    print(f"  {pos_count10}/{len(means10)} 個offset方向為正，{sig_count10}/{len(tvals10)} 達到|t|>=2")

    print()
    print("=" * 70)
    print("4) 支撐牆：子區間穩健性 (near vs far, N=10天，前面初步掃描裡t值最高的一組)")
    print("=" * 70)
    N10b = 10
    fwd_ret10b = (df["near_price"].shift(-N10b) / df["near_price"] - 1) * 100
    for label, start, end in periods:
        mask = ((df["date"] >= start) & (df["date"] < end) & df["dist_put_pct"].notna()
                 & fwd_ret10b.notna() & (df["dist_put_pct"] > 0))
        near = fwd_ret10b[mask & (df["dist_put_pct"] <= 2.0)]
        far = fwd_ret10b[mask & (df["dist_put_pct"] > 2.0)]
        t_n, n_n, m_n = tstat(near)
        t_f, n_f, m_f = tstat(far)
        print(f"  {label}: near n={n_n} mean={m_n:.3f} t={t_n:.2f}  |  far n={n_f} mean={m_f:.3f} t={t_f:.2f}")


if __name__ == "__main__":
    main()
