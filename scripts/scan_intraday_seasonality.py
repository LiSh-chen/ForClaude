"""在 2001-2020（樣本內）資料上，掃描「每天固定 30 分鐘時段」的報酬統計，找出
哪個時段有跨越長時間一致顯著的方向性偏態——這是 lunch_reversal_strategy.py
的訊號來源：不是先假設一個型態，是先看資料在哪裡有穩定的統計規律，再回頭
設計交易規則去捕捉它。

方法：
1. 把每根 1 分鐘K棒依「時鐘時間」（不分日期）分進 30 分鐘桶（00:00, 00:30, ...）。
2. 對每一天／每個連續盤（block_id，避免日盤/夜盤資料混在一起或跨盤），算出
   每個桶的 open->close 報酬跟高低點區間。
3. 對每個桶，把所有天數的報酬拿去做單樣本 t 檢定（H0: 平均報酬 = 0）。
4. 額外把樣本內拆成四個 5 年子區間，檢查顯著的桶是不是「每個子區間都同方向」
   （穩健），還是只有某一段特別強、其他段普通甚至反向（不穩健、可能只是那段
   期間的巧合）。

用法：
    python scripts/scan_intraday_seasonality.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = REPO_ROOT / "data" / "txf_1min.parquet"
OUT_PATH = REPO_ROOT / "data" / "intraday_seasonality_scan.parquet"
SUBPERIOD_OUT_PATH = REPO_ROOT / "data" / "intraday_seasonality_subperiods.parquet"

IS_CUTOFF = "2021-01-01"
MIN_BARS_PER_BLOCK = 20  # 該 30 分鐘桶至少要有這麼多根 1 分鐘K棒才採用（避免區段被切斷太短）


def _bucketize(df: pd.DataFrame) -> pd.DataFrame:
    d = df.sort_values("datetime").reset_index(drop=True).copy()
    gap_min = d["datetime"].diff().dt.total_seconds() / 60
    d["block_id"] = ((gap_min > 2) | gap_min.isna()).cumsum()
    minutes_of_day = d["datetime"].dt.hour * 60 + d["datetime"].dt.minute
    bucket_start = (minutes_of_day // 30) * 30
    d["bucket_label"] = bucket_start.apply(lambda m: f"{m // 60:02d}:{m % 60:02d}")
    d["year"] = d["datetime"].dt.year
    return d


def _block_bucket_returns(d: pd.DataFrame) -> pd.DataFrame:
    agg = d.groupby(["block_id", "bucket_label"]).agg(
        open=("open", "first"), high=("high", "max"), low=("low", "min"), close=("close", "last"),
        n=("close", "size"), year=("year", "first"),
    ).reset_index()
    agg = agg[agg["n"] >= MIN_BARS_PER_BLOCK].copy()
    agg["ret"] = agg["close"] - agg["open"]
    agg["range"] = agg["high"] - agg["low"]
    return agg


def scan(agg: pd.DataFrame) -> pd.DataFrame:
    def _tstat(s: pd.Series) -> float:
        n = len(s)
        se = s.std(ddof=1) / np.sqrt(n)
        return s.mean() / se if se > 0 else np.nan

    summary = agg.groupby("bucket_label").agg(
        n_days=("ret", "size"), mean_ret=("ret", "mean"), std_ret=("ret", "std"),
        win_rate=("ret", lambda s: (s > 0).mean()), mean_range=("range", "mean"),
    )
    summary["tstat"] = agg.groupby("bucket_label")["ret"].apply(_tstat)
    return summary.sort_values("tstat")


def subperiod_check(agg: pd.DataFrame, buckets: list[str]) -> pd.DataFrame:
    rows = []
    for label in buckets:
        sub = agg[agg["bucket_label"] == label]
        for yr_start in [2001, 2006, 2011, 2016]:
            yr_end = yr_start + 4
            s = sub[(sub["year"] >= yr_start) & (sub["year"] <= yr_end)]
            if s.empty:
                continue
            se = s["ret"].std(ddof=1) / np.sqrt(len(s))
            rows.append(dict(
                bucket_label=label, period=f"{yr_start}-{yr_end}", n=len(s),
                mean_ret=s["ret"].mean(), tstat=(s["ret"].mean() / se) if se > 0 else np.nan,
                win_rate=(s["ret"] > 0).mean(),
            ))
    return pd.DataFrame(rows)


def main() -> None:
    df = pd.read_parquet(DATA_PATH)
    is_df = df[df["datetime"] < IS_CUTOFF]
    d = _bucketize(is_df)
    agg = _block_bucket_returns(d)

    summary = scan(agg)
    summary.to_parquet(OUT_PATH)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_rows", 60)
    pd.set_option("display.float_format", lambda x: f"{x:,.3f}")
    print("=== 30 分鐘時段報酬掃描（樣本內 2001-2020，依 t 值排序）===")
    print(summary.to_string())

    # 找出 |t|>=3 的候選桶做子區間穩健性檢查
    candidates = summary[summary["tstat"].abs() >= 3].index.tolist()
    print(f"\n|t|>=3 的候選時段: {candidates}")

    if candidates:
        sub = subperiod_check(agg, candidates)
        sub.to_parquet(SUBPERIOD_OUT_PATH, index=False)
        print("\n=== 候選時段的五年子區間穩健性檢查 ===")
        for label in candidates:
            print(f"\n-- {label} --")
            print(sub[sub["bucket_label"] == label][["period", "n", "mean_ret", "tstat", "win_rate"]].to_string(index=False))


if __name__ == "__main__":
    main()
