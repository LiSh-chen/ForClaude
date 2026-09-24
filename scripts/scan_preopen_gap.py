"""探索「盤前缺口」訊號：夜盤最後一筆交易（約05:00收盤）到日盤開盤
（08:45）之間有一段完全沒有交易的空窗（近3小時45分），這段空窗可能反映
國際盤（美股收盤、亞股早盤）的最新資訊，日盤開盤瞬間的價格跳動（缺口）
可能延續或反轉。

這是一個全新的訊號來源（資訊面缺口），跟這次會話測過的其他東西都不同：
不是價格型態（K棒）、不是技術指標、不是固定時刻套利，是「夜盤停止交易
到日盤開盤這段空窗期間的價格跳動本身」當訊號。

先用樣本內（2001-2020）資料探索：
1. 缺口大小分布（點數/百分比）。
2. 缺口方向 vs 當天日盤收盤的相關性——延續（缺口跟當天日內走勢同向）
   還是回補（反向）？分桶看不同缺口大小的表現差異。
3. 依此決定第二階段要做延續策略還是回補策略，再進一步做完整驗證
   （含真實停損單/滑價模擬）。

用法：
    python scripts/scan_preopen_gap.py
"""

from __future__ import annotations

import sys
from datetime import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = REPO_ROOT / "data" / "txf_1min.parquet"
IS_CUTOFF = "2021-01-01"


def build_gap_table(df: pd.DataFrame) -> pd.DataFrame:
    d = df.sort_values("datetime").copy()
    t = d["datetime"].dt.time
    is_night = (t >= time(15, 0)) | (t <= time(5, 0))
    is_day = (t >= time(8, 45)) & (t <= time(13, 45))

    # 夜盤 session_label：標成「這個夜盤銜接的下一個日盤交易日」，不是夜盤
    # 開始那天——例如 2021-05-11 15:00 ~ 2021-05-12 05:00 這整個夜盤，全部
    # 標成 2021-05-12（因為它是 2021-05-12 日盤開盤前的那一段，不是
    # 2021-05-11 日盤收盤後的事後資訊）。
    # ★ 這裡原本有個嚴重的未來函數 bug：用「當天日期」當 session_date、
    # 只把 t<5:00（不含剛好05:00）的部分往前一天調整，會導致某一個交易日
    # 的 session_date 底下混進「當天晚上才開始、下一個交易日才結束」的
    # 夜盤尾盤——等於拿還沒發生的未來夜盤收盤價去跟當天日盤開盤價比較，
    # 算出來的「缺口」統計量完全是雜訊（一開始跑出 t=19 的離譜結果就是
    # 這個 bug 造成的，用 2021-05-12 那筆異常值反查資料才抓到）。
    night = d[is_night].copy()
    late = night["datetime"].dt.time >= time(15, 0)
    night["session_label"] = night["datetime"].dt.date
    night.loc[late, "session_label"] = night.loc[late, "datetime"].dt.date + pd.Timedelta(days=1)
    night_last = night.sort_values("datetime").groupby("session_label").last()[["close"]].rename(columns={"close": "night_last_close"})
    night_last.index.name = "session_date"

    day = d[is_day].copy()
    day["trading_date"] = day["datetime"].dt.date
    day_agg = day.groupby("trading_date").agg(day_open=("open", "first"), day_close=("close", "last"),
                                                day_high=("high", "max"), day_low=("low", "min")).reset_index()
    day_agg = day_agg.rename(columns={"trading_date": "session_date"})

    merged = day_agg.merge(night_last, on="session_date", how="inner")
    merged["gap_points"] = merged["day_open"] - merged["night_last_close"]
    merged["gap_pct"] = merged["gap_points"] / merged["night_last_close"] * 100
    merged["day_session_return_pts"] = merged["day_close"] - merged["day_open"]
    return merged.sort_values("session_date").reset_index(drop=True)


def tstat(x: pd.Series) -> float:
    n = len(x)
    if n < 2 or x.std(ddof=1) == 0:
        return np.nan
    return x.mean() / (x.std(ddof=1) / np.sqrt(n))


def main() -> None:
    df = pd.read_parquet(DATA_PATH)
    gaps = build_gap_table(df)
    is_gaps = gaps[gaps["session_date"].astype(str) < IS_CUTOFF]

    print(f"總天數: {len(gaps)}, IS天數: {len(is_gaps)}")
    print("\n缺口點數分布（IS）:")
    print(is_gaps["gap_points"].describe())

    print("\n=== 缺口方向 -> 當天日盤走勢 的相關性（IS）===")
    corr = is_gaps["gap_points"].corr(is_gaps["day_session_return_pts"])
    print(f"gap_points vs day_session_return_pts 相關係數: {corr:.4f}")
    print("（正相關=缺口有延續傾向、可做突破；負相關=缺口容易回補、可做反向）")

    print("\n=== 依缺口幅度分桶（IS，用缺口絕對值的三分位）===")
    is_gaps = is_gaps.copy()
    is_gaps["abs_gap"] = is_gaps["gap_points"].abs()
    is_gaps["gap_bucket"] = pd.qcut(is_gaps["abs_gap"], 3, labels=["小", "中", "大"])
    for bucket in ["小", "中", "大"]:
        sub = is_gaps[is_gaps["gap_bucket"] == bucket]
        # 策略一：延續（跟缺口方向同向做）
        continuation_pnl = np.sign(sub["gap_points"]) * sub["day_session_return_pts"]
        # 策略二：回補（反缺口方向做）
        fade_pnl = -np.sign(sub["gap_points"]) * sub["day_session_return_pts"]
        print(f"\n缺口幅度={bucket} (n={len(sub)}, 平均缺口絕對值={sub['abs_gap'].mean():.1f}點)")
        print(f"  延續策略: mean={continuation_pnl.mean():.2f}pt t={tstat(continuation_pnl):.2f}")
        print(f"  回補策略: mean={fade_pnl.mean():.2f}pt t={tstat(fade_pnl):.2f}")

    print("\n=== 只看大缺口（前1/3分位）逐年 t 值，檢查穩健性 ===")
    big_gap = is_gaps[is_gaps["gap_bucket"] == "大"].copy()
    big_gap["year"] = pd.to_datetime(big_gap["session_date"]).dt.year
    for direction, label in [(1, "延續"), (-1, "回補")]:
        pnl = direction * np.sign(big_gap["gap_points"]) * big_gap["day_session_return_pts"]
        big_gap[f"pnl_{label}"] = pnl
    for name, start, end in [("2001-2010", 2001, 2011), ("2011-2020", 2011, 2021)]:
        sub = big_gap[(big_gap["year"] >= start) & (big_gap["year"] < end)]
        print(f"{name}: n={len(sub)}  延續 t={tstat(sub['pnl_延續']):.2f} mean={sub['pnl_延續'].mean():.2f}"
              f"   回補 t={tstat(sub['pnl_回補']):.2f} mean={sub['pnl_回補'].mean():.2f}")

    gaps.to_parquet(REPO_ROOT / "data" / "preopen_gap_table.parquet", index=False)
    print(f"\nsaved: data/preopen_gap_table.parquet")


if __name__ == "__main__":
    main()
