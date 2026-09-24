"""探索「跨日同時段動量延續」：固定選一個每天都存在的時段（例如開盤前15分鐘、
第一小時、整個日盤），檢查「昨天這個時段的報酬方向」能不能預測「今天同一個
時段的報酬方向」——完全不需要任何連續指標，只要記住昨天一個數字（正或負、
或報酬大小），今天在固定時刻用固定方向進出場即可，跟趨勢日/VWAP系列的
「當天內部動態判斷」是不同的資訊來源（跨日記憶，不是當天內部持續性）。

檢查的時段（全部用該時段的「收盤-開盤」報酬當基本單位）：
1. 開盤前15分鐘（08:45-09:00）
2. 開盤第一小時（08:45-09:45）
3. 午盤前半（12:00-12:30，之前驗證過的「午盤放空」時段）
4. 尾盤（13:15-13:45）
5. 整個日盤（08:45-13:45）

策略：昨天該時段報酬為正 -> 今天同時段做多；昨天為負 -> 今天同時段做空
（純粹方向延續，符號動量），另外也測反向（符號反轉，均值回歸）看哪個
方向有訊號。

用法：
    python scripts/scan_cross_day_momentum.py
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

SEGMENTS = {
    "開盤前15分鐘(08:45-09:00)": (time(8, 45), time(9, 0)),
    "開盤第一小時(08:45-09:45)": (time(8, 45), time(9, 45)),
    "午盤前半(12:00-12:30)": (time(12, 0), time(12, 30)),
    "尾盤(13:15-13:45)": (time(13, 15), time(13, 45)),
    "整個日盤(08:45-13:45)": (time(8, 45), time(13, 45)),
}


def build_segment_returns(df: pd.DataFrame, start: time, end: time) -> pd.Series:
    t = df["datetime"].dt.time
    day_df = df[(t >= time(8, 45)) & (t <= time(13, 45))].copy()
    day_df["trading_date"] = day_df["datetime"].dt.date
    day_df["minute_of_day"] = day_df["datetime"].dt.hour * 60 + day_df["datetime"].dt.minute
    start_min = start.hour * 60 + start.minute
    end_min = end.hour * 60 + end.minute

    rows = []
    for d, g in day_df.groupby("trading_date"):
        g = g.sort_values("datetime")
        mo, op = g["minute_of_day"].to_numpy(), g["open"].to_numpy()
        pos_s = np.searchsorted(mo, start_min)
        pos_e = np.searchsorted(mo, end_min)
        if pos_s >= len(mo) or pos_e >= len(mo):
            continue
        rows.append((d, op[pos_e] - op[pos_s]))
    s = pd.Series({d: r for d, r in rows}).sort_index()
    s.index = pd.to_datetime(s.index)
    return s


def tstat(x: pd.Series) -> float:
    n = len(x)
    if n < 2 or x.std(ddof=1) == 0:
        return np.nan
    return x.mean() / (x.std(ddof=1) / np.sqrt(n))


def main() -> None:
    df = pd.read_parquet(DATA_PATH)
    is_df = df[df["datetime"] < IS_CUTOFF]

    print(f"{'時段':<28} {'自相關(lag1)':>12} {'延續 t值':>10} {'延續 mean':>10} {'反轉 t值':>10} {'反轉 mean':>10} {'n':>6}")
    results = {}
    for name, (start, end) in SEGMENTS.items():
        seg_ret = build_segment_returns(is_df, start, end)
        yesterday_ret = seg_ret.shift(1)
        valid = yesterday_ret.notna() & seg_ret.notna() & (yesterday_ret != 0)
        y = yesterday_ret[valid]
        today = seg_ret[valid]

        autocorr = y.corr(today)

        cont_pnl = np.sign(y) * today  # 延續（動量）：昨天正，今天做多；昨天負，今天做空
        rev_pnl = -np.sign(y) * today  # 反轉（均值回歸）

        results[name] = dict(seg_ret=seg_ret, yesterday=y, today=today)
        print(f"{name:<28} {autocorr:>12.4f} {tstat(cont_pnl):>10.2f} {cont_pnl.mean():>10.2f} "
              f"{tstat(rev_pnl):>10.2f} {rev_pnl.mean():>10.2f} {len(today):>6}")

    print("\n=== 子區間穩健性（只看全樣本|t|>=1.5的候選）===")
    for name, (start, end) in SEGMENTS.items():
        seg_ret = results[name]["seg_ret"]
        yesterday_ret = seg_ret.shift(1)
        valid = yesterday_ret.notna() & seg_ret.notna() & (yesterday_ret != 0)
        y_full, today_full = yesterday_ret[valid], seg_ret[valid]
        cont_pnl_full = np.sign(y_full) * today_full
        if abs(tstat(cont_pnl_full)) < 1.5:
            continue
        print(f"\n{name}（延續策略全樣本t={tstat(cont_pnl_full):.2f}）:")
        for sub_name, start_y, end_y in [("2001-2005", 2001, 2006), ("2006-2010", 2006, 2011),
                                          ("2011-2015", 2011, 2016), ("2016-2020", 2016, 2021)]:
            mask = (cont_pnl_full.index.year >= start_y) & (cont_pnl_full.index.year < end_y)
            sub = cont_pnl_full[mask]
            print(f"  {sub_name}: n={len(sub)} mean={sub.mean():.2f} t={tstat(sub):.2f}")


if __name__ == "__main__":
    main()
