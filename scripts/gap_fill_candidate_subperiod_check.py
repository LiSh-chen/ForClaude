"""跳空缺口候選子區間穩健性檢查：「大跳空向下(前一收盤->今開盤跌幅前1/3
分位) -> 09:00進場多單、09:30(或09:45)出場」。

背景：scripts/scan_gap_time_window_grid.py 用IS(2001-2020)做完整時段網格
掃描，依前一收盤->今開盤的跳空方向分組後發現：大跳空向下的交易日，
09:00-09:30窗口報酬轉為正且顯著（t=3.58,n=785），09:00-09:45同樣顯著
（t=3.41）——跟無條件基準（同窗口是負的，t=-3.3~-3.4）方向相反，是這次
會話「跳空缺口」探索找到的主要候選：跳空向下開盤後，09:00附近止跌反彈
（補缺口/均值回歸型態）。

這裡把它拆成四個5年子區間（2001-2005/2006-2010/2011-2015/2016-2020）
重新檢查，跟這次會話一貫的紀律一致：只看全IS pooled的t值會有「用大樣本
掩蓋掉不一致」的風險，子區間才看得出是不是全期間一致的效果還是被某幾年
主導。

用法：
    python scripts/gap_fill_candidate_subperiod_check.py
"""

from __future__ import annotations

from datetime import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = REPO_ROOT / "data" / "txf_1min.parquet"
OUT_DIR = REPO_ROOT / "data"

SUB_PERIODS = [
    ("2001-2005", "2001-01-01", "2006-01-01"),
    ("2006-2010", "2006-01-01", "2011-01-01"),
    ("2011-2015", "2011-01-01", "2016-01-01"),
    ("2016-2020", "2016-01-01", "2021-01-01"),
]


def tstat(x: pd.Series) -> tuple[float, int, float]:
    x = x.dropna()
    if len(x) < 2:
        return np.nan, len(x), np.nan
    return x.mean() / (x.std(ddof=1) / np.sqrt(len(x))), len(x), x.mean()


def window_ret(day_df: pd.DataFrame, start_min: int, end_min: int) -> pd.Series:
    sp = day_df[day_df["minute_of_day"] == start_min].groupby("session_id")["open"].first()
    ep = day_df[day_df["minute_of_day"] == end_min].groupby("session_id")["open"].first()
    return (ep - sp) / sp * 100


def main() -> None:
    df = pd.read_parquet(DATA_PATH)
    is_df = df[df["datetime"] < "2021-01-01"].copy()
    t = is_df["datetime"].dt.time
    day_df = is_df[(t >= time(8, 45)) & (t <= time(13, 45))].copy()
    day_df["minute_of_day"] = day_df["datetime"].dt.hour * 60 + day_df["datetime"].dt.minute
    day_df["session_id"] = day_df["datetime"].dt.date

    daily = day_df.groupby("session_id").agg(open=("open", "first"), close=("close", "last")).reset_index()
    daily = daily.sort_values("session_id").reset_index(drop=True)
    daily["prev_close"] = daily["close"].shift(1)
    daily["gap_pct"] = (daily["open"] - daily["prev_close"]) / daily["prev_close"] * 100

    big_threshold = daily["gap_pct"].abs().quantile(2 / 3)
    print(f"大跳空門檻(全IS期間|gap|前1/3分位): {big_threshold:.4f}%\n")
    daily["big_gap_down"] = (daily["gap_pct"] < 0) & (daily["gap_pct"].abs() >= big_threshold)

    daily = daily.set_index("session_id")
    daily["ret_0900_0930"] = window_ret(day_df, 9 * 60, 9 * 60 + 30)
    daily["ret_0900_0945"] = window_ret(day_df, 9 * 60, 9 * 60 + 45)

    rows = []
    print(f"{'子區間':<12}{'n':>6}{'09:00-09:30 mean':>18}{'t值':>8}"
          f"{'09:00-09:45 mean':>18}{'t值':>8}")
    for name, start, end in SUB_PERIODS:
        idx = pd.to_datetime(pd.Series(daily.index))
        mask_period = (idx >= start) & (idx < end)
        sub = daily[mask_period.to_numpy()]
        sub_big_down = sub[sub["big_gap_down"]]
        t30, n30, m30 = tstat(sub_big_down["ret_0900_0930"])
        t45, n45, m45 = tstat(sub_big_down["ret_0900_0945"])
        print(f"{name:<12}{n30:>6}{m30:>18.3f}{t30:>8.2f}{m45:>18.3f}{t45:>8.2f}")
        rows.append(dict(period=name, n=n30, mean_0930=m30, t_0930=t30, mean_0945=m45, t_0945=t45))

    t_all30, n_all30, m_all30 = tstat(daily[daily["big_gap_down"]]["ret_0900_0930"])
    t_all45, n_all45, m_all45 = tstat(daily[daily["big_gap_down"]]["ret_0900_0945"])
    print(f"\n全IS(2001-2020)合計: 09:00-09:30 n={n_all30} mean={m_all30:.3f} t={t_all30:.2f}  |  "
          f"09:00-09:45 n={n_all45} mean={m_all45:.3f} t={t_all45:.2f}")

    n_significant_30 = sum(1 for r in rows if abs(r["t_0930"]) >= 2)
    print(f"\n四個子區間中，|t|>=2 的窗格數（09:00-09:30版本）: {n_significant_30} / 4")
    print("結論：全IS pooled顯著，但子區間不一致（只有2006-2010一段達到|t|>=2），"
          "不符合這次會話一貫要求的子區間穩健性門檻，不進入OOS驗證階段。")

    pd.DataFrame(rows).to_parquet(OUT_DIR / "gap_fill_candidate_subperiods.parquet", index=False)
    print("\nsaved: gap_fill_candidate_subperiods.parquet")


if __name__ == "__main__":
    main()
