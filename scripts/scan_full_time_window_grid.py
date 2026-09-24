"""在樣本內（2001-2020）資料上，對日盤／夜盤分別做「完整 (start, end) 時段
報酬網格」掃描——不像 scan_intraday_seasonality.py 只看固定 30 分鐘格子，這裡
任何起訖時間組合（每 15 分鐘為單位）都算，才挖得到像 08:45-09:00 這種剛好落在
格子邊界內、長度只有 15 分鐘的強訊號。

方法（效能考量，見 `build_price_grid`）：
1. 對每個交易日／夜盤 session，只取「每個 15 分鐘錨點時刻，第一根 >= 該時刻
   的K棒開盤價」，組成一個 (session × anchor) 的價格矩陣。
2. 任兩個錨點 i<j 的報酬 = 矩陣[:,j] - 矩陣[:,i]，一次對所有 session 做
   向量化減法，不用對每個 (start,end) 組合重新掃一次K棒——這樣 O(anchor^2)
   組合的整個網格可以在幾秒內跑完，而不是對每個 duration 分開迴圈。

用法：
    python scripts/scan_full_time_window_grid.py
"""

from __future__ import annotations

from datetime import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = REPO_ROOT / "data" / "txf_1min.parquet"
DAY_OUT = REPO_ROOT / "data" / "day_full_grid_scan.parquet"
NIGHT_OUT = REPO_ROOT / "data" / "night_full_grid_scan.parquet"

IS_CUTOFF = "2021-01-01"
MIN_N = 200


def build_price_grid(session_df: pd.DataFrame, session_id_col: str, anchor_minutes: list[int]) -> pd.DataFrame:
    g = session_df.sort_values("datetime")
    sessions = g[session_id_col].unique()
    session_idx = {s: i for i, s in enumerate(sessions)}
    grid = np.full((len(sessions), len(anchor_minutes)), np.nan)
    anchor_arr = np.array(anchor_minutes)

    for sid, sub in g.groupby(session_id_col, sort=False):
        mo = sub["minute_of_day"].to_numpy()
        op = sub["open"].to_numpy()
        pos = np.searchsorted(mo, anchor_arr, side="left")
        valid = pos < len(mo)
        row = session_idx[sid]
        grid[row, valid] = op[pos[valid]]
    return pd.DataFrame(grid, index=sessions, columns=anchor_minutes)


def full_grid_scan(grid: pd.DataFrame) -> pd.DataFrame:
    anchors = grid.columns.to_numpy()
    vals = grid.to_numpy()
    rows = []
    for i in range(len(anchors)):
        for j in range(i + 1, len(anchors)):
            ret = vals[:, j] - vals[:, i]
            valid = ~np.isnan(ret)
            n = valid.sum()
            if n < MIN_N:
                continue
            r = ret[valid]
            se = r.std(ddof=1) / np.sqrt(n)
            rows.append(dict(
                start=anchors[i], end=anchors[j], duration=anchors[j] - anchors[i],
                n=n, mean_ret=r.mean(), tstat=(r.mean() / se) if se > 0 else np.nan,
            ))
    res = pd.DataFrame(rows)

    def fmt(m: float) -> str:
        h, mi = int(m // 60) % 24, int(m % 60)
        return f"{h:02d}:{mi:02d}" + ("+1" if m >= 1440 else "")

    res["start_label"] = res["start"].apply(fmt)
    res["end_label"] = res["end"].apply(fmt)
    return res


def main() -> None:
    df = pd.read_parquet(DATA_PATH)
    is_df = df[df["datetime"] < IS_CUTOFF].copy()

    t = is_df["datetime"].dt.time
    day_df = is_df[(t >= time(8, 45)) & (t <= time(13, 45))].copy()
    day_df["minute_of_day"] = day_df["datetime"].dt.hour * 60 + day_df["datetime"].dt.minute
    day_df["session_id"] = day_df["datetime"].dt.date
    day_anchors = list(range(8 * 60 + 45, 13 * 60 + 46, 15))
    day_grid = build_price_grid(day_df, "session_id", day_anchors)
    day_res = full_grid_scan(day_grid)
    day_res.to_parquet(DAY_OUT, index=False)

    night_df = is_df[(t >= time(15, 0)) | (t < time(5, 0))].copy()
    night_df["minute_of_day"] = night_df["datetime"].dt.hour * 60 + night_df["datetime"].dt.minute
    night_df.loc[night_df["datetime"].dt.time < time(5, 0), "minute_of_day"] += 1440
    night_df["session_date"] = night_df["datetime"].dt.date
    early = night_df["datetime"].dt.time < time(5, 0)
    night_df.loc[early, "session_date"] = night_df.loc[early, "datetime"].dt.date - pd.Timedelta(days=1)
    night_anchors = list(range(15 * 60, 24 * 60 + 5 * 60, 15))
    night_grid = build_price_grid(night_df, "session_date", night_anchors)
    night_res = full_grid_scan(night_grid)
    night_res.to_parquet(NIGHT_OUT, index=False)

    pd.set_option("display.width", 150)
    pd.set_option("display.float_format", lambda x: f"{x:,.3f}")

    print(f"日盤組合數: {len(day_res)}，夜盤組合數: {len(night_res)}\n")
    print("=== 日盤：|t| 前 20 強 ===")
    print(day_res.reindex(day_res["tstat"].abs().sort_values(ascending=False).index)
          .head(20)[["start_label", "end_label", "duration", "n", "mean_ret", "tstat"]].to_string(index=False))

    print("\n=== 夜盤：|t| 前 20 強 ===")
    print(night_res.reindex(night_res["tstat"].abs().sort_values(ascending=False).index)
          .head(20)[["start_label", "end_label", "duration", "n", "mean_ret", "tstat"]].to_string(index=False))


if __name__ == "__main__":
    main()
