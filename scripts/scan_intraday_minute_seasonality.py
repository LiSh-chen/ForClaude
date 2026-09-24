"""日盤盤中「分鐘級別季節性」掃描：這次會話測過日曆效應（星期幾/月中
位置/結算日）、夜盤時段效應，但沒測過「盤中每一分鐘」本身在23年間是否
存在統計上穩健的方向性偏態——例如開盤集合競價成交後的短暫修正、收盤前
的慣性效應，這類微結構效應在其他市場文獻中常見，TXF日盤有約299個
交易分鐘（08:46~13:45，每分鐘一個1分鐘報酬率），逐一檢查。

方法論：
- 每個交易日、每一分鐘算 (close_t - close_{t-1}) 當作那一分鐘的報酬率
  （用收盤價差分，不是開高低收混合，避免雜訊）。
- 對每個「分鐘時刻」（例如09:00、09:01...）橫跨23年所有交易日算平均
  報酬率跟t值——這是299次獨立假設檢定，多重比較問題比之前任何一次
  網格搜尋都更嚴重（299 vs 之前最多360組合但那是策略參數組合，這裡
  是299個幾乎獨立的時間切片假設）。
- 用 Bonferroni 校正的思維設定門檻：299次檢定要維持家族錯誤率5%，
  單次檢定門檻大約要 |t| >= 3.6~4（用常態分位數估算），比這次會話
  其餘篩選標準(|t|>=2)嚴格得多，就是為了因應這裡多重比較次數多更多
  這個事實。
- 通過初篩的分鐘再檢查四個5年子區間是否同方向（跟其餘網格搜尋一致的
  紀律）。

用法：
    python scripts/scan_intraday_minute_seasonality.py
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
OUT_DIR = REPO_ROOT / "data"

IS_CUTOFF = "2021-01-01"
BONFERRONI_T_THRESHOLD = 3.6
SUB_PERIODS = [
    ("2001-2005", "2001-01-01", "2006-01-01"),
    ("2006-2010", "2006-01-01", "2011-01-01"),
    ("2011-2015", "2011-01-01", "2016-01-01"),
    ("2016-2020", "2016-01-01", "2021-01-01"),
]


def tstat(x: pd.Series) -> float:
    n = len(x)
    if n < 2 or x.std(ddof=1) == 0:
        return np.nan
    return x.mean() / (x.std(ddof=1) / np.sqrt(n))


def _day_session(df: pd.DataFrame) -> pd.DataFrame:
    t = df["datetime"].dt.time
    d = df[(t >= time(8, 45)) & (t <= time(13, 45))].copy()
    d["trading_date"] = d["datetime"].dt.date
    d["time_of_day"] = d["datetime"].dt.time
    d = d.sort_values(["trading_date", "datetime"])
    d["minute_ret"] = d.groupby("trading_date")["close"].diff()
    return d


def main() -> None:
    df = pd.read_parquet(DATA_PATH)
    day_df = _day_session(df)
    is_df = day_df[day_df["datetime"] < IS_CUTOFF].dropna(subset=["minute_ret"])

    print("=" * 70)
    print(f"1) 每個盤中分鐘時刻的IS(2001-2020)報酬率t值（共 {is_df['time_of_day'].nunique()} 個時刻）")
    print("=" * 70)
    grouped = is_df.groupby("time_of_day")["minute_ret"]
    stats = grouped.agg(n="count", mean_pts="mean", t_stat=tstat).reset_index()
    stats = stats.sort_values("t_stat", ascending=False)
    print("t值最高的10個時刻：")
    print(stats.head(10).to_string(index=False))
    print("\nt值最低的10個時刻：")
    print(stats.tail(10).to_string(index=False))

    stats.to_parquet(OUT_DIR / "intraday_minute_seasonality_is.parquet", index=False)

    print("\n" + "=" * 70)
    print(f"2) Bonferroni校正篩選：|t| >= {BONFERRONI_T_THRESHOLD}（因應{len(stats)}次檢定的多重比較）")
    print("=" * 70)
    screen = stats[stats["t_stat"].abs() >= BONFERRONI_T_THRESHOLD]
    print(f"通過初篩: {len(screen)} / {len(stats)}")
    if screen.empty:
        print("\n沒有任何分鐘時刻通過Bonferroni校正門檻，全面否決，不需要往下驗證OOS。")
        return

    print(screen.to_string(index=False))

    print("\n" + "=" * 70)
    print("3) 通過初篩的時刻，檢查四個子區間是否同方向")
    print("=" * 70)
    candidates = []
    for _, row in screen.iterrows():
        tod = row["time_of_day"]
        sub_tstats = []
        for name, start, end in SUB_PERIODS:
            sub = day_df[(day_df["datetime"] >= start) & (day_df["datetime"] < end) & (day_df["time_of_day"] == tod)]
            sub = sub.dropna(subset=["minute_ret"])
            sub_tstats.append(tstat(sub["minute_ret"]) if len(sub) >= 30 else np.nan)
        overall_sign = np.sign(row["t_stat"])
        same_sign_count = sum(1 for t in sub_tstats if not np.isnan(t) and np.sign(t) == overall_sign)
        candidates.append(dict(time_of_day=tod, n=row["n"], t_stat=row["t_stat"], mean_pts=row["mean_pts"],
                                sub_tstats=sub_tstats, same_sign_count=same_sign_count))
    cand_df = pd.DataFrame(candidates).sort_values("same_sign_count", ascending=False)
    pd.set_option("display.width", 200)
    print(cand_df.to_string(index=False))

    robust = cand_df[cand_df["same_sign_count"] >= 4]
    print(f"\n四個子區間全部同方向的候選數: {len(robust)}")
    cand_df.to_parquet(OUT_DIR / "intraday_minute_seasonality_candidates.parquet", index=False)


if __name__ == "__main__":
    main()
