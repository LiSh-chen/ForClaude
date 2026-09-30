"""補「不同時間週期」的缺口：這次會話目前為止測過的都是1分K(三腿策略/
夜盤)或日線體制分類(高波動濾網)，從沒測過週線級別的波段訊號——完全不同
的操作頻率(一週看一次盤，不是每天甚至每分鐘)，對散戶而言是操作負擔
最低的一種頻率。

用近月期貨重建的日線(2001-2020 IS)，依ISO週resample成週線(週五收盤，
若週五休市則用當週最後一個交易日)，測兩個經典的週線假設：
1. 週動能延續：上週漲(跌)這週繼續漲(跌)
2. 週線位置(收盤 vs N週均線)：趨勢排列是否對下週報酬有預測力

用法：
    python scripts/scan_weekly_timeframe.py
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = REPO_ROOT / "data" / "raw" / "taifex_tx"
IS_END = "2021-01-01"
OOS_END = "2024-01-01"
EXPIRY_RE = re.compile(r"^\d{6}$")


def build_near_month_ohlc() -> pd.DataFrame:
    files = sorted(RAW_DIR.glob("TX_*.csv"))
    frames = [pd.read_csv(fp, dtype=str, index_col=False) for fp in files]
    all_df = pd.concat(frames, ignore_index=True)
    all_df.columns = [c.strip() for c in all_df.columns]
    d = all_df.copy()
    d["到期月份(週別)"] = d["到期月份(週別)"].str.strip()
    d["交易時段"] = d["交易時段"].str.strip()
    d = d[d["交易時段"] == "一般"]
    d = d[d["到期月份(週別)"].str.match(EXPIRY_RE, na=False)]
    d["date"] = pd.to_datetime(d["交易日期"], format="%Y/%m/%d")

    def to_num(col):
        return pd.to_numeric(d[col].str.replace(",", "", regex=False), errors="coerce")

    d["open"] = to_num("開盤價")
    d["high"] = to_num("最高價")
    d["low"] = to_num("最低價")
    d["close"] = to_num("收盤價")
    d["volume"] = to_num("成交量")
    d = d[d["volume"].fillna(0) > 0]

    rows = []
    for dt, sub in d.groupby("date", sort=True):
        sub = sub.sort_values("到期月份(週別)")
        near = sub.iloc[0]
        rows.append(dict(date=dt, open=near["open"], high=near["high"],
                          low=near["low"], close=near["close"]))
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def tstat(x: pd.Series) -> tuple[float, int, float]:
    x = x.dropna()
    n = len(x)
    if n < 2 or x.std(ddof=1) == 0:
        return np.nan, n, np.nan
    return x.mean() / (x.std(ddof=1) / np.sqrt(n)), n, x.mean()


def main() -> None:
    print("重建近月OHLC並轉週線...")
    d = build_near_month_ohlc().set_index("date")
    weekly = d.resample("W-FRI").agg(open=("open", "first"), high=("high", "max"),
                                       low=("low", "min"), close=("close", "last")).dropna()
    weekly["ret_this_week"] = weekly["close"].pct_change() * 100
    weekly["ret_next_week"] = weekly["close"].shift(-1) / weekly["close"] * 100 - 100

    is_w = weekly[weekly.index < IS_END]
    oos_w = weekly[(weekly.index >= IS_END) & (weekly.index < OOS_END)]
    print(f"總週數: {len(weekly)}，IS週數: {len(is_w)}，OOS週數: {len(oos_w)}\n")

    print("=" * 78)
    print("1) 週動能延續：上週漲/跌 -> 這週報酬 (IS 2001-2020)")
    print("=" * 78)
    up_prev = is_w.loc[is_w["ret_this_week"].shift(1) > 0, "ret_this_week"]
    down_prev = is_w.loc[is_w["ret_this_week"].shift(1) < 0, "ret_this_week"]
    t_u, n_u, m_u = tstat(up_prev)
    t_d, n_d, m_d = tstat(down_prev)
    print(f"  上週漲後: n={n_u} mean={m_u:.3f}% t={t_u:.2f}")
    print(f"  上週跌後: n={n_d} mean={m_d:.3f}% t={t_d:.2f}")

    print("\n" + "=" * 78)
    print("2) 週線位置：收盤 vs N週均線 -> 下週報酬 (IS 2001-2020)")
    print("=" * 78)
    for n_weeks in [10, 20, 40, 52]:
        ma = is_w["close"].rolling(n_weeks).mean()
        above = is_w.loc[is_w["close"] > ma, "ret_next_week"]
        below = is_w.loc[is_w["close"] <= ma, "ret_next_week"]
        t_a, n_a, m_a = tstat(above)
        t_b, n_b, m_b = tstat(below)
        print(f"  {n_weeks}週均線: 站上 n={n_a} mean={m_a:.3f}% t={t_a:.2f} | "
              f"跌破 n={n_b} mean={m_b:.3f}% t={t_b:.2f}")

    print("\n" + "=" * 78)
    print("對照：全樣本無條件下週報酬")
    print("=" * 78)
    t, n, m = tstat(is_w["ret_next_week"])
    print(f"  全樣本: n={n} mean={m:.3f}% t={t:.2f}")


if __name__ == "__main__":
    main()
