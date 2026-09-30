"""補齊1分鐘和日線之間的顆粒度：把日盤1分K resample成5/15/30/60分鐘K，
測同一個問題（上一根K棒漲跌 vs 下一根K棒漲跌），跟1分鐘版本
(scan_1min_autocorrelation_scalp.py)用同一套方法，只換時間顆粒度。

動機：1分鐘版本統計顯著但經濟量級太小（相關係數~-0.01，換算點數遠低於
3-4點的來回成本）。時間顆粒度放大後，每根K棒本身的波動幅度會變大，
同樣量級的相關係數換算成點數可能更有意義——這裡直接測，不假設答案。

resample只在日盤時段內做（08:45-13:45），每天重新起算，避免跨日或
跨午休的假K棒。

用法：
    python scripts/scan_multi_minute_timeframe.py
"""

from __future__ import annotations

from datetime import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = REPO_ROOT / "data" / "txf_1min.parquet"
IS_END = "2021-01-01"
TIMEFRAMES_MIN = [5, 15, 30, 60]


def tstat_series(x: pd.Series) -> tuple[float, int, float]:
    x = x.dropna()
    n = len(x)
    if n < 2 or x.std(ddof=1) == 0:
        return np.nan, n, np.nan
    return x.mean() / (x.std(ddof=1) / np.sqrt(n)), n, x.mean()


def resample_day_session(df: pd.DataFrame, minutes: int) -> pd.DataFrame:
    """向量化版本：台灣沒有夏令時間調整，每天日盤開盤時刻(08:45)在時鐘上
    完全固定，所以用origin='epoch'的全域resample，切出來的K棒邊界
    (08:30-09:00/09:00-09:30/...)天天對齊一致，不需要逐日迴圈。
    第一根K棒(08:30-09:00只含08:45-08:59這15分鐘，是不完整的短K棒)直接
    丟棄，避免混進一根波動基準不同的假K棒。"""
    t = df["datetime"].dt.time
    day = df[(t >= time(8, 45)) & (t <= time(13, 45))].copy()
    day["trading_date"] = day["datetime"].dt.date

    g = day.sort_values("datetime").set_index("datetime")
    agg = g.resample(f"{minutes}min", origin="epoch", closed="left", label="left").agg(
        open=("open", "first"), high=("high", "max"), low=("low", "min"), close=("close", "last"),
    ).dropna().reset_index()
    agg["trading_date"] = agg["datetime"].dt.date
    agg["bar_time"] = agg["datetime"].dt.time
    first_bar_time = agg.groupby("trading_date")["bar_time"].min()
    agg = agg[agg["bar_time"] != agg["trading_date"].map(first_bar_time)]
    return agg.drop(columns=["bar_time"]).sort_values("datetime").reset_index(drop=True)


def main() -> None:
    df = pd.read_parquet(DATA_PATH)
    df_is = df[df["datetime"] < IS_END]

    for minutes in TIMEFRAMES_MIN:
        print("=" * 78)
        print(f"時間顆粒度: {minutes}分K")
        print("=" * 78)
        bars = resample_day_session(df_is, minutes)
        bars["ret_pct"] = bars.groupby("trading_date")["close"].pct_change() * 100
        bars["ret_pts"] = bars.groupby("trading_date")["close"].diff()
        bars["next_ret_pts"] = bars.groupby("trading_date")["close"].diff().shift(-1)

        n_per_day = bars.groupby("trading_date").size().mean()
        avg_abs_move = bars["ret_pts"].dropna().abs().mean()
        print(f"  平均每天{n_per_day:.1f}根K棒，平均每根K棒報酬絕對值={avg_abs_move:.2f}點")

        print("\n  自相關(lag1~3，分日內計算):")
        for lag in [1, 2, 3]:
            x = bars.groupby("trading_date")["ret_pct"].apply(lambda s: s.autocorr(lag))
            x = x.dropna()
            if len(x) < 30 or x.std(ddof=1) == 0:
                print(f"    lag{lag}: 樣本不足，略過")
                continue
            t_val = x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))
            print(f"    lag{lag}: 平均自相關={x.mean():.4f}  t={t_val:.2f}  n_days={len(x)}")

        print("\n  轉成逆勢交易規則(門檻=平均波動的0.5/1/2倍)，毛利平均(未扣成本):")
        valid = bars.dropna(subset=["ret_pts", "next_ret_pts"])
        for mult in [0.5, 1.0, 2.0]:
            thresh = avg_abs_move * mult
            long_side = valid.loc[valid["ret_pts"] <= -thresh, "next_ret_pts"]
            short_side = -valid.loc[valid["ret_pts"] >= thresh, "next_ret_pts"]
            combined = pd.concat([long_side, short_side])
            t_val, n_val, m_val = tstat_series(combined)
            print(f"    門檻={thresh:.1f}點({mult}x平均波動): n={n_val} 毛利平均={m_val:.3f}點 t={t_val:.2f}")
        print()


if __name__ == "__main__":
    main()
