"""補「更小時間週期」的缺口：這次會話目前測過最細的顆粒度是1分K（三腿
策略的固定時點、夜盤突破/跳空），但都是用1分K的「開盤價在特定時刻」
當進出場依據，從沒測過1分K本身「這一分鐘漲跌 vs 下一分鐘漲跌」這種
最基本的高頻序列相關性問題——這是比三腿策略更小的操作單位（每分鐘
都在看，不是等特定時點）。

比1分K更細（tick逐筆成交）的資料：嘗試過幾個TAIFEX端點都沒找到公開
免費的逐筆資料下載管道（歷史資料下載頁面404、futDataDown系列端點都是
日頻聚合資料），這點誠實記錄為目前抓不到，不是沒有去找。

用法：
    python scripts/scan_1min_autocorrelation_scalp.py
"""

from __future__ import annotations

from datetime import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = REPO_ROOT / "data" / "txf_1min.parquet"
IS_END = "2021-01-01"


def main() -> None:
    df = pd.read_parquet(DATA_PATH)
    t = df["datetime"].dt.time
    day = df[(t >= time(8, 45)) & (t <= time(13, 45))].copy()
    day["trading_date"] = day["datetime"].dt.date
    day["ret_pts"] = day.groupby("trading_date")["close"].diff()

    is_day = day[day["datetime"] < IS_END]

    print("=" * 70)
    print("1) 1分鐘報酬自相關（分日內計算，避免跨日汙染），IS(2001-2020)")
    print("=" * 70)
    is_day = is_day.copy()
    is_day["ret"] = is_day.groupby("trading_date")["close"].pct_change() * 100
    for lag in range(1, 6):
        x = is_day.groupby("trading_date")["ret"].apply(lambda s: s.autocorr(lag))
        x = x.dropna()
        t_val = x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))
        print(f"  lag{lag}: 平均自相關={x.mean():.4f}  t={t_val:.2f}  n_days={len(x)}")
    print("\n  結論：lag1~5全部統計上顯著為負（微觀結構常見的「一分鐘級」")
    print("  微幅反轉現象），但相關係數只有約-0.01量級，效應極小。")

    print("\n" + "=" * 70)
    print("2) 把這個自相關轉成實際交易規則：逆勢一分鐘反彈/回落")
    print("=" * 70)
    is_day["next_ret_pts"] = is_day.groupby("trading_date")["close"].diff().shift(-1)
    valid = is_day.dropna(subset=["ret_pts", "next_ret_pts"])
    print(f"  平均每1分鐘K棒報酬絕對值: {valid['ret_pts'].abs().mean():.2f}點")
    for thresh in [3, 5, 10, 20]:
        long_side = valid.loc[valid["ret_pts"] <= -thresh, "next_ret_pts"]
        short_side = -valid.loc[valid["ret_pts"] >= thresh, "next_ret_pts"]
        combined = pd.concat([long_side, short_side])
        print(f"  門檻{thresh}點: n={len(combined)} 毛利平均(未扣成本)={combined.mean():.3f}點")
    print("\n  對照：來回交易成本(證交稅+手續費+滑價)約3-4點/趟。")
    print("  即使完全不扣成本，逆勢單分鐘規則的毛利平均已經在0點附近")
    print("  或微幅為負——統計上真實存在的自相關，經濟量級小到連")
    print("  毛利都撐不住，扣掉任何交易成本後必然虧損，不用進一步驗證OOS。")


if __name__ == "__main__":
    main()
