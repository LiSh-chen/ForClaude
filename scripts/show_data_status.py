"""印出目前實際的資料狀態：每檔股票的資料範圍與筆數。

台股資料跟 ingest_daily_data.py 共用 get_data_store()（本機 SQLite，
見 tw_quant/storage.py 開頭說明）。美股資料 2026-09-30 起已經不經過
SQLite，直接讀 Parquet 快照（tw_quant/data_snapshot.py 的
load_us_prices_snapshot，見該檔案「2026-09-30 退役 SQLite」說明）——這支
腳本在台股/美股兩個每日排程 workflow 裡都會跑一次（用 `if: always()`
確保就算抓取那步失敗或逾時也一樣會印出目前實際有什麼），方便不用另外
手動診斷就能看到真實進度。

用法：
    python scripts/show_data_status.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from tw_quant.data_snapshot import load_us_prices_snapshot
from tw_quant.storage import PRICE_COLS, get_data_store


def main() -> None:
    store = get_data_store()
    prices = store.load_prices()
    margin = store.load_margin_short()
    try:
        us_prices = load_us_prices_snapshot()
    except FileNotFoundError:
        us_prices = pd.DataFrame(columns=PRICE_COLS)

    print(f"prices（台股）總筆數: {len(prices)}")
    print(f"margin_short 總筆數: {len(margin)}")
    print(f"us_prices（美股，讀自 Parquet 快照）總筆數: {len(us_prices)}")

    if not prices.empty:
        print(f"\n台股共 {prices['stock_id'].nunique()} 檔股票，每檔的資料範圍：")
        summary = prices.groupby("stock_id")["date"].agg(["min", "max", "count"])
        summary.columns = ["最早日期", "最新日期", "筆數"]
        print(summary.to_string())

    if not us_prices.empty:
        print(f"\n美股共 {us_prices['stock_id'].nunique()} 檔股票，每檔的資料範圍：")
        summary = us_prices.groupby("stock_id")["date"].agg(["min", "max", "count"])
        summary.columns = ["最早日期", "最新日期", "筆數"]
        print(summary.to_string())

    if prices.empty and us_prices.empty:
        print("（資料庫目前是空的）")


if __name__ == "__main__":
    main()
