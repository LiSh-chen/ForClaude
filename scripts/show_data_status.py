"""印出資料庫目前實際的內容：每檔股票的資料範圍與筆數。

跟 ingest_daily_data.py 共用 get_data_store()，所以不管現在是本機 SQLite
還是雲端 Postgres 都直接看得到真實狀態。這支腳本會在每次
daily_data_ingest.yml 執行後自動跑一次（見該 workflow 的
"Show current data status" 步驟，用 `if: always()` 確保就算抓取那步失敗
或逾時也一樣會印出目前資料庫裡實際有什麼），方便不用另外手動診斷就能看到
真實進度。

用法：
    python scripts/show_data_status.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tw_quant.storage import get_data_store


def main() -> None:
    store = get_data_store()
    prices = store.load_prices()
    margin = store.load_margin_short()
    us_prices = store.load_us_prices()

    print(f"prices（台股）總筆數: {len(prices)}")
    print(f"margin_short 總筆數: {len(margin)}")
    print(f"us_prices（美股）總筆數: {len(us_prices)}")

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
