"""把資料庫目前的美股資料（us_prices、us_index_membership）匯出成
commit 進 repo 的 Parquet 快照檔，供策略回測腳本改用本機讀取，不用每次
都連 Postgres 整表撈一次（見 tw_quant/data_snapshot.py 的背景說明）。

要在 ingest_us_daily_data.py 或 backfill_removed_sp500_stocks.py 真的
寫入資料庫「之後」才跑，確保快照反映的是最新資料，不是舊的。

用法：
    python scripts/export_us_data_snapshot.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tw_quant.data_snapshot import US_INDEX_MEMBERSHIP_SNAPSHOT_PATH, US_PRICES_SNAPSHOT_PATH, export_us_snapshot
from tw_quant.storage import get_data_store


def main() -> None:
    store = get_data_store()
    n_prices, n_membership = export_us_snapshot(store)
    print(f"已匯出 us_prices {n_prices} 列 -> {US_PRICES_SNAPSHOT_PATH}")
    print(f"已匯出 us_index_membership {n_membership} 列 -> {US_INDEX_MEMBERSHIP_SNAPSHOT_PATH}")


if __name__ == "__main__":
    main()
