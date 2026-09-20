"""把資料庫目前的美股財報公布資料（us_earnings）匯出成 commit 進 repo 的
Parquet 快照檔，供 PEAD 策略回測腳本改用本機讀取，不用每次都連 Postgres
整表撈一次（見 tw_quant/data_snapshot.py 的背景說明）。

獨立成自己的腳本，不跟 export_us_data_snapshot.py（價量+成分股，每日
排程都會跑）混在一起——避免每天跑的價量排程在 earnings ingestion 真的
執行過一次之前，就用空的 us_earnings 表覆寫掉這份快照。

要在 ingest_us_earnings_data.py 真的寫入資料庫「之後」才跑。

用法：
    python scripts/export_us_earnings_snapshot.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tw_quant.data_snapshot import US_EARNINGS_SNAPSHOT_PATH, export_us_earnings_snapshot
from tw_quant.storage import get_data_store


def main() -> None:
    store = get_data_store()
    n_rows = export_us_earnings_snapshot(store)
    print(f"已匯出 us_earnings {n_rows} 列 -> {US_EARNINGS_SNAPSHOT_PATH}")


if __name__ == "__main__":
    main()
