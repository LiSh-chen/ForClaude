"""一次性/低頻執行的美股財報公布資料回填腳本：從 yfinance 抓每檔股票的
財報公布日歷史（EPS 預期/實際/驚喜幅度），寫進資料庫的 us_earnings 表格。

背景：2026-09-19 用 scripts/probe_us_fundamentals_data.py 探路過 40 檔
樣本，yfinance 的 earnings_dates 資料涵蓋度 100%、平均可回溯約 10~12
年、公布日期幾乎都是真正的公布日（不是季末代理日期）——這是 PEAD（財報
驚喜漂移）策略可行的關鍵前提。跟 quarterly_income_stmt（只有約 5 季
深度，同一次探路量到的）完全不同，不要混淆，也不是這支腳本要抓的資料。

跟 ingest_us_daily_data.py 不同：這裡不做逐股增量水位判斷。財報公布
頻率是季度、資料量遠小於逐日價量，重新整批抓取一次的成本很低，每次
執行都對整份股票清單重新抓一次可用歷史、合併進快照——冪等，可以安全
重跑，不需要增量同步的複雜度。單檔抓取失敗時，該檔在快照裡的既有資料
會被保留（合併語意，不是整表覆寫），不會因為這次抓取失敗就消失。

股票清單直接讀本機的 us_prices_snapshot.parquet（不連資料庫撈整表），
確保跟目前策略回測腳本用的股票池一致。2026-09-30 起財報資料本身也不再
經過 tw_quant/storage.py 的 DataStore/SQLite，直接合併寫回 Parquet 快照
（見 tw_quant/data_snapshot.py 開頭「2026-09-30 退役 SQLite」說明）。

環境變數：
  REQUEST_SLEEP_SECONDS  每次 yfinance 呼叫間隔秒數，避免被 Yahoo 暫時
                         限速（預設 0.3）
  EARNINGS_LIMIT         每檔股票最多抓幾筆財報公布紀錄（預設 80，實際
                         能拿到多少受 Yahoo Finance 保留深度限制）

用法：
    python scripts/ingest_us_earnings_data.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tw_quant.data_snapshot import US_EARNINGS_SNAPSHOT_PATH, load_us_prices_snapshot, upsert_us_earnings_snapshot
from tw_quant.us_data_provider import YFinanceUSDataProvider


def main() -> None:
    sleep_s = float(os.environ.get("REQUEST_SLEEP_SECONDS", "0.3"))
    earnings_limit = int(os.environ.get("EARNINGS_LIMIT", "80"))

    us_prices = load_us_prices_snapshot()
    if us_prices.empty:
        print("快照裡沒有任何美股價量資料（data/us_prices_snapshot.parquet 是空的）。", file=sys.stderr)
        sys.exit(1)

    tickers = sorted(us_prices["stock_id"].unique())
    print(f"共 {len(tickers)} 檔股票要抓財報公布歷史（每檔間隔 {sleep_s} 秒，每檔最多 {earnings_limit} 筆）\n")

    provider = YFinanceUSDataProvider()

    total_rows = 0
    failures: list[str] = []
    new_frames: list[pd.DataFrame] = []
    for i, stock_id in enumerate(tickers, start=1):
        try:
            df = provider.fetch_earnings_history(stock_id, limit=earnings_limit)
            if not df.empty:
                new_frames.append(df)
                total_rows += len(df)
            print(f"[{i}/{len(tickers)}] {stock_id}: {len(df)} 筆")
        except Exception as exc:  # noqa: BLE001 -- 單一檔失敗不該中斷整個排程
            print(f"[warn] [{i}/{len(tickers)}] {stock_id} 抓取失敗: {exc}", file=sys.stderr)
            failures.append(stock_id)
        time.sleep(sleep_s)

    print(f"\n抓取完成，共 {total_rows} 筆，合併進 Parquet 快照...")
    if new_frames:
        n_total = upsert_us_earnings_snapshot(pd.concat(new_frames, ignore_index=True))
        print(f"已寫入美股財報公布資料 {total_rows} 筆，快照現在共 {n_total} 列 -> {US_EARNINGS_SNAPSHOT_PATH}")
    else:
        print("沒有新資料，快照維持不變。")
    if failures:
        print(f"[warn] {len(failures)} 檔抓取失敗: {failures}", file=sys.stderr)


if __name__ == "__main__":
    main()
