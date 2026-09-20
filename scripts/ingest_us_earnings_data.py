"""一次性/低頻執行的美股財報公布資料回填腳本：從 yfinance 抓每檔股票的
財報公布日歷史（EPS 預期/實際/驚喜幅度），寫進資料庫的 us_earnings 表格。

背景：2026-09-19 用 scripts/probe_us_fundamentals_data.py 探路過 40 檔
樣本，yfinance 的 earnings_dates 資料涵蓋度 100%、平均可回溯約 10~12
年、公布日期幾乎都是真正的公布日（不是季末代理日期）——這是 PEAD（財報
驚喜漂移）策略可行的關鍵前提。跟 quarterly_income_stmt（只有約 5 季
深度，同一次探路量到的）完全不同，不要混淆，也不是這支腳本要抓的資料。

跟 ingest_us_daily_data.py 不同：這裡不做逐股增量水位判斷。財報公布
頻率是季度、資料量遠小於逐日價量，重新整批抓取一次的成本很低，每次
執行都對整份股票清單重新抓一次可用歷史、upsert 覆蓋——冪等，可以安全
重跑，不需要增量同步的複雜度。

股票清單直接讀本機的 us_prices_snapshot.parquet（不連資料庫撈整表），
確保跟目前策略回測腳本用的股票池一致，也避免佔用 Neon 免費方案的網路
傳出流量額度（見 tw_quant/data_snapshot.py 開頭的背景說明）。

環境變數：
  DATABASE_URL           雲端 Postgres 連線字串（跟其他美股資料共用同一個
                         資料庫，存在獨立的 us_earnings 表格，互不污染）
  SQLITE_DB_PATH         SQLite 檔案路徑（預設 data/tw_market.db）
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tw_quant.data_snapshot import load_us_prices_snapshot
from tw_quant.storage import get_data_store
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
    store = get_data_store()

    total_rows = 0
    failures: list[str] = []
    for i, stock_id in enumerate(tickers, start=1):
        try:
            df = provider.fetch_earnings_history(stock_id, limit=earnings_limit)
            if not df.empty:
                store.upsert_us_earnings(df)
                total_rows += len(df)
            print(f"[{i}/{len(tickers)}] {stock_id}: {len(df)} 筆")
        except Exception as exc:  # noqa: BLE001 -- 單一檔失敗不該中斷整個排程
            print(f"[warn] [{i}/{len(tickers)}] {stock_id} 抓取失敗: {exc}", file=sys.stderr)
            failures.append(stock_id)
        time.sleep(sleep_s)

    print(f"\n完成。寫入美股財報公布資料 {total_rows} 筆。")
    if failures:
        print(f"[warn] {len(failures)} 檔抓取失敗: {failures}", file=sys.stderr)

    store.close()


if __name__ == "__main__":
    main()
