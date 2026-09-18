"""每日資料抓取排程腳本（美股版）：從 yfinance 抓 S&P 500 成分股的最新
價量資料，寫進資料庫的 us_prices 表格。

設計跟 scripts/ingest_daily_data.py（台股版）刻意保持同一套架構——逐股
判斷要回填還是增量同步、同一套 wall-clock timeout 保護——差異只在資料
來源跟不需要 API token 這件事。一樣沒辦法在這個開發沙盒裡直接跑（沙盒
網路白名單擋掉外部網站），要在 GitHub Actions runner 上執行。

跟台股版不同的地方：
  - 不需要 FINMIND_TOKEN，yfinance 不用申請 API key。
  - S&P 500 成分股清單是每次執行都即時從維基百科抓（見
    tw_quant/us_data_provider.py 的 fetch_sp500_constituents），不像台股
    版需要另外一支 generate_stock_universe.py 產生並 commit 清單檔——
    這份清單變動很少、抓取成本又低，沒有 FinMind 那種「當日額度」的
    顧慮，即時抓即用比維護一份另外的快取檔案簡單。
  - 沒有融資券/月營收/已發行股數這些台股特有的資料，只有價量。

環境變數：
  DATABASE_URL           雲端 Postgres 連線字串（跟台股版共用同一個資料庫，
                         美股資料存在獨立的 us_prices 表格，不會互相污染）
  SQLITE_DB_PATH         SQLite 檔案路徑（預設 data/tw_market.db，注意這是
                         跟台股共用同一個檔案，只是不同表格）
  LOOKBACK_DAYS          增量同步時往回抓幾天（預設 10）
  REQUEST_SLEEP_SECONDS  每次 yfinance 呼叫間隔秒數，避免被 Yahoo 暫時限速
                         （預設 0.3）
  FORCE_BACKFILL         設為 "true" 時忽略每檔股票現有資料，強制全部
                         重新回填

用法：
    python scripts/ingest_us_daily_data.py
"""

from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path
from typing import Callable, TypeVar

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from tw_quant.us_data_provider import YFinanceUSDataProvider
from tw_quant.storage import get_data_store

T = TypeVar("T")


def _call_with_timeout(func: Callable[[], T], timeout_s: float) -> T:
    """跟台股版 ingest_daily_data.py 同一套保護，理由見該檔案的說明：
    requests/yfinance 遇到伺服器慢速吐資料時，自己的 timeout 參數不一定
    有效，背景執行緒 + join(timeout=...) 才是真正保證上限。
    """
    result: list[T] = []
    error: list[BaseException] = []

    def _worker() -> None:
        try:
            result.append(func())
        except BaseException as exc:  # noqa: BLE001
            error.append(exc)

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()
    thread.join(timeout=timeout_s)

    if thread.is_alive():
        raise TimeoutError(f"呼叫逾時（{timeout_s} 秒），背景執行緒仍在等待回應")
    if error:
        raise error[0]
    return result[0]


def main() -> None:
    lookback_days = int(os.environ.get("LOOKBACK_DAYS", "10"))
    sleep_s = float(os.environ.get("REQUEST_SLEEP_SECONDS", "0.3"))
    force_backfill = os.environ.get("FORCE_BACKFILL", "").lower() == "true"

    provider = YFinanceUSDataProvider()
    store = get_data_store()

    print("抓取 S&P 500 成分股清單（維基百科）...")
    constituents = _call_with_timeout(provider.fetch_sp500_constituents, timeout_s=30.0)
    print(f"共 {len(constituents)} 檔成分股\n")

    end_date = pd.Timestamp.today().strftime("%Y-%m-%d")
    backfill_start = (pd.Timestamp.today() - pd.Timedelta(days=365 * 3)).strftime("%Y-%m-%d")

    total_rows = 0
    failures: list[str] = []

    for i, row in enumerate(constituents.itertuples(), start=1):
        stock_id = row.stock_id
        industry = row.industry
        try:
            stock_latest = None if force_backfill else store.latest_date("us_prices", stock_id=stock_id)
            if stock_latest is not None:
                start_date = (stock_latest - pd.Timedelta(days=lookback_days)).strftime("%Y-%m-%d")
            else:
                start_date = backfill_start

            price_df = _call_with_timeout(
                lambda sid=stock_id, s=start_date, e=end_date, ind=industry: provider.fetch_price(sid, s, e, ind),
                timeout_s=30.0,
            )
            if not price_df.empty:
                store.upsert_us_prices(price_df)
                total_rows += len(price_df)

            print(f"[{i}/{len(constituents)}] {stock_id}: {len(price_df)} 筆（{start_date} ~ {end_date}）")
        except Exception as exc:  # noqa: BLE001 - 單一檔失敗不該中斷整個排程
            print(f"[warn] [{i}/{len(constituents)}] {stock_id} 抓取失敗: {exc}", file=sys.stderr)
            failures.append(stock_id)
        time.sleep(sleep_s)

    print(f"\n完成。寫入美股價量 {total_rows} 筆。")
    if failures:
        print(f"[warn] {len(failures)} 檔抓取失敗: {failures}", file=sys.stderr)

    store.close()


if __name__ == "__main__":
    main()
