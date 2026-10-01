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

2026-09-30 起不再經過 tw_quant/storage.py 的 DataStore/SQLite：直接讀本機
已 commit 的 Parquet 快照判斷每檔股票的既有歷史、抓到新資料後直接合併寫回
快照（tw_quant/data_snapshot.py 的 upsert_us_prices_snapshot /
upsert_us_index_membership_snapshot），完全不碰資料庫，Parquet 快照本身
就是唯一的持久化來源（見該檔案開頭「2026-09-30 退役 SQLite」說明）。

環境變數：
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

from tw_quant.data_snapshot import (
    US_PRICES_SNAPSHOT_PATH,
    load_us_prices_snapshot,
    upsert_us_index_membership_snapshot,
    upsert_us_prices_snapshot,
)
from tw_quant.us_data_provider import YFinanceUSDataProvider

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
    backfill_years = int(os.environ.get("BACKFILL_YEARS", "8"))

    provider = YFinanceUSDataProvider()

    # 直接讀本機已 commit 的 Parquet 快照判斷「每檔股票既有資料到哪一天」，
    # 不經過 SQLite（見 tw_quant/data_snapshot.py 開頭「2026-09-30 退役
    # SQLite」說明）。快照不存在（全新 repo）時當作空歷史，全部當成首次
    # 回填。
    try:
        existing_prices = load_us_prices_snapshot()
    except FileNotFoundError:
        existing_prices = pd.DataFrame(columns=["date", "stock_id"])
    latest_by_stock: dict[str, pd.Timestamp] = (
        {}
        if force_backfill or existing_prices.empty
        else existing_prices.groupby("stock_id")["date"].max().to_dict()
    )
    print(
        f"既有快照：價量 {len(existing_prices)} 筆"
        f"（{existing_prices['stock_id'].nunique() if not existing_prices.empty else 0} 檔股票）\n"
    )

    print("抓取 S&P 500 成分股清單（維基百科）...")
    constituents = _call_with_timeout(provider.fetch_sp500_constituents, timeout_s=30.0)
    print(f"共 {len(constituents)} 檔成分股\n")

    # 存指數加入日期，供 tw_quant/us_universe.py 在回測時排除「當時還沒
    # 加入指數」的股票，部分修正存活者偏差（見該模組檔頭說明）。這裡每次
    # 執行都存，不受 FORCE_BACKFILL 影響——只是覆寫最新的加入日期紀錄，
    # 成本很低，不需要另外判斷要不要更新。
    membership = constituents[["stock_id", "date_added"]].rename(columns={"date_added": "start_date"})
    membership["end_date"] = pd.NaT  # 目前仍是成分股，還沒觀察到剔除日期（開放式區間）
    n_membership_total = upsert_us_index_membership_snapshot(membership)
    n_missing_date = membership["start_date"].isna().sum()
    print(
        f"已更新 {len(membership)} 檔的指數加入日期記錄（快照現在共 {n_membership_total} 筆區間，"
        f"{n_missing_date} 檔缺加入日期，回測時視為一直都在指數裡）\n"
    )

    end_date = pd.Timestamp.today().strftime("%Y-%m-%d")
    # 原本只回填 3 年，為了讓 RSI/布林通道均值回歸這種在美股上意外表現亮眼
    # 的策略（見 docs/research_findings.md 第10.3節）能做真正的樣本外驗證
    # （拿沒看過的更早期間跑同一組固定參數），這裡拉長到 8 年——多數 S&P 500
    # 成分股 yfinance 都能回溯到這麼久，個別較晚上市/加入指數的公司會自然
    # 從實際掛牌日開始，不會是錯誤。
    #
    # BACKFILL_YEARS 開放可調（預設仍是 8，不影響既有排程）：2026-09-20
    # 為了測試動能策略在 2008 金融風暴期間的表現，一次性拉長到 20 年，
    # 讓現有 577 檔成分股回溯到 2006 年左右，2008 危機當時還在暖身期
    # （min_history_days=252 + momentum_window=126）之前留出緩衝。注意
    # 這只解決「現有 577 檔股票有更早的股價資料」，不解決「雷曼兄弟、
    # Bear Stearns、Washington Mutual 這些 2008 危機當時就被剔除指數、
    # 現在完全不在 577 檔名單裡」的存活者偏差——那一半要靠另外重跑
    # scripts/backfill_removed_sp500_stocks.py（見 tw_quant/sp500_history.py）
    # 才可能補回一部分（且很多當年破產下市的公司 yfinance 本來就查無資料）。
    backfill_start = (pd.Timestamp.today() - pd.Timedelta(days=365 * backfill_years)).strftime("%Y-%m-%d")

    total_rows = 0
    failures: list[str] = []
    # 逐股抓到的新資料先收集在記憶體裡，迴圈跑完才一次性合併寫回 Parquet
    # 快照（upsert_us_prices_snapshot 每次呼叫都要重寫整份快照，630 檔
    # 股票每檔呼叫一次會變成 O(630 × 快照總列數)，跑一次要重寫快照 630
    # 次——不只慢，也完全沒必要：新資料不會在同一次執行裡互相覆蓋，合併
    # 一次就夠了）。
    new_price_frames: list[pd.DataFrame] = []

    for i, row in enumerate(constituents.itertuples(), start=1):
        stock_id = row.stock_id
        industry = row.industry
        try:
            stock_latest = latest_by_stock.get(stock_id)
            if stock_latest is not None:
                start_date = (stock_latest - pd.Timedelta(days=lookback_days)).strftime("%Y-%m-%d")
            else:
                start_date = backfill_start

            price_df = _call_with_timeout(
                lambda sid=stock_id, s=start_date, e=end_date, ind=industry: provider.fetch_price(sid, s, e, ind),
                timeout_s=30.0,
            )
            if not price_df.empty:
                new_price_frames.append(price_df)
                total_rows += len(price_df)

            print(f"[{i}/{len(constituents)}] {stock_id}: {len(price_df)} 筆（{start_date} ~ {end_date}）")
        except Exception as exc:  # noqa: BLE001 - 單一檔失敗不該中斷整個排程
            print(f"[warn] [{i}/{len(constituents)}] {stock_id} 抓取失敗: {exc}", file=sys.stderr)
            failures.append(stock_id)
        time.sleep(sleep_s)

    print(f"\n抓取完成，共 {total_rows} 筆新資料，合併進 Parquet 快照...")
    if new_price_frames:
        n_total = upsert_us_prices_snapshot(pd.concat(new_price_frames, ignore_index=True))
        print(f"已寫入美股價量 {total_rows} 筆，快照現在共 {n_total} 列 -> {US_PRICES_SNAPSHOT_PATH}")
    else:
        print("沒有新資料，快照維持不變。")
    if failures:
        print(f"[warn] {len(failures)} 檔抓取失敗: {failures}", file=sys.stderr)


if __name__ == "__main__":
    main()
