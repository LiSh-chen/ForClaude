"""正式批次回填：把 scripts/test_yfinance_full_backfill_scan.py 驗證過
「yfinance 抓得到」的被剔除 S&P 500 股票，真的寫進資料庫——這是
tw_quant/us_universe.py 存活者偏差修正的「被剔除」半邊，第一次真的補資料
進去，不只是列名單或測試連通性。

背景：177 檔缺漏股票裡，全樣本測試量出 76 檔 yfinance 抓得到歷史股價。
其中 AVB、EQR 這兩檔雖然回報「抓到資料」，但實際列數（22 列、10 列）跟
它們的區間長度（8 年）完全對不上、原因未查清楚（見 2026-09-19 對話紀錄），
這裡明確排除、不寫入資料庫，避免用可疑資料污染回測結果——寧可少補兩檔、
不要補進錯的。

寫入兩張表：
  1. us_prices：這檔股票的歷史股價（只抓 clipped_start~clipped_end，也就是
     資料庫既有資料涵蓋期間內的部分，不抓超出這個範圍的歷史）
  2. us_index_membership：這檔股票在指數裡的區間，用未裁切的真實
     start_date/end_date（不是 clipped_start/clipped_end）——這樣
     filter_prices_by_index_membership 看到的是真實加入/剔除日期，不是
     被資料庫涵蓋範圍裁切過的日期（即使某檔股票真實加入日期早於資料庫
     最早的資料，也不影響過濾結果，因為資料庫本來就不會有那之前的價量
     資料）

冪等：upsert_us_prices / upsert_us_index_membership 都是以主鍵覆寫，
重複執行這個腳本不會產生重複資料，可以安全重跑。

用法：
    python scripts/backfill_removed_sp500_stocks.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tw_quant.sp500_history import (
    build_membership_intervals,
    fetch_snapshot_csv_text,
    find_missing_intervals,
    parse_snapshot_table,
)
from tw_quant.storage import get_data_store
from tw_quant.us_data_provider import YFinanceUSDataProvider

SLEEP_SECONDS = 1.0  # 對 Yahoo Finance 客氣一點，避免連續呼叫被暫時限速

# 2026-09-19 全樣本測試（test_yfinance_full_backfill_scan.py）裡，這兩檔
# 雖然 yfinance 有回應，但實際列數（22、10 列）跟區間長度（8 年）對不上，
# 原因未查清楚，明確排除、不寫入資料庫。
EXCLUDED_ANOMALOUS_TICKERS = {"AVB", "EQR"}


def backfill_one(store, provider: YFinanceUSDataProvider, row) -> dict:
    """試抓一檔股票的歷史股價，抓得到才寫入 us_prices + us_index_membership。
    回傳結果摘要，獨立成函式方便測試（傳假的 store/provider 進來即可）。
    """
    stock_id = row.stock_id
    try:
        df = provider.fetch_price(stock_id, str(row.clipped_start.date()), str(row.clipped_end.date()))
    except Exception as exc:  # noqa: BLE001 -- 任何一檔失敗都不該中斷其他檔
        return {"stock_id": stock_id, "written": False, "rows": 0, "error": f"{type(exc).__name__}: {exc}"}

    if df.empty:
        return {"stock_id": stock_id, "written": False, "rows": 0, "error": None}

    store.upsert_us_prices(df)

    membership_row = pd.DataFrame(
        {"stock_id": [stock_id], "start_date": [row.start_date], "end_date": [row.end_date]}
    )
    store.upsert_us_index_membership(membership_row)

    return {"stock_id": stock_id, "written": True, "rows": len(df), "error": None}


def main() -> None:
    store = get_data_store()
    us_prices = store.load_us_prices()
    if us_prices.empty:
        print("資料庫裡沒有任何美股價量資料。", file=sys.stderr)
        sys.exit(1)

    db_stock_ids = set(us_prices["stock_id"].unique())
    window_start = us_prices["date"].min()
    window_end = us_prices["date"].max()

    print("抓取 fja05680/sp500 的歷史成分股快照 CSV ...")
    csv_text = fetch_snapshot_csv_text()
    snapshot = parse_snapshot_table(csv_text)
    intervals = build_membership_intervals(snapshot)
    missing = find_missing_intervals(db_stock_ids, intervals, window_start, window_end)
    missing = missing[~missing["stock_id"].isin(EXCLUDED_ANOMALOUS_TICKERS)]

    n_stocks = missing["stock_id"].nunique()
    print(
        f"共 {n_stocks} 檔缺漏股票要回填（已排除 {sorted(EXCLUDED_ANOMALOUS_TICKERS)} 這 2 檔異常結果），"
        f"每檔間隔 {SLEEP_SECONDS} 秒\n"
    )

    provider = YFinanceUSDataProvider()
    results = []
    for i, row in enumerate(missing.itertuples(), start=1):
        result = backfill_one(store, provider, row)
        results.append(result)
        if result["written"]:
            print(f"[{i}/{n_stocks}] ✓ {row.stock_id}：寫入 {result['rows']} 列")
        else:
            note = f"（{result['error']}）" if result["error"] else "（查無資料，跳過）"
            print(f"[{i}/{n_stocks}] ✗ {row.stock_id} {note}")
        time.sleep(SLEEP_SECONDS)

    written = [r for r in results if r["written"]]
    skipped = [r for r in results if not r["written"]]

    print(f"\n=== 回填結果：{n_stocks} 檔裡，成功寫入 {len(written)} 檔、跳過 {len(skipped)} 檔 ===\n")
    print("成功寫入：")
    print("  " + "、".join(r["stock_id"] for r in written) if written else "  （無）")
    print("\n跳過（查無資料或例外，資料庫未變動）：")
    print("  " + "、".join(r["stock_id"] for r in skipped) if skipped else "  （無）")


if __name__ == "__main__":
    main()
