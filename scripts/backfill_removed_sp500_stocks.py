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

冪等：upsert_us_prices_snapshot / upsert_us_index_membership_snapshot 都是
以主鍵覆寫合併，重複執行這個腳本不會產生重複資料，可以安全重跑。

2026-09-30 起不再經過 tw_quant/storage.py 的 DataStore/SQLite：直接讀寫
本機已 commit 的 Parquet 快照（見 tw_quant/data_snapshot.py 開頭
「2026-09-30 退役 SQLite」說明）。

用法：
    python scripts/backfill_removed_sp500_stocks.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tw_quant.data_snapshot import (
    load_us_prices_snapshot,
    upsert_us_index_membership_snapshot,
    upsert_us_prices_snapshot,
)
from tw_quant.sp500_history import (
    build_membership_intervals,
    fetch_snapshot_csv_text,
    find_missing_intervals,
    parse_snapshot_table,
)
from tw_quant.us_data_provider import YFinanceUSDataProvider

SLEEP_SECONDS = 1.0  # 對 Yahoo Finance 客氣一點，避免連續呼叫被暫時限速

# 2026-09-19 全樣本測試（test_yfinance_full_backfill_scan.py）裡，這兩檔
# 雖然 yfinance 有回應，但實際列數（22、10 列）跟區間長度（8 年）對不上，
# 原因未查清楚，明確排除、不寫入資料庫。
EXCLUDED_ANOMALOUS_TICKERS = {"AVB", "EQR"}


def backfill_one(provider: YFinanceUSDataProvider, row) -> dict:
    """試抓一檔股票的歷史股價。回傳結果摘要（成功時含抓到的 price_df /
    membership_row），獨立成函式方便測試（傳假的 provider 進來即可）。

    刻意不在這裡直接寫回 Parquet 快照：upsert_us_prices_snapshot 每次
    呼叫都要重寫整份快照（O(快照總列數)），每檔股票都個別呼叫一次會變成
    O(檔數 × 快照總列數)——main() 收集所有結果後只在最後合併寫入一次。
    """
    stock_id = row.stock_id
    try:
        df = provider.fetch_price(stock_id, str(row.clipped_start.date()), str(row.clipped_end.date()))
    except Exception as exc:  # noqa: BLE001 -- 任何一檔失敗都不該中斷其他檔
        return {
            "stock_id": stock_id, "written": False, "rows": 0, "error": f"{type(exc).__name__}: {exc}",
            "price_df": None, "membership_row": None,
        }

    if df.empty:
        return {"stock_id": stock_id, "written": False, "rows": 0, "error": None, "price_df": None, "membership_row": None}

    membership_row = pd.DataFrame(
        {"stock_id": [stock_id], "start_date": [row.start_date], "end_date": [row.end_date]}
    )

    return {"stock_id": stock_id, "written": True, "rows": len(df), "error": None, "price_df": df, "membership_row": membership_row}


def main() -> None:
    try:
        us_prices = load_us_prices_snapshot()
    except FileNotFoundError:
        us_prices = pd.DataFrame()
    if us_prices.empty:
        print("Parquet 快照裡沒有任何美股價量資料。", file=sys.stderr)
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
        result = backfill_one(provider, row)
        results.append(result)
        if result["written"]:
            print(f"[{i}/{n_stocks}] ✓ {row.stock_id}：抓到 {result['rows']} 列")
        else:
            note = f"（{result['error']}）" if result["error"] else "（查無資料，跳過）"
            print(f"[{i}/{n_stocks}] ✗ {row.stock_id} {note}")
        time.sleep(SLEEP_SECONDS)

    written = [r for r in results if r["written"]]
    skipped = [r for r in results if not r["written"]]

    if written:
        all_prices = pd.concat([r["price_df"] for r in written], ignore_index=True)
        all_membership = pd.concat([r["membership_row"] for r in written], ignore_index=True)
        n_prices_total = upsert_us_prices_snapshot(all_prices)
        n_membership_total = upsert_us_index_membership_snapshot(all_membership)
        print(
            f"\n已合併進 Parquet 快照：價量快照現在共 {n_prices_total} 列，"
            f"成分股區間快照現在共 {n_membership_total} 筆"
        )
    else:
        print("\n沒有任何股票成功抓到資料，快照維持不變。")

    print(f"\n=== 回填結果：{n_stocks} 檔裡，成功寫入 {len(written)} 檔、跳過 {len(skipped)} 檔 ===\n")
    print("成功寫入：")
    print("  " + "、".join(r["stock_id"] for r in written) if written else "  （無）")
    print("\n跳過（查無資料或例外，資料庫未變動）：")
    print("  " + "、".join(r["stock_id"] for r in skipped) if skipped else "  （無）")


if __name__ == "__main__":
    main()
