"""探路腳本：測試 Stooq 對 yfinance 兩輪嘗試後仍完全查無資料的 2008 年代
被剔除 S&P 500 成分股，有沒有更完整的收錄。純診斷、不寫入資料庫——先看
實際能補回幾檔、資料涵蓋的日期範圍夠不夠用，再決定要不要做正式整合
（見 tw_quant/stooq_provider.py 檔頭的完整背景說明）。

跟 list_sp500_removed_stocks_from_db.py / test_yfinance_full_backfill_scan.py
共用同一套「誰缺資料」邏輯（tw_quant.sp500_history.find_missing_intervals），
確保這裡測的名單跟資料庫目前實際缺漏的股票完全一致。

用法：
    python scripts/probe_stooq_missing_sp500_stocks.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tw_quant.sp500_history import (
    build_membership_intervals,
    fetch_snapshot_csv_text,
    find_missing_intervals,
    parse_snapshot_table,
)
from tw_quant.storage import get_data_store
from tw_quant.stooq_provider import fetch_stooq_price

SLEEP_SECONDS = 1.0  # 對 Stooq 客氣一點，避免連續呼叫被暫時限速


def probe_one(stock_id: str, needed_start, needed_end, fetch_fn=fetch_stooq_price) -> dict:
    """試抓一檔股票的 Stooq 歷史股價，回傳判讀用的結果摘要（是否有資料、
    是否覆蓋到需要的日期範圍）。fetch_fn 開放覆寫方便測試，傳假的函式
    進來即可，不用真的連網。
    """
    try:
        df = fetch_fn(stock_id)
    except Exception as exc:  # noqa: BLE001 -- 任何一檔失敗都不該中斷其他檔
        return {"stock_id": stock_id, "rows": 0, "covers_window": False, "error": f"{type(exc).__name__}: {exc}"}

    if df.empty:
        return {"stock_id": stock_id, "rows": 0, "covers_window": False, "error": None}

    windowed = df[(df["date"] >= needed_start) & (df["date"] <= needed_end)]
    return {
        "stock_id": stock_id,
        "rows": len(df),
        "rows_in_window": len(windowed),
        "covers_window": not windowed.empty,
        "earliest": df["date"].min(),
        "latest": df["date"].max(),
        "error": None,
    }


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

    n_stocks = missing["stock_id"].nunique()
    print(f"共 {n_stocks} 檔仍缺漏股票要用 Stooq 探路，每檔間隔 {SLEEP_SECONDS} 秒\n")

    results = []
    for i, row in enumerate(missing.itertuples(), start=1):
        result = probe_one(row.stock_id, row.clipped_start, row.clipped_end)
        results.append(result)
        if result.get("covers_window"):
            print(
                f"[{i}/{n_stocks}] ✓ {row.stock_id}：Stooq 有 {result['rows']} 列"
                f"（{result['earliest'].date()}~{result['latest'].date()}），"
                f"落在需要窗口內 {result['rows_in_window']} 列"
            )
        elif result["rows"] > 0:
            print(
                f"[{i}/{n_stocks}] ~ {row.stock_id}：Stooq 有 {result['rows']} 列"
                f"（{result['earliest'].date()}~{result['latest'].date()}），但不落在需要窗口"
                f"（{row.clipped_start.date()}~{row.clipped_end.date()}）內"
            )
        else:
            note = f"（{result['error']}）" if result["error"] else "（查無資料）"
            print(f"[{i}/{n_stocks}] ✗ {row.stock_id} {note}")
        time.sleep(SLEEP_SECONDS)

    covers = [r for r in results if r.get("covers_window")]
    partial = [r for r in results if not r.get("covers_window") and r["rows"] > 0]
    empty = [r for r in results if r["rows"] == 0]

    print(f"\n=== Stooq 探路結果：{n_stocks} 檔裡 ===")
    print(f"  完整覆蓋需要的日期範圍：{len(covers)} 檔")
    print(f"  有資料但不覆蓋需要範圍：{len(partial)} 檔")
    print(f"  完全查無資料：{len(empty)} 檔\n")

    print("完整覆蓋（可以考慮正式整合寫入資料庫）：")
    print("  " + "、".join(r["stock_id"] for r in covers) if covers else "  （無）")
    print("\n有資料但範圍對不上（跟 yfinance period=max 重試遇到的情況一樣，")
    print("要小心是不是代號後來被別家公司重複使用，不建議直接採信）：")
    print("  " + "、".join(r["stock_id"] for r in partial) if partial else "  （無）")


if __name__ == "__main__":
    main()
