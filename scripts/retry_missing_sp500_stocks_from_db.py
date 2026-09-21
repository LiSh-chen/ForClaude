"""對 scripts/backfill_removed_sp500_stocks.py 仍抓不到資料的股票，換一種
查詢方式重試一次——用 period="max"（要「yfinance 有的全部歷史」）取代原本
指定 start_date/end_date 的查法，再篩選出需要的日期範圍。

背景：2026-09-20 正式回填時，364 檔缺漏股票裡只有 52 檔成功，其餘 320 檔
查無資料，錯誤訊息大致三種：
  - "possibly delisted; no timezone found"：yfinance 完全不認得這個代號
  - "Data doesn't exist for startDate=X, endDate=Y"：yfinance 認得這檔，
    但指定的查詢窗口內沒有資料
  - "possibly no price data found (1d start -> end)"：跟上面類似

第二、三種有可能只是查詢窗口跟 yfinance 實際記錄的掛牌區間對不上（例如
fja05680/sp500 快照認定的加入/剔除日期，跟 yfinance 資料庫實際收錄的
交易區間有落差），不代表 yfinance 真的完全沒有這檔股票的資料——用
period="max" 直接要全部歷史，繞開「查詢窗口設錯」這個可能的假陰性。
第一種（完全不認得代號）通常代表公司真的下市已久、yfinance 資料庫裡
從來沒收錄過，period="max" 大概率還是拿不到，但一併嘗試不會有額外成本
（同一次 API 呼叫）。

這裡刻意誠實：這支腳本能補到的，本來就只是「查詢方式問題」造成的假陰性，
不會、也不可能讓 Lehman Brothers、Bear Stearns、Washington Mutual 這些
真正破產下市、股價歸零的公司出現在資料庫裡——那些公司的股票交易紀錄在
下市後就不會再有任何資料源持續收錄「正常股價」，這不是查詢方式能解決的
問題（見 test_us_momentum_2008_crisis_from_db.py 檔頭的完整討論）。

冪等：跟 backfill_removed_sp500_stocks.py 一樣用 upsert，可以安全重跑。

用法：
    python scripts/retry_missing_sp500_stocks_from_db.py
"""

from __future__ import annotations

import sys
import time
from collections import Counter
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

SLEEP_SECONDS = 1.0

# 沿用 backfill_removed_sp500_stocks.py 的排除清單：yfinance 有回應但列數
# 跟區間長度對不上、原因未查清楚，不管用哪種查詢方式都不信任這兩檔。
EXCLUDED_ANOMALOUS_TICKERS = {"AVB", "EQR"}


def retry_one(store, provider: YFinanceUSDataProvider, row) -> dict:
    """用 period="max" 重試一檔股票，篩選出 clipped_start~clipped_end
    範圍內的資料才寫入。回傳結果摘要，獨立成函式方便測試。
    """
    stock_id = row.stock_id
    try:
        full_history = provider.fetch_price_full_history(stock_id)
    except Exception as exc:  # noqa: BLE001
        return {"stock_id": stock_id, "written": False, "rows": 0, "error": f"{type(exc).__name__}: {exc}"}

    if full_history.empty:
        return {"stock_id": stock_id, "written": False, "rows": 0, "error": None}

    windowed = full_history[
        (full_history["date"] >= row.clipped_start) & (full_history["date"] <= row.clipped_end)
    ]
    if windowed.empty:
        return {
            "stock_id": stock_id, "written": False, "rows": 0,
            "error": f"period=max 抓到 {len(full_history)} 列但都落在查詢窗口外（{row.clipped_start.date()}~{row.clipped_end.date()}）",
        }

    store.upsert_us_prices(windowed)
    membership_row = pd.DataFrame({"stock_id": [stock_id], "start_date": [row.start_date], "end_date": [row.end_date]})
    store.upsert_us_index_membership(membership_row)

    return {"stock_id": stock_id, "written": True, "rows": len(windowed), "error": None}


def _error_category(error: str | None) -> str:
    if error is None:
        return "查無資料（無例外，空表）"
    if "落在查詢窗口外" in error:
        return "抓到資料但不在需要的日期範圍內"
    if "delisted" in error or "timezone" in error:
        return "yfinance 完全不認得這個代號"
    if "Data doesn't exist" in error or "no price data found" in error:
        return "yfinance 認得但該窗口無資料"
    return "其他例外"


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
    print(f"共 {n_stocks} 檔仍缺漏股票要重試（period=\"max\"），每檔間隔 {SLEEP_SECONDS} 秒\n")

    provider = YFinanceUSDataProvider()
    results = []
    for i, row in enumerate(missing.itertuples(), start=1):
        result = retry_one(store, provider, row)
        results.append(result)
        if result["written"]:
            print(f"[{i}/{n_stocks}] ✓ {row.stock_id}：新增寫入 {result['rows']} 列")
        else:
            note = f"（{result['error']}）" if result["error"] else "（查無資料，跳過）"
            print(f"[{i}/{n_stocks}] ✗ {row.stock_id} {note}")
        time.sleep(SLEEP_SECONDS)

    written = [r for r in results if r["written"]]
    skipped = [r for r in results if not r["written"]]

    print(f"\n=== 重試結果：{n_stocks} 檔裡，新補上 {len(written)} 檔、依然查無資料 {len(skipped)} 檔 ===\n")
    print("新補上：")
    print("  " + "、".join(r["stock_id"] for r in written) if written else "  （無）")

    categories = Counter(_error_category(r["error"]) for r in skipped)
    print("\n依然查無資料，依原因分類：")
    for category, count in categories.most_common():
        print(f"  {category}：{count} 檔")

    print("\n依然查無資料的完整清單：")
    print("  " + "、".join(r["stock_id"] for r in skipped) if skipped else "  （無）")


if __name__ == "__main__":
    main()
