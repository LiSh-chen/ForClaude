"""把 scripts/list_sp500_removed_stocks_from_db.py 抓出來的完整缺漏名單
（177 檔，見 2026-09-19 對話紀錄）全部拿去 yfinance 試抓一次，而不是只測
scripts/test_yfinance_delisted_tickers.py 那 8 檔抽樣——8 檔抽樣已經看出
一個清楚的規律（仍在市場正常交易的股票抓得到，因併購/倒閉/私有化/改名
下市的股票完全抓不到），但只測 8 檔沒辦法給出「177 檔裡實際能補回幾檔」
這個真正需要的數字，這裡跑全樣本才能拿到誠實的實測結果。

跟 list_sp500_removed_stocks_from_db.py 共用同一套「誰缺資料」邏輯
（tw_quant.sp500_history.find_missing_intervals），確保這裡測的名單跟
先前列出來回報使用者的名單完全一致，不會兜不起來。

這仍然是診斷/測量腳本，不寫入資料庫——先看實際能補回幾檔、把清單交給
使用者確認，再決定要不要做正式的批次回填 + us_index_membership schema
擴充。

用法：
    python scripts/test_yfinance_full_backfill_scan.py
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
from tw_quant.us_data_provider import YFinanceUSDataProvider

SLEEP_SECONDS = 1.0  # 對 Yahoo Finance 客氣一點，避免連續呼叫被暫時限速


def fetch_one(provider: YFinanceUSDataProvider, stock_id: str, start_date: str, end_date: str) -> dict:
    """試抓一檔股票的歷史股價，回傳判讀用的結果摘要。獨立成函式方便測試
    格式化/摘要邏輯本身，不用真的連網（傳一個假的 provider 進來即可）。
    """
    try:
        df = provider.fetch_price(stock_id, start_date, end_date)
    except Exception as exc:  # noqa: BLE001 -- 177 檔裡任何一檔失敗都不該中斷其他檔
        return {"stock_id": stock_id, "rows": 0, "error": f"{type(exc).__name__}: {exc}"}
    if df.empty:
        return {"stock_id": stock_id, "rows": 0, "error": None}
    return {
        "stock_id": stock_id,
        "rows": len(df),
        "first": df["date"].min().date(),
        "last": df["date"].max().date(),
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
    print(f"共 {n_stocks} 檔缺漏股票要測試，每檔間隔 {SLEEP_SECONDS} 秒，預估耗時約 {n_stocks * (SLEEP_SECONDS + 1):.0f} 秒\n")

    provider = YFinanceUSDataProvider()
    results = []
    for i, row in enumerate(missing.itertuples(), start=1):
        result = fetch_one(provider, row.stock_id, str(row.clipped_start.date()), str(row.clipped_end.date()))
        results.append(result)
        if result["rows"] > 0:
            print(f"[{i}/{n_stocks}] ✓ {row.stock_id}：{result['rows']} 列，{result['first']} ~ {result['last']}")
        else:
            note = f"（{result['error']}）" if result["error"] else "（查無資料）"
            print(f"[{i}/{n_stocks}] ✗ {row.stock_id} {note}")
        time.sleep(SLEEP_SECONDS)

    ok = [r for r in results if r["rows"] > 0]
    fail = [r for r in results if r["rows"] == 0]

    print(f"\n=== 結果總覽：{n_stocks} 檔裡，抓得到 {len(ok)} 檔、抓不到 {len(fail)} 檔 ===\n")

    print(f"抓得到的 {len(ok)} 檔（下一步可以考慮真的補進資料庫）：")
    print("  " + "、".join(r["stock_id"] for r in ok) if ok else "  （無）")

    print(f"\n抓不到的 {len(fail)} 檔（yfinance 這條路走不通，需要另找資料源或放棄）：")
    print("  " + "、".join(r["stock_id"] for r in fail) if fail else "  （無）")

    print(
        "\n（誠實揭露：這是對真實 yfinance API 的一次性全樣本測試，不是每天都會重跑的\n"
        "自動化流程；「抓不到」有可能是 Yahoo Finance 當下暫時性問題，但根據先前 8 檔\n"
        "抽樣的規律（因併購/倒閉/私有化/改名下市的股票完全抓不到、只有仍在市場正常\n"
        "交易的股票抓得到），大部分「抓不到」應該是永久性的，不是暫時性的。）"
    )


if __name__ == "__main__":
    main()
