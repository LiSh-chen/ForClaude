"""一次性除錯腳本：probe_stooq_missing_sp500_stocks.py 對 320 檔全部回報
「查無資料」，連 TWTR（2022 年才下市，理論上歷史資料應該完整）都失敗，
這個 100% 失敗率太可疑，比較可能是 API 串接本身有問題（ticker 格式、
被限速/擋掉），不是 Stooq 真的對這批股票完全沒有資料。

這裡直接印出幾檔股票（含現役股票 AAPL 當作「一定要成功」的對照組）的
HTTP 回應原始內容（前 500 字元），藉此判斷問題出在哪一層：
  - 如果連 AAPL 都失敗/回應異常 -> API 串接方式本身有問題（URL 格式、
    被 Stooq 擋掉自動化請求等），不是資料涵蓋率問題
  - 如果 AAPL 成功但下市股票的回應是合法但空的 CSV -> 才是真的沒收錄

不寫入資料庫，跑完就刪除，不是常駐腳本。

用法：
    python scripts/debug_stooq_raw_response.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tw_quant.stooq_provider import STOOQ_CSV_URL, normalize_stooq_ticker

PROBE_TICKERS = ["AAPL", "MSFT", "TWTR", "STI", "LEHMQ"]


def main() -> None:
    import requests

    headers = {"User-Agent": "Mozilla/5.0 (compatible; research-bot/1.0)"}

    for ticker in PROBE_TICKERS:
        normalized = normalize_stooq_ticker(ticker)
        url = STOOQ_CSV_URL.format(ticker=normalized)
        print(f"=== {ticker} -> {normalized} ===")
        print(f"URL: {url}")
        try:
            resp = requests.get(url, headers=headers, timeout=30.0)
            print(f"status_code: {resp.status_code}")
            print(f"content-type: {resp.headers.get('content-type')}")
            print(f"body (前 500 字元):\n{resp.text[:500]!r}")
        except Exception as exc:  # noqa: BLE001
            print(f"例外: {type(exc).__name__}: {exc}")
        print()


if __name__ == "__main__":
    main()
