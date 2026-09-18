"""臨時診斷腳本：查 yfinance 能不能提供美股版 PEAD/股本效應需要的基本面
資料（月/季營收、已發行股數/市值），評估任務 #48（美股營收/市值資料
管線）的可行性。用完即刪，不是常駐腳本。

查兩個東西：
  1. Ticker.get_shares_full(start, end)：歷史已發行股數（台股版股本效應
     用的是「回測期間開始時的股本」，需要歷史數字，不能只有現在的快照）。
  2. Ticker.quarterly_income_stmt：季營收（台股版 PEAD 用的是月營收年增率
     意外程度，美股沒有月營收揭露慣例，改用季營收是最接近的替代）。

用法：
    python scripts/debug_us_fundamentals_availability.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yfinance as yf

SAMPLE_TICKERS = ["AAPL", "JPM", "IEX", "ECHO"]  # 大型股 + 兩檔本輪回測交易過的中型股


def main() -> None:
    for ticker in SAMPLE_TICKERS:
        print(f"\n{'=' * 60}\n{ticker}\n{'=' * 60}")
        t = yf.Ticker(ticker)

        print("[1] get_shares_full（歷史已發行股數）...")
        try:
            shares = t.get_shares_full(start="2018-01-01", end="2026-09-17")
            if shares is None or shares.empty:
                print("  -> 空的")
            else:
                print(f"  -> {len(shares)} 筆，時間範圍 {shares.index.min()} ~ {shares.index.max()}")
                print(f"  -> 前 3 筆:\n{shares.head(3)}")
                print(f"  -> 後 3 筆:\n{shares.tail(3)}")
        except Exception as exc:  # noqa: BLE001
            print(f"  -> 失敗: {type(exc).__name__}: {str(exc)[:200]}")

        print("\n[2] quarterly_income_stmt（季損益表，查 Total Revenue）...")
        try:
            stmt = t.quarterly_income_stmt
            if stmt is None or stmt.empty:
                print("  -> 空的")
            else:
                print(f"  -> columns（季別）: {list(stmt.columns)}")
                revenue_rows = [idx for idx in stmt.index if "revenue" in str(idx).lower()]
                print(f"  -> 營收相關列: {revenue_rows}")
                if revenue_rows:
                    print(f"  -> {revenue_rows[0]}:\n{stmt.loc[revenue_rows[0]]}")
        except Exception as exc:  # noqa: BLE001
            print(f"  -> 失敗: {type(exc).__name__}: {str(exc)[:200]}")

        print("\n[3] fast_info market cap（現在快照，僅供對照）...")
        try:
            mc = t.fast_info.get("marketCap") if hasattr(t.fast_info, "get") else t.fast_info.market_cap
            print(f"  -> {mc}")
        except Exception as exc:  # noqa: BLE001
            print(f"  -> 失敗: {type(exc).__name__}: {str(exc)[:200]}")


if __name__ == "__main__":
    main()
