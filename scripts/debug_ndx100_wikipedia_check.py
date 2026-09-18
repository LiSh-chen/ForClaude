"""一次性診斷：改用 Nasdaq-100（QQQ 追蹤的指數）之前，先確認兩件事：
  1. 維基百科「List of NASDAQ-100 companies」頁面（成分股清單，跟
     「Nasdaq-100」指數介紹頁是分開的頁面——第一次診斷抓錯了介紹頁，
     那頁只有指數里程碑/歷史高點資料，沒有成分股表，這裡改抓正確頁面）
     目前成分股表格長什麼樣（欄位、檔數），有沒有跟 S&P 500 頁面一樣的
     "Date added" 欄位。
  2. 有沒有一張真正可用的「歷史異動紀錄」表（加入/剔除日期+標的），
     這是換成 Nasdaq-100 是否真的能做「完整時點成分股名單」修正的關鍵。

不動生產資料表，純粹確認資料來源可行性。

用法：
    python scripts/debug_ndx100_wikipedia_check.py
"""

from __future__ import annotations

import io

import pandas as pd
import requests

LIST_URL = "https://en.wikipedia.org/wiki/List_of_NASDAQ-100_companies"


def main() -> None:
    headers = {"User-Agent": "Mozilla/5.0 (compatible; research-bot/1.0)"}
    resp = requests.get(LIST_URL, headers=headers, timeout=30)
    resp.raise_for_status()
    tables = pd.read_html(io.StringIO(resp.text))

    print(f"維基百科 {LIST_URL} 頁面共抓到 {len(tables)} 張表格\n")

    for i, t in enumerate(tables):
        cols = list(t.columns)
        print(f"--- tables[{i}]（{len(t)} 列）欄位 ---")
        print(cols)
        col_str = " ".join(str(c).lower() for c in cols)
        looks_relevant = any(k in col_str for k in ["ticker", "symbol", "company", "date", "added", "removed", "effective"])
        if looks_relevant or (30 <= len(t) <= 150):
            print(t.head(6).to_string(index=False))
        print()


if __name__ == "__main__":
    main()
