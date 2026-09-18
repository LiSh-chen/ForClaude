"""一次性診斷：改用 Nasdaq-100（QQQ 追蹤的指數）之前，先確認兩件事：
  1. 維基百科「Nasdaq-100」頁面目前成分股表格長什麼樣（欄位、檔數），
     有沒有跟 S&P 500 頁面一樣的 "Date added" 欄位。
  2. 有沒有一張真正可用的「歷史異動紀錄」表（加入/剔除日期+標的），
     這是換成 Nasdaq-100 是否真的能做「完整時點成分股名單」修正的關鍵——
     S&P 500 頁面的第二張表實測只是分產業列出現有成分股的導覽側欄，
     不是異動紀錄，Nasdaq-100 頁面不一定會不一樣，需要實際確認。

不動生產資料表，純粹確認資料來源可行性。

用法：
    python scripts/debug_ndx100_wikipedia_check.py
"""

from __future__ import annotations

import io

import pandas as pd
import requests

URL = "https://en.wikipedia.org/wiki/Nasdaq-100"


def main() -> None:
    headers = {"User-Agent": "Mozilla/5.0 (compatible; research-bot/1.0)"}
    resp = requests.get(URL, headers=headers, timeout=30)
    resp.raise_for_status()
    tables = pd.read_html(io.StringIO(resp.text))

    print(f"維基百科 Nasdaq-100 頁面共抓到 {len(tables)} 張表格\n")

    for i, t in enumerate(tables):
        cols = list(t.columns)
        print(f"--- tables[{i}]（{len(t)} 列）欄位 ---")
        print(cols)
        # 找看起來像是股票代號的欄位，判斷這張表是不是成分股/異動紀錄表
        col_str = " ".join(str(c).lower() for c in cols)
        looks_relevant = any(k in col_str for k in ["ticker", "symbol", "company", "date", "added", "removed"])
        if looks_relevant:
            print(t.head(8).to_string(index=False))
        print()

    # 嘗試自動判斷目前成分股表（欄位裡有 Ticker/Symbol 且列數接近100）
    candidate = None
    for t in tables:
        col_str = " ".join(str(c).lower() for c in t.columns)
        if ("ticker" in col_str or "symbol" in col_str) and 80 <= len(t) <= 120:
            candidate = t
            break
    if candidate is not None:
        print(f"=== 猜測這張是目前成分股表（{len(candidate)} 檔）===")
        print(list(candidate.columns))
        date_col = next((c for c in candidate.columns if "date" in str(c).lower() or "added" in str(c).lower()), None)
        print(f"是否有日期欄位：{date_col!r}")
    else:
        print("沒有自動判斷出明顯的目前成分股表，需要人工看上面每張表格的欄位跟內容。")


if __name__ == "__main__":
    main()
