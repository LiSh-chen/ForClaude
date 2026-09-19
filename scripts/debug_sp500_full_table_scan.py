"""一次性診斷：完整列出維基百科「List of S&P 500 companies」頁面上
「所有」表格的欄位跟列數，不只看前兩個。用完即刪。

背景：先前的調查（docs/research_findings.md 10.5節的存活者偏差修正
過程）只檢查了 tables[0]（目前成分股，有 Date added 欄位）跟
tables[1]（確認是 GICS 產業導覽側邊欄），沒有系統性列出頁面上後續
所有表格——如果真的有一個「Selected changes to the list of S&P 500
components」這種歷史異動紀錄表（記錄哪天哪檔被剔除、哪檔被加入），
很可能是在更後面的索引，不是 tables[1]。這裡把整個頁面所有表格的
欄位名稱、列數、前 3 列內容都印出來，一次性判斷清楚，不用再靠猜的。

用法：
    python scripts/debug_sp500_full_table_scan.py
"""

from __future__ import annotations

import io
import sys

import pandas as pd
import requests


def main() -> None:
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {"User-Agent": "Mozilla/5.0 (compatible; research-bot/1.0)"}
    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()
    tables = pd.read_html(io.StringIO(resp.text))

    print(f"頁面共有 {len(tables)} 個表格\n")
    for i, t in enumerate(tables):
        cols = list(t.columns)
        print(f"=== tables[{i}]：{len(t)} 列，欄位：{cols} ===")
        if len(t) > 0:
            print(t.head(3).to_string())
        print()


if __name__ == "__main__":
    main()
