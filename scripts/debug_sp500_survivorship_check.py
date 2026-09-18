"""一次性診斷：量化美股回測目前的存活者偏差有多嚴重。

現況（tw_quant/us_data_provider.py 的 fetch_sp500_constituents）：每次執行
都即時抓維基百科「List of S&P 500 companies」頁面的第一張表（目前的
503 檔成分股），回填 8 年歷史價量——等於用「今天還在指數裡的公司」的
名單去回測過去 8 年，任何在這段期間被剔除（下市/被併購/表現太差被踢出）
的公司完全不在資料庫裡。這對動量策略尤其危險：動量策略買最近漲最多的
股票，而「今天還在指數裡」這個篩選條件本身就已經是一種事後諸葛的贏家
篩選，兩者疊加可能嚴重高估報酬。

這裡只做診斷、不動生產資料表：
  1. 看 tables[0]（目前成分股）有沒有 "Date added" 欄位——如果有，能看出
     現在 503 檔裡有幾檔是最近幾年才新加入指數的（用來量化「新進戶」
     偏差）。
  2. 看 tables[1]（"Selected changes to the list of S&P 500 components"，
     維基百科通常會有這張表）有沒有抓到，列出 2018 年以後的加入/剔除
     紀錄筆數，量化「剔除者流失」偏差的規模。

用法：
    python scripts/debug_sp500_survivorship_check.py
"""

from __future__ import annotations

import io

import pandas as pd
import requests

URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"


def main() -> None:
    headers = {"User-Agent": "Mozilla/5.0 (compatible; research-bot/1.0)"}
    resp = requests.get(URL, headers=headers, timeout=30)
    resp.raise_for_status()
    tables = pd.read_html(io.StringIO(resp.text))

    print(f"維基百科頁面共抓到 {len(tables)} 張表格\n")

    current = tables[0]
    print(f"=== tables[0]（目前成分股，共 {len(current)} 檔）欄位 ===")
    print(list(current.columns))
    print()

    date_col = next((c for c in current.columns if "date" in c.lower() or "added" in c.lower()), None)
    if date_col:
        print(f"找到日期欄位：{date_col!r}，前5筆範例：")
        print(current[["Symbol", date_col]].head(5).to_string(index=False))
        print()
        parsed = pd.to_datetime(current[date_col], errors="coerce")
        for cutoff_label, cutoff in [("2018-09-20（8年回填起點）", "2018-09-20"), ("2023-09-19（3年樣本內起點）", "2023-09-19")]:
            n_after = (parsed >= pd.Timestamp(cutoff)).sum()
            n_unknown = parsed.isna().sum()
            print(f"{date_col} >= {cutoff_label} 的檔數：{n_after} / {len(current)}（另有 {n_unknown} 筆日期解析失敗/缺值）")
        print()
    else:
        print("沒有找到日期欄位，tables[0] 無法直接看出新進戶時間。\n")

    if len(tables) > 1:
        changes = tables[1]
        print(f"=== tables[1]（異動紀錄）欄位 ===")
        print(list(changes.columns))
        print(f"共 {len(changes)} 筆原始紀錄")
        print(changes.head(10).to_string(index=False))
        print()

        # 嘗試找日期欄位（通常是 MultiIndex 或 "Date" 開頭）
        date_cols = [c for c in changes.columns if "date" in str(c).lower()]
        if date_cols:
            dc = date_cols[0]
            parsed_changes = pd.to_datetime(changes[dc], errors="coerce")
            n_since_2018 = (parsed_changes >= pd.Timestamp("2018-09-20")).sum()
            n_since_2023 = (parsed_changes >= pd.Timestamp("2023-09-19")).sum()
            print(f"2018-09-20 之後的異動紀錄筆數：{n_since_2018}")
            print(f"2023-09-19 之後的異動紀錄筆數：{n_since_2023}")
        else:
            print("異動紀錄表格式跟預期不同，欄位裡沒有明顯的日期欄——需要手動看上面印出的欄位名稱跟前10筆內容判斷怎麼解析。")
    else:
        print("維基百科頁面沒有第二張表，或格式跟預期不同——可能需要換別的資料來源才能拿到剔除紀錄。")


if __name__ == "__main__":
    main()
