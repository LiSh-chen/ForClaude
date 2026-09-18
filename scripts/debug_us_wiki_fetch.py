"""臨時診斷腳本：查 fetch_sp500_constituents() 實際失敗在哪一步。

us_daily_data_ingest.yml 第一次真的跑，"Run US ingestion" 這個 step
失敗了，但 log 裡塞滿了維基百科頁面本身的完整 HTML（好幾百 KB），把
真正的錯誤訊息擠出 log 尾端看不到。這支腳本刻意把每一步的輸出都截斷
成幾百字元以內，確保看得到真正的錯誤而不會被巨大的 HTML 蓋掉。

用完即刪，不是常駐腳本。
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import requests


def main() -> None:
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {"User-Agent": "Mozilla/5.0 (compatible; research-bot/1.0)"}

    print("[1] 發送請求...")
    resp = requests.get(url, headers=headers, timeout=30)
    print(f"[1] status_code={resp.status_code}, len(resp.text)={len(resp.text)}")
    resp.raise_for_status()

    print("[2] pd.read_html(resp.text)...")
    try:
        tables = pd.read_html(resp.text)
    except Exception as exc:  # noqa: BLE001
        print(f"[2] read_html 失敗: {type(exc).__name__}: {str(exc)[:300]}")
        raise SystemExit(1)

    print(f"[2] 共抓到 {len(tables)} 張表格")
    for i, t in enumerate(tables[:3]):
        print(f"[2]   table[{i}] shape={t.shape} columns={list(t.columns)[:10]}")

    print("[3] 檢查 table[0] 欄位...")
    t0 = tables[0]
    print(f"[3] table[0] columns = {list(t0.columns)}")
    print(f"[3] table[0] head:\n{t0.head(3).to_string()[:1000]}")

    print("[4] parse_sp500_wikipedia_table(tables[0])...")
    from tw_quant.us_data_provider import parse_sp500_wikipedia_table

    try:
        parsed = parse_sp500_wikipedia_table(t0)
        print(f"[4] 成功，共 {len(parsed)} 檔，前 5 檔:\n{parsed.head().to_string()}")
    except Exception as exc:  # noqa: BLE001
        print(f"[4] 失敗: {type(exc).__name__}: {str(exc)[:300]}")
        raise SystemExit(1)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001
        print(f"[FATAL] {type(exc).__name__}: {str(exc)[:500]}")
        tb_lines = traceback.format_exc().splitlines()
        print("[FATAL traceback 最後 15 行]")
        for line in tb_lines[-15:]:
            print(line[:300])
        raise SystemExit(1)
