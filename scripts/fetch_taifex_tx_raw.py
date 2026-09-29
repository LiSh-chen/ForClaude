"""逐月下載 TAIFEX 台股期貨(TX) 原始每日行情 CSV（Big5編碼），存到
data/raw/taifex_tx/。這個端點（futDataDown）一次最多只能查約1個月，範圍
太大會回傳錯誤頁，所以逐月請求，中間加延遲避免對官方伺服器造成負擔。

回傳資料包含**同一天所有到期月份的合約**（近月+遠月+價差單），才能真正
算出近月/遠月價差（這是這次會話先前判定「做不到」的分析，找到
www.taifex.com.tw 這個公開端點後才變得可行——使用者在web版把這個網域
加入環境的網路白名單後才連得到，之前這個沙盒的網路政策擋住）。

用法：
    python scripts/fetch_taifex_tx_raw.py 2001-01 2023-12
"""

from __future__ import annotations

import sys
import time
from datetime import date
from pathlib import Path

import urllib.request
import urllib.parse

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "data" / "raw" / "taifex_tx"
ENDPOINT = "https://www.taifex.com.tw/cht/3/futDataDown"
SLEEP_SECONDS = 1.5
MAX_RETRIES = 3


def month_range(start_ym: str, end_ym: str) -> list[str]:
    sy, sm = map(int, start_ym.split("-"))
    ey, em = map(int, end_ym.split("-"))
    months = []
    y, m = sy, sm
    while (y, m) <= (ey, em):
        months.append(f"{y:04d}-{m:02d}")
        m += 1
        if m > 12:
            m = 1
            y += 1
    return months


def month_bounds(ym: str) -> tuple[str, str]:
    y, m = map(int, ym.split("-"))
    start = date(y, m, 1)
    end = date(y + 1, 1, 1) if m == 12 else date(y, m + 1, 1)
    end = date.fromordinal(end.toordinal() - 1)
    return start.strftime("%Y/%m/%d"), end.strftime("%Y/%m/%d")


def fetch_month(ym: str) -> bytes | None:
    start_s, end_s = month_bounds(ym)
    payload = urllib.parse.urlencode(dict(
        down_type="1", commodity_id="TX", commodity_id2="",
        queryStartDate=start_s, queryEndDate=end_s,
    )).encode()
    req = urllib.request.Request(ENDPOINT, data=payload, headers={
        "User-Agent": "Mozilla/5.0 (research script; TXF quant analysis)",
        "Content-Type": "application/x-www-form-urlencoded",
    })
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return resp.read()
        except Exception as e:
            print(f"  [{ym}] attempt {attempt} failed: {e}")
            time.sleep(3)
    return None


def main() -> None:
    if len(sys.argv) != 3:
        print("用法: python scripts/fetch_taifex_tx_raw.py START_YYYY-MM END_YYYY-MM")
        sys.exit(1)
    start_ym, end_ym = sys.argv[1], sys.argv[2]
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    months = month_range(start_ym, end_ym)
    print(f"共 {len(months)} 個月，從 {start_ym} 到 {end_ym}")

    ok, failed, skipped = 0, [], 0
    for i, ym in enumerate(months, 1):
        out_path = OUT_DIR / f"TX_{ym}.csv"
        if out_path.exists() and out_path.stat().st_size > 100:
            skipped += 1
            continue
        raw = fetch_month(ym)
        if raw is None:
            failed.append(ym)
            print(f"[{i}/{len(months)}] {ym}: 失敗")
            continue
        text = raw.decode("big5", errors="replace")
        if "交易日期" not in text:
            failed.append(ym)
            print(f"[{i}/{len(months)}] {ym}: 回傳內容不含表頭，可能是錯誤頁，跳過")
            continue
        out_path.write_text(text, encoding="utf-8")
        ok += 1
        if i % 20 == 0 or i == len(months):
            print(f"[{i}/{len(months)}] 進度：成功={ok} 失敗={len(failed)} 已存在跳過={skipped}")
        time.sleep(SLEEP_SECONDS)

    print(f"\n完成。成功={ok} 失敗={len(failed)} 已存在跳過={skipped}")
    if failed:
        print("失敗月份:", failed)


if __name__ == "__main__":
    main()
