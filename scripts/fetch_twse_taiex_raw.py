"""逐月下載 TWSE 台股加權指數(TAIEX)每日收盤 JSON（FMTQIK端點），存到
data/raw/twse_taiex/。這個端點回傳整個月的資料，一次一個月份即可涵蓋，
逐月請求並加延遲避免對官方伺服器造成負擔。

用法：
    python scripts/fetch_twse_taiex_raw.py 2001-01 2023-12
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "data" / "raw" / "twse_taiex"
ENDPOINT = "https://www.twse.com.tw/rwd/en/afterTrading/FMTQIK"
SLEEP_SECONDS = 3.0
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


def fetch_month(ym: str) -> dict | None:
    # 改用 curl 子行程：urllib.request 對這個端點會被導向307（推測是伺服器端
    # 用TLS/HTTP client特徵做的簡易防爬蟲機制），curl則能正常取得資料，
    # 已實測驗證過（見這次會話的除錯過程）。
    y, m = ym.split("-")
    url = f"{ENDPOINT}?date={y}{m}01&response=json"
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            result = subprocess.run(
                ["curl", "-sSL", "-m", "30", url],
                capture_output=True, timeout=35,
            )
            if result.returncode == 0 and result.stdout:
                return json.loads(result.stdout.decode("utf-8"))
            print(f"  [{ym}] attempt {attempt} failed: returncode={result.returncode} "
                  f"stderr={result.stderr[:200]!r}")
        except Exception as e:
            print(f"  [{ym}] attempt {attempt} failed: {e}")
        time.sleep(3)
    return None


def main() -> None:
    if len(sys.argv) != 3:
        print("用法: python scripts/fetch_twse_taiex_raw.py START_YYYY-MM END_YYYY-MM")
        sys.exit(1)
    start_ym, end_ym = sys.argv[1], sys.argv[2]
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    months = month_range(start_ym, end_ym)
    print(f"共 {len(months)} 個月，從 {start_ym} 到 {end_ym}")

    ok, failed, skipped, no_data = 0, [], 0, []
    for i, ym in enumerate(months, 1):
        out_path = OUT_DIR / f"TAIEX_{ym}.json"
        if out_path.exists() and out_path.stat().st_size > 50:
            skipped += 1
            continue
        data = fetch_month(ym)
        if data is None:
            failed.append(ym)
            print(f"[{i}/{len(months)}] {ym}: 失敗")
            continue
        if data.get("stat") != "OK":
            no_data.append(ym)
            print(f"[{i}/{len(months)}] {ym}: stat={data.get('stat')}（可能該月無交易日資料，如未來月份）")
            continue
        out_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        ok += 1
        if i % 20 == 0 or i == len(months):
            print(f"[{i}/{len(months)}] 進度：成功={ok} 失敗={len(failed)} 已存在跳過={skipped}")
        time.sleep(SLEEP_SECONDS)

    print(f"\n完成。成功={ok} 失敗={len(failed)} 無資料={len(no_data)} 已存在跳過={skipped}")
    if failed:
        print("失敗月份:", failed)
    if no_data:
        print("無資料月份:", no_data)


if __name__ == "__main__":
    main()
