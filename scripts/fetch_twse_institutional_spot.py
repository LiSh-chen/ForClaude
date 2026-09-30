"""逐日下載 TWSE 三大法人買賣金額統計表(BFI82U端點)，只抓跟
data/taifex_txf_institutional_daily.parquet完全同一組交易日
（2023-10-02~2026-09-24，725天）——因為期貨籌碼資料本身就只有這個窗口
（TAIFEX端點的服務限制，見383424a commit），現貨買賣超抓更早的資料也
沒用武之地，不如把時間省下來只抓真正能跟期貨OI對接的這725天。

這個端點只支援單日查詢(dayDate=YYYYMMDD)，沒有月範圍版本（試過type=month
會直接被WAF擋下，跟先前FMTQIK/BWIBBU端點踩過的陷阱一樣），所以是逐日
請求，725次算是可以接受的量。

輸出欄位對齊期貨籌碼檔案的identity分類（外資及陸資/投信/自營商），方便
直接跟 taifex_txf_institutional_daily.parquet 用date+identity做join：
- 外資及陸資 = 「外資及陸資(不含外資自營商)」+「外資自營商」買賣金額加總
- 投信 = 原始「投信」
- 自營商 = 「自營商(自行買賣)」+「自營商(避險)」買賣金額加總
（TWSE把自營商拆成自行買賣/避險兩類，期貨籌碼只有一個「自營商」，這裡
加總對齊，避免join後多出兩個對不上的類別）

用法：
    python scripts/fetch_twse_institutional_spot.py
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
TXF_INST_PATH = REPO_ROOT / "data" / "taifex_txf_institutional_daily.parquet"
OUT_PATH = REPO_ROOT / "data" / "twse_institutional_spot_daily.parquet"
PROGRESS_DIR = REPO_ROOT / "data" / "raw" / "twse_institutional_spot_progress"
ENDPOINT = "https://www.twse.com.tw/rwd/zh/fund/BFI82U"
SLEEP_SECONDS = 3.0
MAX_RETRIES = 2

# TWSE原始「單位名稱」-> 對齊期貨籌碼分類的identity
NAME_TO_IDENTITY = {
    "自營商(自行買賣)": "自營商",
    "自營商(避險)": "自營商",
    "投信": "投信",
    "外資及陸資(不含外資自營商)": "外資及陸資",
    "外資自營商": "外資及陸資",
}


def to_int(s: str) -> int:
    return int(s.replace(",", ""))


def fetch_day(day_str: str) -> dict | None:
    url = f"{ENDPOINT}?dayDate={day_str}&type=day&response=json"
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            result = subprocess.run(["curl", "-sSL", "-m", "30", url], capture_output=True, timeout=35)
            if result.returncode == 0 and result.stdout:
                text = result.stdout.decode("utf-8", errors="replace")
                if "安全性考量" in text or "網站維護中" in text:
                    print(f"  [{day_str}] 疑似被WAF擋下或維護中，跳過")
                    return None
                return json.loads(text)
        except Exception as e:
            print(f"  [{day_str}] attempt {attempt} failed: {e}")
        time.sleep(3)
    return None


def parse_day(day_str: str, data: dict) -> list[dict]:
    if data.get("stat") != "OK":
        return []
    agg: dict[str, list[int, int]] = {}
    for row in data["data"]:
        name, buy_s, sell_s, _net_s = row
        identity = NAME_TO_IDENTITY.get(name)
        if identity is None:
            continue
        buy, sell = to_int(buy_s), to_int(sell_s)
        if identity not in agg:
            agg[identity] = [0, 0]
        agg[identity][0] += buy
        agg[identity][1] += sell
    date = pd.Timestamp(f"{day_str[:4]}-{day_str[4:6]}-{day_str[6:]}")
    return [dict(date=date, identity=k, buy_amount=v[0], sell_amount=v[1], net_amount=v[0] - v[1])
            for k, v in agg.items()]


def main() -> None:
    PROGRESS_DIR.mkdir(parents=True, exist_ok=True)
    txf_inst = pd.read_parquet(TXF_INST_PATH)
    days = sorted(txf_inst["date"].dt.strftime("%Y%m%d").unique())
    print(f"共 {len(days)} 個交易日，從 {days[0]} 到 {days[-1]}")

    existing_days = {fp.stem.replace("spot_", "") for fp in PROGRESS_DIR.glob("spot_*.parquet")}
    all_frames = [pd.read_parquet(fp) for fp in PROGRESS_DIR.glob("spot_*.parquet")]

    ok, failed, consecutive_failures = 0, [], 0
    for i, day_str in enumerate(days, 1):
        if day_str in existing_days:
            continue
        data = fetch_day(day_str)
        if data is None:
            failed.append(day_str)
            consecutive_failures += 1
            print(f"[{i}/{len(days)}] {day_str}: 失敗")
            if consecutive_failures >= 3:
                print("連續3次失敗，研判被伺服器端擋下，提早停止避免延長封鎖，剩餘留到下次冷卻後再跑。")
                break
            time.sleep(SLEEP_SECONDS)
            continue
        consecutive_failures = 0
        rows = parse_day(day_str, data)
        if rows:
            day_df = pd.DataFrame(rows)
            day_df.to_parquet(PROGRESS_DIR / f"spot_{day_str}.parquet", index=False)
            all_frames.append(day_df)
            ok += 1
        if i % 50 == 0 or i == len(days):
            print(f"[{i}/{len(days)}] 進度：成功={ok} 失敗={len(failed)}")
        time.sleep(SLEEP_SECONDS)

    if all_frames:
        combined = pd.concat(all_frames, ignore_index=True).sort_values(["date", "identity"]).reset_index(drop=True)
        combined.to_parquet(OUT_PATH, index=False)
        print(f"\n合併後總筆數: {len(combined)} ({combined['date'].nunique()} 個交易日)")
        print(f"saved: {OUT_PATH}")

    print(f"\n完成。成功={ok} 失敗={len(failed)}")
    if failed:
        print("失敗日期:", failed)


if __name__ == "__main__":
    main()
