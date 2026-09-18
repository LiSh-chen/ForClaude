"""診斷用：實際呼叫 FinMind API，確認有沒有分鐘級/tick 級的盤中歷史資料可用。

這支腳本不是要建立正式資料管線，純粹是「打電話問清楚」——嘗試幾個可能的
盤中資料集名稱，把 FinMind 實際回應的內容印出來（狀態、筆數、日期範圍、
欄位），這樣才能確定：
  1. 這個資料集到底存不存在
  2. 免費 token 能不能拿到資料，還是要付費/贊助方案
  3. 如果拿得到，歷史回溯的深度夠不夠（backtest 通常需要至少幾年）

這個沙盒環境連不到 api.finmindtrade.com，所以只能在 GitHub Actions
（有正常網路）上實際跑一次才知道答案。

用法：
    python scripts/check_finmind_intraday_availability.py
"""

from __future__ import annotations

import os
import sys

import requests

BASE_URL = "https://api.finmindtrade.com/api/v4/data"

# 依 FinMind 官方常見命名慣例猜測的候選資料集名稱，逐一實測有沒有資料。
CANDIDATE_DATASETS = [
    "TaiwanStockPriceTick",
    "TaiwanStockPriceMinute",
    "TaiwanStockTick",
    "TaiwanStock5MinuteKline",
    "TaiwanStockKBar",
    "TaiwanStockPricePerMinute",
    "TaiwanStockStatisticsOfOrderBookAndTrade",
]

TEST_STOCK_ID = "2330"  # 台積電，流動性最高，最有機會有完整資料
# 用一個「幾年前」的日期測試歷史回溯深度，而不是只測最近一天
TEST_START_DATE = "2023-01-01"
TEST_END_DATE = "2023-01-10"


def check_dataset(dataset: str, token: str | None) -> None:
    params = {"dataset": dataset, "data_id": TEST_STOCK_ID, "start_date": TEST_START_DATE, "end_date": TEST_END_DATE}
    if token:
        params["token"] = token

    print(f"\n=== dataset={dataset} ===")
    try:
        resp = requests.get(BASE_URL, params=params, timeout=20)
    except requests.RequestException as e:
        print(f"  請求失敗: {e}")
        return

    print(f"  HTTP 狀態碼: {resp.status_code}")
    try:
        payload = resp.json()
    except ValueError:
        print(f"  回應不是 JSON，前 300 字元: {resp.text[:300]}")
        return

    msg = payload.get("msg", "")
    status = payload.get("status", "")
    data = payload.get("data", [])
    print(f"  status={status}  msg={msg}")
    print(f"  資料筆數: {len(data)}")
    if data:
        print(f"  欄位: {list(data[0].keys())}")
        print(f"  第一筆: {data[0]}")
        print(f"  最後一筆: {data[-1]}")


def check_recent_availability(dataset: str, token: str | None) -> None:
    """有些盤中資料集免費版可能只給「最近幾天」，這裡額外測一次最近日期的範圍。"""
    import datetime

    today = datetime.date.today()
    start = (today - datetime.timedelta(days=5)).isoformat()
    end = today.isoformat()
    params = {"dataset": dataset, "data_id": TEST_STOCK_ID, "start_date": start, "end_date": end}
    if token:
        params["token"] = token

    print(f"\n=== dataset={dataset}（測最近 5 天：{start} ~ {end}）===")
    try:
        resp = requests.get(BASE_URL, params=params, timeout=20)
        payload = resp.json()
    except (requests.RequestException, ValueError) as e:
        print(f"  請求失敗: {e}")
        return
    data = payload.get("data", [])
    print(f"  status={payload.get('status')}  msg={payload.get('msg')}  資料筆數={len(data)}")
    if data:
        print(f"  第一筆: {data[0]}")
        print(f"  最後一筆: {data[-1]}")


def main() -> None:
    token = os.environ.get("FINMIND_TOKEN")
    print(f"是否有 FINMIND_TOKEN: {bool(token)}")

    for dataset in CANDIDATE_DATASETS:
        check_dataset(dataset, token)

    # 對看起來有資料的資料集，額外測一次「最近幾天」的可用性
    print("\n\n########## 額外測試：最近日期的可用性 ##########")
    for dataset in CANDIDATE_DATASETS:
        check_recent_availability(dataset, token)


if __name__ == "__main__":
    main()
