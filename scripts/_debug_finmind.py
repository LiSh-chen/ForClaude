"""一次性診斷腳本：印出 FinMind TaiwanStockPrice 的真實回傳欄位與原始資料，
用來排查 fetch_price() 為什麼在 GitHub Actions 實測時全部拋出
"cannot assemble with duplicate keys"。跑完診斷、修好 data_provider.py 後
這個檔案應該刪除，不是常態要跑的東西。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import requests

BASE_URL = "https://api.finmindtrade.com/api/v4/data"


def main() -> None:
    print(f"pandas version: {pd.__version__}")

    resp = requests.get(
        BASE_URL,
        params={
            "dataset": "TaiwanStockPrice",
            "data_id": "2330",
            "start_date": "2024-01-01",
            "end_date": "2024-01-10",
        },
        timeout=30,
    )
    print(f"HTTP status: {resp.status_code}")
    payload = resp.json()
    print(f"top-level keys: {list(payload.keys())}")
    print(f"msg: {payload.get('msg')}")
    print(f"status field: {payload.get('status')}")

    data = payload.get("data", [])
    print(f"data length: {len(data)}")
    if data:
        print(f"first record keys: {list(data[0].keys())}")
        print(f"first record: {data[0]}")

    raw = pd.DataFrame(data)
    print(f"DataFrame columns: {raw.columns.tolist()}")
    print(f"DataFrame dtypes:\n{raw.dtypes}")
    print(f"DataFrame shape: {raw.shape}")
    if not raw.empty:
        print(raw.head())


if __name__ == "__main__":
    main()
