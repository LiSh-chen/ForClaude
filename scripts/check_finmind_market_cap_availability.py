"""一次性診斷：查 FinMind 免費層有沒有股本/市值/在外流通股數相關的 dataset，
用來判斷能不能做「排除股本 > 50 億」的中小型股濾網。試完這輪就會刪除
（跟之前的 check_finmind_intraday_availability.py 同一種用法）。

用法：
    python scripts/check_finmind_market_cap_availability.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tw_quant.data_provider import FinMindDataProvider

CANDIDATE_DATASETS = [
    "TaiwanStockPER",
    "TaiwanStockMarketValue",
    "TaiwanStockCapital",
    "TaiwanStockShareholding",
    "TaiwanStockTotalOutstandingShares",
    "TaiwanStockCapitalReductionReferencePrice",
    "TaiwanStockMarketValueWeight",
    "TaiwanStockStatisticsOfOrderBookAndTrade",
]


def main() -> None:
    token = os.environ.get("FINMIND_TOKEN")
    provider = FinMindDataProvider(token=token)
    test_stock = "2330"  # 台積電，資料最不可能缺

    for dataset in CANDIDATE_DATASETS:
        try:
            df = provider._get(dataset, test_stock, "2024-01-01", "2024-01-31")
            if df.empty:
                print(f"[{dataset}] 空結果（dataset 名稱可能不存在，或這檔股票這段期間沒資料）")
            else:
                print(f"[{dataset}] 成功！欄位：{list(df.columns)}")
                print(df.head(3).to_string())
        except Exception as exc:  # noqa: BLE001
            print(f"[{dataset}] 錯誤：{exc}")
        print()


if __name__ == "__main__":
    main()
