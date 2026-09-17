"""一次性診斷：印出 Neon 資料庫裡每一檔股票實際的資料範圍與筆數。
用完應該刪除，不是常態要跑的東西。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tw_quant.storage import get_data_store


def main() -> None:
    store = get_data_store()
    prices = store.load_prices()
    margin = store.load_margin_short()

    print(f"prices 總筆數: {len(prices)}")
    print(f"margin_short 總筆數: {len(margin)}")
    print()
    print("每檔股票的資料範圍：")
    if not prices.empty:
        summary = prices.groupby("stock_id")["date"].agg(["min", "max", "count"])
        print(summary.to_string())
    else:
        print("（空）")

    store.close()


if __name__ == "__main__":
    main()
