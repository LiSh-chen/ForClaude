"""產生股票清單檔案：從 FinMind 抓全市場股票基本資料，篩選出普通股，
取前 N 檔寫成 data/stock_universe.txt（每行一個股票代號），供
ingest_daily_data.py 的 STOCK_UNIVERSE_FILE 機制讀取。

篩選規則：
  - type 為 twse（上市）或 tpex（上櫃）
  - stock_id 是 4 碼純數字（排除 5 碼以上的權證、可轉債等衍生商品代號）
  - 排除以 "00" 開頭的代號（ETF 慣例，如 0050、0056）

這不是「依市值/流動性排序取前 N 大」，純粹是全市場清單過濾掉 ETF/衍生
商品後、依代號排序取前 N 檔——如果你想要更講究的選股邏輯（例如真的照
成交金額排序），這支腳本只是起點，可以再接資料庫裡已經有的
turnover_value 欄位做二次篩選。

跟其他 ingest 腳本一樣，這支只能在有正常網路存取權的環境執行（GitHub
Actions runner），這個開發沙盒連不到 FinMind。

用法：
    python scripts/generate_stock_universe.py --limit 150
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tw_quant.data_provider import FinMindDataProvider

_ORDINARY_STOCK_ID = re.compile(r"^(?!00)\d{4}$")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=150)
    parser.add_argument("--output", default="data/stock_universe.txt")
    args = parser.parse_args()

    import os

    provider = FinMindDataProvider(token=os.environ.get("FINMIND_TOKEN"))
    info = provider.fetch_stock_info()

    if info.empty:
        print("FinMind 沒有回傳任何股票基本資料，中止。", file=sys.stderr)
        sys.exit(1)

    filtered = info[
        info["type"].isin(["twse", "tpex"]) & info["stock_id"].astype(str).str.match(_ORDINARY_STOCK_ID)
    ].drop_duplicates(subset=["stock_id"])
    filtered = filtered.sort_values("stock_id")

    selected = filtered["stock_id"].head(args.limit).tolist()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(selected) + "\n")

    print(f"全市場普通股（上市+上櫃）共 {len(filtered)} 檔，取前 {len(selected)} 檔寫入 {output_path}")


if __name__ == "__main__":
    main()
