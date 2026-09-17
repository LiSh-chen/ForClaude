"""從累積下來的真實資料（由 ingest_daily_data.py 每日排程寫入）跑正式回測。

資料來源與 ingest 腳本共用同一個 get_data_store()：沒設 DATABASE_URL 就讀
本機 data/tw_market.db，設了就讀雲端 Postgres，兩邊完全不用改程式碼。

用法：
    python scripts/run_backtest_from_db.py
    python scripts/run_backtest_from_db.py --start-date 2022-01-01 --end-date 2024-12-31
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tw_quant.backtest import run_backtest, summarize_performance
from tw_quant.config import StrategyConfig
from tw_quant.storage import get_data_store


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    args = parser.parse_args()

    store = get_data_store()
    prices = store.load_prices(start_date=args.start_date, end_date=args.end_date)
    margin_short = store.load_margin_short(start_date=args.start_date, end_date=args.end_date)

    if prices.empty:
        print(
            "資料庫裡沒有任何價量資料。請先手動跑一次 "
            "`python scripts/ingest_daily_data.py`，或等 GitHub Actions 排程跑過至少一次。",
            file=sys.stderr,
        )
        sys.exit(1)

    n_stocks = prices["stock_id"].nunique()
    n_days = prices["date"].nunique()
    print(f"讀到 {n_stocks} 檔股票、{n_days} 個交易日的資料"
          f"（{prices['date'].min().date()} ~ {prices['date'].max().date()}）")

    cfg = StrategyConfig()  # 規格書預設參數；請先閱讀 README 第 6.1 節的產業上限說明
    result = run_backtest(prices, margin_short, cfg, historical_mdd=None)
    metrics = summarize_performance(result, cfg.initial_capital)

    print("\n=== 回測結果（規格書預設參數） ===")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    print(f"  尚未平倉部位數: {len(result.open_positions)}")
    print(f"  被全域鎖/防呆機制捨棄的候選數: {len(result.rejected_log)}")


if __name__ == "__main__":
    main()
