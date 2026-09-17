"""大盤環境判定參數敏感度檢驗——讀取累積下來的真實資料（而非合成假資料）。

跟 run_sensitivity_test.py 邏輯完全一樣，差別只在資料來源：這裡用
get_data_store() 讀 ingest_daily_data.py 每日排程累積下來的真實台股資料。

用法：
    python scripts/run_sensitivity_test_from_db.py
    python scripts/run_sensitivity_test_from_db.py --start-date 2023-01-01
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tw_quant.backtest import run_backtest, summarize_performance
from tw_quant.regime import detect_performance_cliff, run_regime_sensitivity_grid
from tw_quant.storage import get_data_store
from scripts.run_demo_backtest import build_demo_config


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
    print(
        f"讀到 {n_stocks} 檔股票的資料"
        f"（{prices['date'].min().date()} ~ {prices['date'].max().date()}）"
    )

    # 用跟 run_demo_backtest.py 一樣的「demo 專用寬鬆設定」（放寬大盤環境門檻、
    # 產業曝險上限調到 30%），原因見 README 第 6.1 節：規格書預設的
    # 2%風險/20%單檔上限/10%產業上限三個數字疊在一起，在標的數少的情況下
    # 幾乎會讓全域鎖擋掉所有訊號。正式上線前請改回 StrategyConfig() 預設值，
    # 自行評估這個交互作用是否符合你的風控意圖。
    base_cfg = build_demo_config()

    def backtest_fn(breadth_threshold: float, volume_threshold: float) -> dict:
        cfg = build_demo_config()
        cfg.regime.breadth_threshold = breadth_threshold
        cfg.regime.volume_ratio_threshold = volume_threshold
        result = run_backtest(prices, margin_short, cfg)
        return summarize_performance(result, cfg.initial_capital)

    results = run_regime_sensitivity_grid(
        base_cfg.regime,
        backtest_fn,
        breadth_grid=(0.20, 0.25, 0.30),
        volume_grid=(0.8, 0.9, 1.0),
    )

    print("\nbreadth_threshold  volume_threshold  total_return  max_dd  n_trades")
    for r in results:
        m = r.metrics
        print(
            f"{r.breadth_threshold:>17.2f}  {r.volume_threshold:>16.2f}  "
            f"{m['total_return']:>12.2%}  {m['max_dd']:>6.2%}  {m['n_trades']:>8d}"
        )

    warnings = detect_performance_cliff(results, metric="total_return", tolerance_pct=0.30)
    print("\n懸崖警報：")
    if warnings:
        for w in warnings:
            print(" ", w)
    else:
        print("  無（相鄰參數點績效落差都在容忍度內——但標的數/資料量若還很少，")
        print("  這個結論的統計意義有限，見 README 第 6.4 節）")


if __name__ == "__main__":
    main()
