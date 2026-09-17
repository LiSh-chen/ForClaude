"""WFA 滾動驗證——讀取累積下來的真實資料（而非合成假資料）。

跟 run_wfa_demo.py 邏輯完全一樣，差別只在資料來源：這裡用 get_data_store()
讀 ingest_daily_data.py 每日排程累積下來的真實台股資料。

⚠️ WFA 需要「3 年訓練 + 1 年盲測」再加上指標暖身期（約 250+252+40 天），
單一視窗大概要 4 年多的歷史才跑得出來。剛開始用這個系統累積資料時，
資料量還不夠長，這支腳本會印出「視窗數 = 0」而不是報錯，這是正常現象，
不是 bug——資料庫需要繼續累積到有足夠長的歷史才有 WFA 視窗可以驗證。

用法：
    python scripts/run_wfa_from_db.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tw_quant.storage import get_data_store
from tw_quant.wfa import run_wfa_with_large_number_check
from scripts.run_demo_backtest import build_demo_config


def main() -> None:
    store = get_data_store()
    prices = store.load_prices()
    margin_short = store.load_margin_short()

    if prices.empty:
        print(
            "資料庫裡沒有任何價量資料。請先手動跑一次 "
            "`python scripts/ingest_daily_data.py`，或等 GitHub Actions 排程跑過至少一次。",
            file=sys.stderr,
        )
        sys.exit(1)

    n_stocks = prices["stock_id"].nunique()
    span_days = (prices["date"].max() - prices["date"].min()).days
    print(
        f"讀到 {n_stocks} 檔股票的資料"
        f"（{prices['date'].min().date()} ~ {prices['date'].max().date()}，"
        f"約 {span_days / 365:.1f} 年）"
    )

    cfg = build_demo_config()
    cfg.wfa.breakout_window_grid = (40, 60, 80)

    outcome = run_wfa_with_large_number_check(prices, margin_short, cfg)

    if not outcome.final_results:
        print(
            "\n⚠️ 目前歷史資料長度不足以產生任何 WFA 視窗（3 年訓練 + 1 年盲測 "
            "再加上指標暖身期，單一視窗大概需要 4 年多的歷史）。這不是錯誤，"
            "純粹是資料還沒累積夠長；等每日排程繼續跑一段時間、資料庫裡的歷史"
            "跨度變長之後再重跑這支腳本。"
        )
        return

    print(f"\n套用的大數檢驗放寬方案: {outcome.applied_relaxations or '無'}")
    print(f"最終盲測平均年交易次數: {outcome.avg_annual_trades:.1f}\n")

    print("視窗別結果：")
    for r in outcome.final_results:
        print(
            f"  訓練 {r.train_start.date()}~{r.train_end.date()} (最佳 N={r.best_param}) "
            f"-> 盲測 {r.test_start.date()}~{r.test_end.date()}: "
            f"報酬 {r.test_metrics['total_return']:.2%}, "
            f"MDD {r.test_metrics['max_dd']:.2%}, "
            f"交易數 {r.test_metrics['n_trades']}"
        )

    print("\n過度擬合警報：")
    if outcome.final_alerts:
        for a in outcome.final_alerts:
            print(" ", a)
    else:
        print("  無（相鄰訓練窗最佳參數變異在容忍度內）")


if __name__ == "__main__":
    main()
