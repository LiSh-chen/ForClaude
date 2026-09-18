"""把台股目前最佳單一策略（結構轉折 C，Market Structure Shift + 量能確認，
見 docs/research_findings.md 第 3 節）原封不動套用到美股 S&P 500 資料上，
驗證這套訊號邏輯是否只是台股特有的市場微結構產物，還是能轉移到美股。

沿用跟台股完全一樣的訊號參數（downtrend_window=20, swing_window=15,
volume_confirm=1.3x, ATR停損=2.5x, chandelier_lookback=10, 大盤環境門檻
breadth=0.25/volume_ratio=0.9, 產業曝險上限30%），直接重用
scripts/test_holiday_overlay_from_db.py 的 mss_signal_fn（跟研究文件記載的
最終版一致）——唯一的改動是成本模型跟整股限制換成美股版
（tw_quant/us_config.py + tw_quant/us_costs.py：無證交稅、零手續費、
$0.01 tick、無 1000 股整張限制），資料來源換成 us_prices。

美股沒有台股的融資券資料，run_backtest 需要的 margin_short 參數傳一個
空的、欄位對齊的 DataFrame——mss_signal_fn 本來就完全不會用到這個參數，
只是引擎簽名要求要傳。

用法：
    python scripts/test_us_mss_strategy_from_db.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tw_quant import us_costs
from tw_quant.backtest import run_backtest
from tw_quant.backtest_stats import metrics_from_result
from tw_quant.storage import get_data_store
from tw_quant.us_config import build_us_config

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_holiday_overlay_from_db import mss_signal_fn  # noqa: E402


def build_us_mss_config():
    cfg = build_us_config()
    cfg.regime.breadth_threshold = 0.25
    cfg.regime.volume_ratio_threshold = 0.9
    cfg.global_risk.max_industry_exposure_pct = 0.30
    cfg.sizing.atr_multiplier = 2.5
    cfg.sizing.chandelier_lookback = 10
    return cfg


def main() -> None:
    store = get_data_store()
    us_prices = store.load_us_prices()

    if us_prices.empty:
        print("資料庫裡沒有任何美股價量資料。", file=sys.stderr)
        sys.exit(1)

    n_stocks = us_prices["stock_id"].nunique()
    print(
        f"讀到 {n_stocks} 檔美股的資料"
        f"（{us_prices['date'].min().date()} ~ {us_prices['date'].max().date()}）\n"
    )

    empty_margin_short = pd.DataFrame(columns=["date", "stock_id", "margin_purchase_balance", "short_balance"])

    cfg = build_us_mss_config()
    result = run_backtest(
        us_prices, empty_margin_short, cfg, entry_signal_fn=mss_signal_fn, cost_module=us_costs
    )
    m = metrics_from_result(result, cfg.initial_capital, prices=us_prices)

    print("=== 結構轉折 C 策略（台股原版參數）套用在美股 S&P 500 ===")
    print(f"總報酬: {m['total_return']:.2%}")
    print(f"CAGR: {m['cagr']:.2%}")
    print(f"MDD: {m['max_dd']:.2%}")
    print(f"Sharpe: {m['sharpe']:.2f}")
    print(f"Calmar: {m['calmar']:.2f}")
    print(f"交易筆數: {m['n_trades']:.0f}")
    print(f"勝率: {m['win_rate']:.1%}")
    rr = m["risk_reward_ratio"]
    print(f"風報比 RR: {'inf' if rr == float('inf') else f'{rr:.2f}'}")
    print(f"EV%（勝率加權後單筆期望報酬率）: {m['ev_pct']:.2%}")
    pf = m["profit_factor"]
    print(f"獲利因子 PF: {'inf' if pf == float('inf') else f'{pf:.2f}'}")

    if not result.trades.empty:
        print(f"\n交易標的清單（共 {result.trades['stock_id'].nunique()} 檔）：")
        print(sorted(result.trades["stock_id"].unique().tolist()))

    print(
        "\n（台股原版對照：總報酬 11.37%、CAGR 3.79%、MDD 11.69%、Sharpe 0.43、"
        "Calmar 0.32、42 筆交易、勝率 31.0%——同樣訊號邏輯、同樣風控參數，"
        "唯一差異是成本模型換成美股版跟資料來源換成 503 檔 S&P 500 成分股）"
    )


if __name__ == "__main__":
    main()
