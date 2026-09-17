"""端到端示範：在合成資料上跑完整回測（含 MDD 熔斷兩階段流程）。

★ 本腳本使用合成假資料，僅用來證明整條 pipeline（regime -> signals ->
   global lock -> sizing -> MDD -> costs）可以正確串接執行，不代表任何真實
   績效。真正上線前必須換成第肆章（README）所述的真實台股資料。

★ 為了在小規模合成資料上就能觀察到訊號與成交，這裡把大盤環境與濾網門檻
   放寬，並把「單一產業曝險上限」從規格預設的 10% 調到 30%——這是刻意的
   demo 妥協，見 README「已知限制」一節說明為什麼預設的 10%/20% 組合在
   絕大多數情況下會讓全域鎖直接擋掉所有交易。正式上線請改回 StrategyConfig()
   預設值，並自行評估這個交互作用是否符合你的風控意圖。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tw_quant.backtest import run_backtest, summarize_performance
from tw_quant.config import StrategyConfig
from tw_quant.data_provider import SyntheticUniverseConfig, generate_synthetic_universe


def build_demo_config() -> StrategyConfig:
    cfg = StrategyConfig()
    cfg.regime.breadth_threshold = 0.25
    cfg.regime.volume_ratio_threshold = 0.9
    cfg.squeeze.pr_threshold = 40
    cfg.ignition.volume_multiplier = 1.2
    cfg.global_risk.max_industry_exposure_pct = 0.30
    return cfg


def main() -> None:
    data = generate_synthetic_universe(SyntheticUniverseConfig(n_stocks=40, n_days=700, seed=7))
    cfg = build_demo_config()

    print("=== 第一階段：不啟用 MDD 熔斷，取得基準 MDD ===")
    baseline = run_backtest(data["prices"], data["margin_short"], cfg, historical_mdd=None)
    baseline_metrics = summarize_performance(baseline, cfg.initial_capital)
    for k, v in baseline_metrics.items():
        print(f"  {k}: {v}")

    historical_mdd = baseline_metrics["max_dd"]
    print(f"\n基準 MDD = {historical_mdd:.2%}，熔斷門檻 = {historical_mdd * cfg.mdd.circuit_multiplier:.2%}")

    print("\n=== 第二階段：啟用 MDD 熔斷與降級恢復矩陣 ===")
    result = run_backtest(data["prices"], data["margin_short"], cfg, historical_mdd=historical_mdd)
    metrics = summarize_performance(result, cfg.initial_capital)
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    print(f"  熔斷觸發次數: {result.mdd_breach_count}")
    print(f"  尚未平倉部位數: {len(result.open_positions)}")
    print(f"  被全域鎖/防呆機制捨棄的候選數: {len(result.rejected_log)}")

    if not result.trades.empty:
        print("\n最近 5 筆交易：")
        print(result.trades.tail(5).to_string(index=False))


if __name__ == "__main__":
    main()
