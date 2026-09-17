"""伍、WFA 滾動驗證示範：3 年訓練 + 1 年盲測，含過度擬合警報與大數檢驗。

用合成資料跑通整條 WFA 流程（切窗 -> 訓練窗網格選參數 -> 盲測窗驗證 ->
參數穩定度檢查 -> 若盲測年均交易次數不足則依序放寬濾網）。WFA 需要較長的
歷史（每個視窗前面還要再疊加暖身期），這裡刻意產生較長的合成資料
（約 10 年）並縮小股票數與參數網格以控制執行時間；真正上線請用實際
台股歷史資料，且務必評估執行時間隨股票數與天數增長的成本（見 README）。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tw_quant.data_provider import SyntheticUniverseConfig, generate_synthetic_universe
from tw_quant.wfa import run_wfa_with_large_number_check
from scripts.run_demo_backtest import build_demo_config


def main() -> None:
    data = generate_synthetic_universe(SyntheticUniverseConfig(n_stocks=10, n_days=2500, seed=99))
    cfg = build_demo_config()
    cfg.wfa.breakout_window_grid = (40, 60, 80)

    outcome = run_wfa_with_large_number_check(data["prices"], data["margin_short"], cfg)

    print(f"套用的大數檢驗放寬方案: {outcome.applied_relaxations or '無'}")
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
