"""壹、大盤環境判定參數敏感度檢驗示範。

對 breadth_threshold x volume_ratio_threshold 做網格回測，並偵測相鄰參數點
是否有「績效懸崖」（換一個門檻，績效就雪崩式惡化，代表策略踩在懸崖邊而非
穩健的參數平原上）。

同樣使用合成假資料與 demo 專用的寬鬆濾網設定，見 run_demo_backtest.py 的
說明；真正上線前務必用真實歷史資料重跑這個網格。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tw_quant.backtest import run_backtest, summarize_performance
from tw_quant.data_provider import SyntheticUniverseConfig, generate_synthetic_universe
from tw_quant.regime import detect_performance_cliff, run_regime_sensitivity_grid
from scripts.run_demo_backtest import build_demo_config


def main() -> None:
    data = generate_synthetic_universe(SyntheticUniverseConfig(n_stocks=30, n_days=700, seed=7))
    base_cfg = build_demo_config()

    def backtest_fn(breadth_threshold: float, volume_threshold: float) -> dict:
        cfg = build_demo_config()
        cfg.regime.breadth_threshold = breadth_threshold
        cfg.regime.volume_ratio_threshold = volume_threshold
        result = run_backtest(data["prices"], data["margin_short"], cfg)
        return summarize_performance(result, cfg.initial_capital)

    results = run_regime_sensitivity_grid(
        base_cfg.regime,
        backtest_fn,
        breadth_grid=(0.20, 0.25, 0.30),
        volume_grid=(0.8, 0.9, 1.0),
    )

    print("breadth_threshold  volume_threshold  total_return  max_dd  n_trades")
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
        print("  無（相鄰參數點績效落差都在容忍度內）")


if __name__ == "__main__":
    main()
