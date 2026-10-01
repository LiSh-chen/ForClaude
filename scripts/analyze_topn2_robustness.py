"""針對「動量策略完整探索」報告裡被認為是目前最佳候選的 top_n=2 組合，
做比點估計更深入的穩健性檢驗：用移動區塊拔靴法（見 tw_quant/robustness.py）
量化 Sharpe 估計值本身的不確定性，而不是只看單一個數字就下結論。

背景：168 組網格（data/topn_grid_metrics.parquet）顯示 top_n=2、
momentum_window=126 這個組合的樣本外 Sharpe 對調倉頻率極度敏感——
21 天調倉（也是「動量策略完整探索」分頁預設引用的設定）只有 Sharpe 0.57
（略輸 QQQ 的 0.69），42/63/126 天調倉卻分別有 0.78/0.75/0.83（看似贏過
QQQ）。但調倉越慢、樣本外窗格（約 5 年）裡的交易次數就越少——126 天調倉
只交易了 20 次——20 筆交易算出來的 Sharpe，點估計本身的抽樣誤差可能大到
讓「贏過 QQQ」這個結論根本站不住腳。這支腳本就是要量化這個不確定性有多大。

完全讀本機已經 commit 的 Parquet 資料（data/strategy_equity_curves.parquet
的 QQQ 與 top_n=2 預設設定、data/topn_grid_equity.parquet 的 top_n=2 其他
調倉頻率），不連資料庫、不用重新回測，可以在這個沙盒環境直接跑。

用法：
    python scripts/analyze_topn2_robustness.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from tw_quant.robustness import bootstrap_sharpe_ci, daily_returns_from_equity, paired_bootstrap_sharpe_diff

PERIOD = "樣本外"  # 只檢驗樣本外——樣本內的點估計本來就不該拿來評估策略能不能用
N_BOOT = 5000
SEED = 42

# (顯示名稱, rebalance_freq_days) —— 21 是報告目前引用的預設值，
# 42/63/126 是 168 網格裡同一個 top_n=2/mw=126 切片中看起來最好的幾組。
REBAL_CONFIGS = [("rebal=21（報告目前引用的預設值）", 21), ("rebal=42", 42), ("rebal=63", 63), ("rebal=126", 126)]
BLOCK_SIZE_BY_REBAL = {21: 21, 42: 21, 63: 21, 126: 21}  # 區塊長度統一用約一個月交易日，不隨調倉頻率放大信賴區間寬度的假象


def _load_equity(path: Path, period: str, strategy: str) -> pd.Series:
    df = pd.read_parquet(path)
    sub = df[(df["period"] == period) & (df["strategy"] == strategy)].sort_values("date")
    if sub.empty:
        raise ValueError(f"找不到 period={period!r} strategy={strategy!r} 的資料，檔案：{path}")
    return sub.set_index("date")["equity"]


def main() -> None:
    qqq_equity = _load_equity(Path("data/strategy_equity_curves.parquet"), PERIOD, "QQQ")
    qqq_returns = daily_returns_from_equity(qqq_equity)
    qqq_result = bootstrap_sharpe_ci(qqq_returns, block_size=21, n_boot=N_BOOT, seed=SEED)

    print(f"=== {PERIOD}：top_n=2（momentum_window=126）穩健性檢驗，vs QQQ 買進持有 ===\n")
    print(
        f"QQQ 對照組：Sharpe 點估計 {qqq_result['point_estimate']:.3f}，"
        f"95% 拔靴信賴區間 [{qqq_result['ci_low']:.3f}, {qqq_result['ci_high']:.3f}]\n"
    )

    rows = []
    for label, rebal in REBAL_CONFIGS:
        if rebal == 21:
            equity = _load_equity(Path("data/strategy_equity_curves.parquet"), PERIOD, "top_n=2")
        else:
            equity = _load_equity(
                Path("data/topn_grid_equity.parquet"), PERIOD, f"top_n=2|rebal={rebal}|mw=126"
            )
        returns = daily_returns_from_equity(equity)
        block_size = BLOCK_SIZE_BY_REBAL[rebal]

        sharpe_result = bootstrap_sharpe_ci(returns, block_size=block_size, n_boot=N_BOOT, seed=SEED)
        diff_result = paired_bootstrap_sharpe_diff(returns, qqq_returns, block_size=block_size, n_boot=N_BOOT, seed=SEED)

        n_days = len(returns)
        print(f"--- {label}（{n_days} 個交易日報酬） ---")
        print(
            f"  Sharpe 點估計: {sharpe_result['point_estimate']:.3f}  "
            f"95% CI: [{sharpe_result['ci_low']:.3f}, {sharpe_result['ci_high']:.3f}]"
        )
        print(
            f"  vs QQQ 的 Sharpe 差值點估計: {diff_result['point_diff']:+.3f}  "
            f"95% CI: [{diff_result['ci_low']:+.3f}, {diff_result['ci_high']:+.3f}]  "
            f"拔靴分布裡「沒贏過 QQQ」的比例: {diff_result['prob_strategy_not_better']:.1%}"
        )
        verdict = (
            "統計上站得住腳（95% CI 下界 > 0）"
            if diff_result["ci_low"] > 0
            else "95% CI 跨過 0，不能排除「贏過 QQQ只是運氣」"
        )
        print(f"  => {verdict}\n")

        rows.append(
            {
                "config": label,
                "sharpe_point": sharpe_result["point_estimate"],
                "sharpe_ci_low": sharpe_result["ci_low"],
                "sharpe_ci_high": sharpe_result["ci_high"],
                "diff_vs_qqq_point": diff_result["point_diff"],
                "diff_vs_qqq_ci_low": diff_result["ci_low"],
                "diff_vs_qqq_ci_high": diff_result["ci_high"],
                "prob_not_better_than_qqq": diff_result["prob_strategy_not_better"],
                "n_days": n_days,
            }
        )

    summary = pd.DataFrame(rows)
    print("=== 彙整表 ===")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
