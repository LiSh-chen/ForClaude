"""短期反轉策略：把 scripts/analyze_us_cross_sectional_ic.py 用 Newey-West
修正後，唯一撐過自相關修正的訊號——5/10/20 天落後報酬率在 5~20 天
forward horizon 上的「負」IC（也就是短期跌深反彈、不是動量延續）——
真的拿去回測，看看能不能轉成實際報酬，而不是停在「IC 顯著」這一步就
宣稱找到策略。

背景（見 2026-10-02 的 IC 掃描 + Newey-West 修正結果）：原始 100 組
(因子, horizon) 裡，未修正 t 值判定「顯著」的有 2 組（52週高點位置、
120天波動度，兩者都在 horizon=60 天），但用 Newey-West（lag=horizon-1）
修正自相關後，這兩組的 |t| 從 6~8 直接崩到 1 以下，完全不顯著——反而是
原本沒被排進「前五名」的短期因子撐住了：price_momentum_L5/L10/L20 配
horizon=5~20 天，還有 bollinger_zscore_L20 配 horizon=5~20 天，Newey-West
修正後 |t| 仍然落在 2~4.3 之間，而且全部都是負號（trailing 報酬率越高，
未來報酬反而越低——短期反轉，不是動量延續）。momentum_window 5/10/20
這三個獨立選出來的訊號方向一致、互相印證，比單一一組孤立的顯著結果
更可信。

做法：直接重用 tw_quant.factor_backtest.run_factor_backtest 內建的「落後
報酬率排名」訊號，不用另外寫 signal_fn——factor_cfg.ascending=True 時
排名反過來，買「過去 momentum_window 天報酬率最低」的 top_n 檔股票，
這正是 IC 掃描指出的方向（買最近跌最多的，不是買最近漲最多的）。
rebalance_freq_days 跟 momentum_window 設成同一個數字，對應 IC 掃描的
forward horizon（訊號測的是「trailing L 天報酬 -> 接下來 L 天報酬」，
調倉頻率也跟著用 L 天才是同一件事的回測版本）。

誠實的已知限制：run_factor_backtest 沿用 tw_quant.signals.build_pool_mask
當魚池篩選，要求 T-1 日收盤價站上 60 日均線——這是當初為動量/趨勢策略
設計的多頭排列濾網，跟「買最近跌最多的股票」這個反轉邏輯方向上有張力
（真正跌最深的股票很可能已經跌破 60 日均線，反而被這個濾網排除在外）。
這不是這支腳本特有的問題——專案裡既有的 RSI/布林通道均值回歸策略
（test_us_rsi_bollinger_reversion_from_db.py）用的是同一個 run_factor_backtest、
同一個預設魚池，從來沒有另外處理過這件事——這裡沿用同樣的慣例以維持
跟既有結果可比，但誠實揭露：這支腳本測到的，是「在多頭排列的股票裡挑
短期跌最多的」這種溫和反轉，不是「專買破底搶反彈」的激進版本。

跟 top_n=2 動量策略同樣的教訓：點估計不能直接相信，回測完會再用
tw_quant/robustness.py 的移動區塊拔靴法量化 Sharpe 對 QQQ 的優勢是不是
禁得起抽樣不確定性的考驗。

用法：
    python scripts/test_us_short_term_reversal_from_db.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from tw_quant import us_costs
from tw_quant.backtest_stats import metrics_from_result
from tw_quant.data_snapshot import load_us_index_membership_snapshot, load_us_prices_snapshot
from tw_quant.factor_backtest import FactorConfig, run_factor_backtest
from tw_quant.robustness import bootstrap_sharpe_ci, daily_returns_from_equity, paired_bootstrap_sharpe_diff
from tw_quant.us_config import build_us_config

MOMENTUM_WINDOW_GRID = (5, 10, 20)  # 對應 IC 掃描裡 Newey-West 修正後仍顯著的三個 L
TOP_N = 20

IN_SAMPLE_START = "2023-09-19"
OOS_START = "2018-09-20"
OOS_END = pd.Timestamp(IN_SAMPLE_START) - pd.Timedelta(days=1)

# 跟 scripts/test_us_dual_momentum_from_db.py 等既有腳本同一組硬編碼參考值
# （2026-09-21 membership 架構修正後、GitHub Actions 實跑的結果），這裡只是
# 拿來在彙整表裡順便列出對照，不是這支腳本自己重新抓的。
QQQ_IN_SAMPLE = {"total_return": 0.9817, "cagr": 0.2572, "max_dd": 0.2277, "sharpe": 1.22, "calmar": 1.13}
QQQ_OOS = {"total_return": 1.0782, "cagr": 0.1581, "max_dd": 0.3512, "sharpe": 0.69, "calmar": 0.45}

N_BOOT = 5000
SEED = 42


def _run(prices: pd.DataFrame, base_cfg, momentum_window: int, start_date, end_date, membership):
    factor_cfg = FactorConfig(
        momentum_window=momentum_window, rebalance_freq_days=momentum_window, top_n=TOP_N, ascending=True
    )
    result = run_factor_backtest(
        prices, base_cfg, factor_cfg, start_date=start_date, end_date=end_date, cost_module=us_costs, membership=membership
    )
    metrics = metrics_from_result(result, base_cfg.initial_capital, prices=prices)
    return result, metrics


def _load_qqq_equity(period_label: str) -> pd.Series:
    df = pd.read_parquet("data/strategy_equity_curves.parquet")
    sub = df[(df["period"] == period_label) & (df["strategy"] == "QQQ")].sort_values("date")
    return sub.set_index("date")["equity"]


def _print_period(label: str, period_label_for_qqq: str, prices, base_cfg, start_date, end_date, qqq_bench, membership) -> None:
    print(f"\n=== {label} ===")
    header = (
        f"{'mom_window':>10} {'total_ret':>10} {'cagr':>8} {'max_dd':>8} "
        f"{'sharpe':>7} {'calmar':>7} {'n_trd':>6} {'win%':>7}"
    )
    print(header)

    qqq_equity = _load_qqq_equity(period_label_for_qqq)
    qqq_returns = daily_returns_from_equity(qqq_equity)

    for W in MOMENTUM_WINDOW_GRID:
        result, m = _run(prices, base_cfg, W, start_date, end_date, membership)
        print(
            f"{W:>10} {m['total_return']:>9.2%} {m['cagr']:>7.2%} {m['max_dd']:>7.2%} "
            f"{m['sharpe']:>7.2f} {m['calmar']:>7.2f} {m['n_trades']:>6.0f} {m['win_rate']:>6.1%}"
        )

        strategy_returns = daily_returns_from_equity(result.equity_curve["equity"])
        block_size = max(W, 5)
        sharpe_ci = bootstrap_sharpe_ci(strategy_returns, block_size=block_size, n_boot=N_BOOT, seed=SEED)
        diff = paired_bootstrap_sharpe_diff(strategy_returns, qqq_returns, block_size=block_size, n_boot=N_BOOT, seed=SEED)
        verdict = "95% CI 下界 > 0，統計上站得住腳" if diff["ci_low"] > 0 else "95% CI 跨過 0，不能排除是運氣"
        print(
            f"    拔靴 Sharpe 95% CI: [{sharpe_ci['ci_low']:.3f}, {sharpe_ci['ci_high']:.3f}]  "
            f"vs QQQ 差值 95% CI: [{diff['ci_low']:+.3f}, {diff['ci_high']:+.3f}]  "
            f"沒贏過QQQ比例: {diff['prob_strategy_not_better']:.1%}  => {verdict}"
        )

    print(
        f"\n（對照：QQQ 買進持有同期間總報酬 {qqq_bench['total_return']:.2%}、"
        f"CAGR {qqq_bench['cagr']:.2%}、MDD {qqq_bench['max_dd']:.2%}、"
        f"Sharpe {qqq_bench['sharpe']:.2f}、Calmar {qqq_bench['calmar']:.2f}）"
    )


def main() -> None:
    prices = load_us_prices_snapshot()
    membership = load_us_index_membership_snapshot()

    if prices.empty:
        print("快照裡沒有任何美股價量資料。", file=sys.stderr)
        sys.exit(1)

    print(
        f"讀到 {prices['stock_id'].nunique()} 檔美股（{prices['date'].min().date()} ~ {prices['date'].max().date()}）"
    )
    print(f"固定參數：top_n={TOP_N}、ascending=True（買跌最多的）、momentum_window 網格：{MOMENTUM_WINDOW_GRID}")
    print("（rebalance_freq_days 跟 momentum_window 用同一個數字，對應 IC 掃描『trailing L天 -> forward L天』的設計）")

    base_cfg = build_us_config()

    _print_period(
        "樣本內（2023-09-19 ~ 資料庫最新日期）", "樣本內", prices, base_cfg, IN_SAMPLE_START, None, QQQ_IN_SAMPLE, membership
    )
    _print_period(
        "樣本外（2018-09-20 ~ 2023-09-18）", "樣本外", prices, base_cfg, OOS_START, OOS_END, QQQ_OOS, membership
    )

    print(
        "\n（誠實揭露：魚池沿用 build_pool_mask 的『站上60日均線』多頭排列篩選，"
        "跟買最近跌最多股票的反轉邏輯方向上有張力，見檔頭說明；"
        "QQQ 對照用的是本機既有的 strategy_equity_curves.parquet 快照，"
        "跟這支腳本本身用的 us_prices 快照可能不是同一次匯出，當作同時期近似值看待）"
    )


if __name__ == "__main__":
    main()
