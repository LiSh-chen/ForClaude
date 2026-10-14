"""買 Top-N 月輪替動量策略，加上個股固定跌幅停損，績效會怎麼變？

使用者問的是最基本的假設測試：standard 的「排名前 N 強動量股、每月調倉」
策略（跟 test_us_dual_momentum_from_db.py 的「無濾網」基準列同一組參數：
momentum_window=126 約半年、rebalance_freq_days=21 約月調倉、top_n=20），
如果再加一層「個股從進場價跌超過某個百分比就停損」的風控，整體績效
（總報酬/CAGR/MDD/Sharpe/Calmar）會怎麼變化？這是 2026-10-14 剛加進
tw_quant/factor_backtest.py 的 FactorConfig.stop_loss_pct 新功能的第一個
真實資料應用（機制細節、反未來函數設計見該檔案的 docstring 跟
tests/test_factor_backtest.py 的停損測試）。

停損門檻網格刻意涵蓋從緊到寬：10%/15%/20%，加上 None（不停損）作基準，
不是調參找「最佳停損點」，是老實呈現「加這層風控、不管鬆緊，大概會
怎麼影響報酬跟風險」這個形狀。

跟本專案這一輪所有研究一致的紀律：不只看點估計，每個設定都用
tw_quant/robustness.py 的移動區塊拔靴法量化 Sharpe 對 QQQ 買進持有的
優勢是不是禁得起抽樣不確定性（QQQ 對照用本機既有的
data/strategy_equity_curves.parquet 快照，這個沙箱環境網路被擋，
沒辦法像 test_us_dual_momentum_from_db.py 那樣即時抓 QQQ）。

用法：
    python scripts/test_us_topn_stop_loss_from_db.py
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

MOM_WINDOW = 126
REBALANCE_FREQ_DAYS = 21
TOP_N = 20
STOP_LOSS_GRID = (None, 0.10, 0.15, 0.20)  # None = 不停損的基準

IN_SAMPLE_START = "2023-09-19"
OOS_START = "2018-09-20"
OOS_END = pd.Timestamp(IN_SAMPLE_START) - pd.Timedelta(days=1)

# 跟 scripts/test_us_dual_momentum_from_db.py 等既有腳本同一組硬編碼參考值
# （2026-09-21 membership 架構修正後、GitHub Actions 實跑的結果），只是拿來
# 在彙整表裡列出對照，不是這支腳本自己重新抓的。
QQQ_IN_SAMPLE = {"total_return": 0.9358, "cagr": 0.2481, "max_dd": 0.2277, "sharpe": 1.19, "calmar": 1.09}
QQQ_OOS = {"total_return": 1.0760, "cagr": 0.1580, "max_dd": 0.3512, "sharpe": 0.69, "calmar": 0.45}

N_BOOT = 5000
SEED = 42


def _run(prices: pd.DataFrame, base_cfg, stop_loss_pct: float | None, start_date, end_date, membership):
    factor_cfg = FactorConfig(
        momentum_window=MOM_WINDOW,
        rebalance_freq_days=REBALANCE_FREQ_DAYS,
        top_n=TOP_N,
        ascending=False,
        stop_loss_pct=stop_loss_pct,
    )
    result = run_factor_backtest(
        prices, base_cfg, factor_cfg, start_date=start_date, end_date=end_date, cost_module=us_costs, membership=membership
    )
    metrics = metrics_from_result(result, base_cfg.initial_capital, prices=prices)
    n_stop_loss = 0
    if not result.trades.empty and "exit_reason" in result.trades.columns:
        n_stop_loss = int((result.trades["exit_reason"] == "STOP_LOSS").sum())
    return result, metrics, n_stop_loss


def _load_qqq_equity(period_label: str) -> pd.Series:
    df = pd.read_parquet("data/strategy_equity_curves.parquet")
    sub = df[(df["period"] == period_label) & (df["strategy"] == "QQQ")].sort_values("date")
    return sub.set_index("date")["equity"]


def _print_period(label: str, period_label_for_qqq: str, prices, base_cfg, start_date, end_date, qqq_bench, membership) -> None:
    print(f"\n=== {label} ===")
    header = (
        f"{'stop_loss':>9} {'total_ret':>10} {'cagr':>8} {'max_dd':>8} "
        f"{'sharpe':>7} {'calmar':>7} {'n_trd':>6} {'n_stop':>6} {'win%':>7}"
    )
    print(header)

    qqq_equity = _load_qqq_equity(period_label_for_qqq)
    qqq_returns = daily_returns_from_equity(qqq_equity)

    for stop_loss_pct in STOP_LOSS_GRID:
        result, m, n_stop_loss = _run(prices, base_cfg, stop_loss_pct, start_date, end_date, membership)
        label_sl = "無停損" if stop_loss_pct is None else f"{stop_loss_pct:.0%}"
        print(
            f"{label_sl:>9} {m['total_return']:>9.2%} {m['cagr']:>7.2%} {m['max_dd']:>7.2%} "
            f"{m['sharpe']:>7.2f} {m['calmar']:>7.2f} {m['n_trades']:>6.0f} {n_stop_loss:>6} {m['win_rate']:>6.1%}"
        )

        strategy_returns = daily_returns_from_equity(result.equity_curve["equity"])
        block_size = REBALANCE_FREQ_DAYS
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

    print(f"讀到 {prices['stock_id'].nunique()} 檔美股（{prices['date'].min().date()} ~ {prices['date'].max().date()}）")
    print(
        f"固定動量參數（跟 test_us_dual_momentum_from_db.py 的「無濾網」基準列同一組，非調參最佳解）："
        f"momentum_window={MOM_WINDOW}、rebalance_freq_days={REBALANCE_FREQ_DAYS}、top_n={TOP_N}；"
        f"停損門檻網格：{STOP_LOSS_GRID}"
    )

    base_cfg = build_us_config()

    _print_period(
        "樣本內（2023-09-19 ~ 資料庫最新日期）", "樣本內", prices, base_cfg, IN_SAMPLE_START, None, QQQ_IN_SAMPLE, membership
    )
    _print_period(
        "樣本外（2018-09-20 ~ 2023-09-18）", "樣本外", prices, base_cfg, OOS_START, OOS_END, QQQ_OOS, membership
    )

    print(
        "\n（誠實揭露：QQQ 對照用的是本機既有的 strategy_equity_curves.parquet 快照，"
        "跟這支腳本本身用的 us_prices 快照可能不是同一次匯出，當作同時期近似值看待；"
        "停損觸發用收盤價、成交在次日開盤，跟 tw_quant/backtest.py 既有的吊燈停利同一套"
        "反未來函數慣例，細節見 tw_quant/factor_backtest.py 檔頭）"
    )


if __name__ == "__main__":
    main()
