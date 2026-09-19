"""動量（買強勢股）策略原型：測試能不能打敗美股 QQQ 買進持有。

背景：RSI/布林通道均值回歸策略（樣本內 Sharpe 0.93）雖然絕對報酬為正，
但同期間（2023-09-19~2026-09-17）遠遠輸給單純買進持有 QQQ（總報酬
93.58% vs 策略 39.96%，見 docs/research_findings.md 第10.3節）。QQQ 這
三年的漲幅集中在少數幾檔巨型科技/AI股，均值回歸策略買的是弱勢股、賭
反彈，方向剛好跟「抱緊最強勢的少數贏家不放」相反——這裡改測動量策略
（買最近報酬率最高的股票），看能不能貼近或打敗這種集中度極高的牛市。

不需要另外寫訊號函式：tw_quant.factor_backtest.run_factor_backtest 在
signal_fn=None 時，預設排名依據就是「T-1 為止 momentum_window 日報酬率」
（見該檔案 signal_fn 說明），ascending=False（預設）就是買排名最高（動量
最強）的股票，正好是這裡要測的邏輯，直接重用即可。

反未來函數：跟 test_us_rsi_bollinger_out_of_sample_from_db.py 用同一套
規則——永遠傳完整 us_prices（不切片），只用 start_date 限制「哪些日期
允許實際調倉」，動量排名計算永遠用完整歷史，不會因切片而重新累積暖身期。

兩段對照：
  1. QQQ_START 起 ~ 資料庫最新日期：跟 QQQ 買進持有同一段期間（見前述
     debug 診斷的 93.58%/24.81%/22.77%/1.19/1.09），直接比較。
  2. 完整 8 年資料：初步看看最佳組合換到涵蓋 2022 熊市、2020 COVID 崩盤
     的更長期間還站不站得住——這不是嚴謹的樣本外驗證（參數是在同一次
     跑裡選出來的），只是第一輪篩選，真正決定要不要信任還需要後續正式
     的樣本外切分。

用法：
    python scripts/test_us_momentum_strategy_from_db.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tw_quant import us_costs
from tw_quant.backtest_stats import metrics_from_result
from tw_quant.factor_backtest import FactorConfig, run_factor_backtest
from tw_quant.data_snapshot import load_us_index_membership_snapshot, load_us_prices_snapshot
from tw_quant.us_config import build_us_config
from tw_quant.us_universe import filter_prices_by_index_membership

QQQ_START = "2023-09-19"
QQQ_TOTAL_RETURN, QQQ_CAGR, QQQ_MDD, QQQ_SHARPE, QQQ_CALMAR = 0.9358, 0.2481, 0.2277, 1.19, 1.09

MOMENTUM_WINDOW_GRID = (21, 63, 126, 252)  # 約 1 / 3 / 6 / 12 個月
REBALANCE_FREQ_GRID = (21, 63)  # 約月調倉 / 季調倉
TOP_N_GRID = (10, 20, 30)
MIN_TRADES_FOR_RANKING = 10

HEADER = (
    f"{'mom_win':>8} {'hold_d':>7} {'top_n':>6}  {'total_ret':>10} {'cagr':>8} {'max_dd':>8} "
    f"{'sharpe':>7} {'calmar':>7}  {'n_trd':>6} {'win%':>7}"
)


def _fmt_row(mom_win: int, hold_days: int, top_n: int, m: dict) -> str:
    return (
        f"{mom_win:>8} {hold_days:>7} {top_n:>6}  "
        f"{m['total_return']:>9.2%} {m['cagr']:>7.2%} {m['max_dd']:>7.2%} {m['sharpe']:>7.2f} {m['calmar']:>7.2f}  "
        f"{m['n_trades']:>6.0f} {m['win_rate']:>6.1%}"
    )


def main() -> None:
    us_prices = load_us_prices_snapshot()
    membership = load_us_index_membership_snapshot()

    if us_prices.empty:
        print("快照裡沒有任何美股價量資料（data/us_prices_snapshot.parquet 是空的）。", file=sys.stderr)
        sys.exit(1)

    n_stocks_before = us_prices["stock_id"].nunique()
    n_rows_before = len(us_prices)
    us_prices = filter_prices_by_index_membership(us_prices, membership)
    n_rows_dropped = n_rows_before - len(us_prices)
    print(
        f"存活者偏差部分修正：依指數加入日期過濾後，丟掉 {n_rows_dropped} / {n_rows_before} 列"
        f"（{n_rows_dropped / n_rows_before:.1%}，只排除「當時還沒加入指數」的日期，"
        "不解決被剔除股票完全消失那一半，見 tw_quant/us_universe.py）\n"
    )

    n_stocks = us_prices["stock_id"].nunique()
    earliest, latest = us_prices["date"].min(), us_prices["date"].max()
    print(f"讀到 {n_stocks_before} 檔美股的資料，過濾後 {n_stocks} 檔仍有資料（{earliest.date()} ~ {latest.date()}）\n")

    base_cfg = build_us_config()

    print(f"=== 動量策略（買最近報酬率最高的股票）：{QQQ_START} ~ {latest.date()}，跟 QQQ 買進持有同期間 ===")
    print(HEADER)

    rows = []
    for mom_win in MOMENTUM_WINDOW_GRID:
        for hold_days in REBALANCE_FREQ_GRID:
            for top_n in TOP_N_GRID:
                factor_cfg = FactorConfig(
                    momentum_window=mom_win, rebalance_freq_days=hold_days, top_n=top_n, ascending=False
                )
                result = run_factor_backtest(
                    us_prices, base_cfg, factor_cfg, start_date=QQQ_START, cost_module=us_costs
                )
                m = metrics_from_result(result, base_cfg.initial_capital, prices=us_prices)
                rows.append({"mom_win": mom_win, "hold_days": hold_days, "top_n": top_n, **m})

    df = pd.DataFrame(rows)
    ranked = df[df["n_trades"] >= MIN_TRADES_FOR_RANKING].sort_values("sharpe", ascending=False)
    for _, r in ranked.head(15).iterrows():
        print(_fmt_row(int(r["mom_win"]), int(r["hold_days"]), int(r["top_n"]), r.to_dict()))

    n_beats_qqq_return = (df["total_return"] > QQQ_TOTAL_RETURN).sum()
    n_beats_qqq_sharpe = (df["sharpe"] > QQQ_SHARPE).sum()
    print(f"\n{len(df)} 組合中：{n_beats_qqq_return} 組總報酬贏過 QQQ 買進持有（{QQQ_TOTAL_RETURN:.2%}）")
    print(f"{len(df)} 組合中：{n_beats_qqq_sharpe} 組 Sharpe 贏過 QQQ 買進持有（{QQQ_SHARPE:.2f}）")
    print(
        f"\n（對照：QQQ 買進持有同期間總報酬 {QQQ_TOTAL_RETURN:.2%}、CAGR {QQQ_CAGR:.2%}、"
        f"MDD {QQQ_MDD:.2%}、Sharpe {QQQ_SHARPE:.2f}、Calmar {QQQ_CALMAR:.2f}）"
    )

    if ranked.empty:
        print("\n沒有任何組合達到最低成交筆數門檻，無法排名。")
        return

    best = ranked.iloc[0]
    best_mom_win, best_hold, best_top_n = int(best["mom_win"]), int(best["hold_days"]), int(best["top_n"])
    print(
        f"\n=== 最佳組合（momentum_window={best_mom_win}, hold={best_hold}, top_n={best_top_n}）"
        f"換到完整 8 年資料（{earliest.date()} ~ {latest.date()}，含 2022 熊市/2020 COVID 崩盤）初步檢查 ==="
    )
    full_cfg = FactorConfig(momentum_window=best_mom_win, rebalance_freq_days=best_hold, top_n=best_top_n, ascending=False)
    full_result = run_factor_backtest(us_prices, base_cfg, full_cfg, cost_module=us_costs)
    full_m = metrics_from_result(full_result, base_cfg.initial_capital, prices=us_prices)
    print(HEADER)
    print(_fmt_row(best_mom_win, best_hold, best_top_n, full_m))
    print(
        "\n（這不是正式的樣本外驗證——參數是從跟這裡完全相同的資料窗格選出來的，只是先看換一段更長、"
        "涵蓋熊市的期間會不會立刻明顯崩潰；真的要信任還需要另外切一段沒用來選參數的期間重跑，"
        "比照 test_us_rsi_bollinger_out_of_sample_from_db.py 的做法）"
    )


if __name__ == "__main__":
    main()
