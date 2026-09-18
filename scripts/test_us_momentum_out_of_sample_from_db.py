"""動量策略（買最近報酬率最高的股票）的樣本外驗證。

跟 test_us_rsi_bollinger_out_of_sample_from_db.py 用同一套方法論：重用
scripts/test_us_momentum_strategy_from_db.py 完全相同、沒有重新調參的
24 組網格，套在挑參數時從沒看過的 2018-09-20~2023-09-17 這段期間（也就是
原始腳本拿來跟 QQQ 對照的 2023-09-19~2026-09-17 樣本內窗格「之前」的
全部歷史）。

背景：樣本內最佳組合（momentum_window=252, hold=21, top_n=10）Sharpe
高達 2.10，24 組全部總報酬/Sharpe 贏過 QQQ 買進持有；但換到完整 8 年
資料做過一次不嚴謹的初步檢查，Sharpe 就掉到 1.06——這裡才是真正乾淨的
樣本外驗證：完全獨立的期間、完全相同的參數，不做任何調整。

反未來函數：永遠傳完整 us_prices（不切片），只用 end_date 限制交易只
發生在樣本外區間，動量排名計算永遠用完整歷史，不會因切片而重新累積
暖身期。

用法：
    python scripts/test_us_momentum_out_of_sample_from_db.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tw_quant import us_costs
from tw_quant.backtest_stats import metrics_from_result
from tw_quant.factor_backtest import FactorConfig, run_factor_backtest
from tw_quant.storage import get_data_store
from tw_quant.us_config import build_us_config
from tw_quant.us_universe import filter_prices_by_index_membership

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_us_momentum_strategy_from_db import (  # noqa: E402
    MOMENTUM_WINDOW_GRID,
    REBALANCE_FREQ_GRID,
    TOP_N_GRID,
    QQQ_START,
)

IN_SAMPLE_START = pd.Timestamp(QQQ_START)  # 原始腳本拿來跟 QQQ 對照、也用來挑參數的窗格起點
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
    store = get_data_store()
    us_prices = store.load_us_prices()
    membership = store.load_us_index_membership()

    if us_prices.empty:
        print("資料庫裡沒有任何美股價量資料。", file=sys.stderr)
        sys.exit(1)

    n_rows_before = len(us_prices)
    us_prices = filter_prices_by_index_membership(us_prices, membership)
    n_rows_dropped = n_rows_before - len(us_prices)
    print(
        f"存活者偏差部分修正：依指數加入日期過濾後，丟掉 {n_rows_dropped} / {n_rows_before} 列"
        f"（{n_rows_dropped / n_rows_before:.1%}）——這段樣本外期間（2018-2023）正是現在 503 檔裡"
        "24.5% 新進戶還沒加入指數的期間，過濾效果在這裡應該最明顯，"
        "不解決被剔除股票完全消失那一半（見 tw_quant/us_universe.py）\n"
    )

    earliest = us_prices["date"].min()
    n_stocks = us_prices["stock_id"].nunique()
    print(
        f"讀到 {n_stocks} 檔美股的資料（{earliest.date()} ~ {us_prices['date'].max().date()}）\n"
        f"樣本外測試區間：{earliest.date()} ~ {(IN_SAMPLE_START - pd.Timedelta(days=1)).date()}\n"
        f"（樣本內區間 {IN_SAMPLE_START.date()} 之後、跟 QQQ 對照的結果已在 "
        "test_us_momentum_strategy_from_db.py，這裡完全不重複、不重新調參，用同一組固定參數網格套在沒看過的更早期間）\n"
    )

    base_cfg = build_us_config()

    print("=== 動量策略 樣本外驗證（美股 S&P 500，完整交易成本）===")
    print(HEADER)

    rows = []
    for mom_win in MOMENTUM_WINDOW_GRID:
        for hold_days in REBALANCE_FREQ_GRID:
            for top_n in TOP_N_GRID:
                factor_cfg = FactorConfig(
                    momentum_window=mom_win, rebalance_freq_days=hold_days, top_n=top_n, ascending=False
                )
                result = run_factor_backtest(
                    us_prices, base_cfg, factor_cfg,
                    end_date=IN_SAMPLE_START - pd.Timedelta(days=1),
                    cost_module=us_costs,
                )
                m = metrics_from_result(result, base_cfg.initial_capital, prices=us_prices)
                rows.append({"mom_win": mom_win, "hold_days": hold_days, "top_n": top_n, **m})

    df = pd.DataFrame(rows)
    ranked = df[df["n_trades"] >= MIN_TRADES_FOR_RANKING].sort_values("sharpe", ascending=False)
    for _, r in ranked.head(15).iterrows():
        print(_fmt_row(int(r["mom_win"]), int(r["hold_days"]), int(r["top_n"]), r.to_dict()))

    n_profitable = (df["total_return"] > 0).sum()
    print(f"\n{len(df)} 組合中有 {n_profitable} 組總報酬為正（{n_profitable / len(df):.1%}）")

    if ranked.empty:
        print("\n沒有任何組合達到最低成交筆數門檻，無法排名。")

    print(
        "\n（對照：同一組參數網格在樣本內區間（2023-09~2026-09）的結果是 24/24 組全部正報酬、"
        "且全部贏過 QQQ 買進持有，最佳 Sharpe 2.10——這裡是完全沒調過參數、沒看過的更早期間，"
        "是否還維持這個一致性，直接決定這個發現是真的市場結構優勢還是特定期間的巧合）"
    )


if __name__ == "__main__":
    main()
