"""把 RSI 超賣 / 布林通道乖離反彈這兩個短期均值回歸訊號重新套用在美股
S&P 500 資料上，跟台股版（scripts/test_rsi_bollinger_reversion_from_db.py，
結果是否證：36 組只有 5 組正報酬，最佳 Sharpe 0.41）做同樣的網格搜尋。

沿用台股版一樣的因子式定期調倉引擎（tw_quant/factor_backtest.py）跟排名
函式（RSI(14)/布林通道(20日)），差異只有資料來源換成 us_prices、成本模型
換成美股版（tw_quant/us_config.py + tw_quant/us_costs.py）。

2026-09-21 架構修正：這支腳本先前直接連 get_data_store()/store.
load_us_prices()（本機空的 SQLite DB，這個沙盒環境完全連不到資料、
根本跑不動），也完全沒有處理 S&P 500 存活者偏差。改成跟其他美股腳本
一致，讀本機快照（load_us_prices_snapshot()/
load_us_index_membership_snapshot()），並把 membership 參數傳進
run_factor_backtest（詳見 tw_quant/us_universe.py 檔頭）。

用法：
    python scripts/test_us_rsi_bollinger_reversion_from_db.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tw_quant import us_costs
from tw_quant.backtest_stats import metrics_from_result
from tw_quant.data_snapshot import load_us_index_membership_snapshot, load_us_prices_snapshot
from tw_quant.factor_backtest import FactorConfig, run_factor_backtest
from tw_quant.us_config import build_us_config

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_rsi_bollinger_reversion_from_db import (  # noqa: E402
    make_bollinger_ranking_signal_fn,
    rsi_ranking_signal_fn,
)

REBALANCE_FREQ_GRID = (3, 5, 10, 21)
TOP_N_GRID = (10, 20, 30)
MIN_TRADES_FOR_RANKING = 15

HEADER = (
    f"{'signal':<20} {'hold_d':>7} {'top_n':>6}  {'total_ret':>10} {'cagr':>8} {'max_dd':>8} "
    f"{'sharpe':>7} {'calmar':>7}  {'n_trd':>6} {'win%':>7} {'RR':>5} {'EV%':>8} {'PF':>6}"
)


def _fmt_row(label: str, hold_days: int, top_n: int, m: dict) -> str:
    pf = m["profit_factor"]
    pf_str = "  inf" if pf == float("inf") else f"{pf:5.2f}"
    rr = m["risk_reward_ratio"]
    rr_str = " inf" if rr == float("inf") else f"{rr:4.2f}"
    return (
        f"{label:<20} {hold_days:>7} {top_n:>6}  "
        f"{m['total_return']:>9.2%} {m['cagr']:>7.2%} {m['max_dd']:>7.2%} {m['sharpe']:>7.2f} {m['calmar']:>7.2f}  "
        f"{m['n_trades']:>6.0f} {m['win_rate']:>6.1%} {rr_str} {m['ev_pct']:>7.2%} {pf_str}"
    )


def main() -> None:
    us_prices = load_us_prices_snapshot()
    membership = load_us_index_membership_snapshot()

    if us_prices.empty:
        print("快照裡沒有任何美股價量資料（data/us_prices_snapshot.parquet 是空的）。", file=sys.stderr)
        sys.exit(1)

    n_stocks = us_prices["stock_id"].nunique()
    print(
        f"讀到 {n_stocks} 檔美股的資料"
        f"（{us_prices['date'].min().date()} ~ {us_prices['date'].max().date()}）\n"
    )

    base_cfg = build_us_config()

    print("=== RSI 超賣 / 布林通道乖離反彈回測（美股 S&P 500、因子式定期調倉、完整交易成本）===")
    print(HEADER)

    signal_fns = {
        "RSI": rsi_ranking_signal_fn,
        "BB_zscore": make_bollinger_ranking_signal_fn(False),
        "BB_zscore+vol": make_bollinger_ranking_signal_fn(True),
    }

    rows = []
    for signal_name, fn in signal_fns.items():
        for hold_days in REBALANCE_FREQ_GRID:
            for top_n in TOP_N_GRID:
                factor_cfg = FactorConfig(rebalance_freq_days=hold_days, top_n=top_n, ascending=True)
                result = run_factor_backtest(
                    us_prices, base_cfg, factor_cfg, signal_fn=fn, cost_module=us_costs, membership=membership
                )
                m = metrics_from_result(result, base_cfg.initial_capital, prices=us_prices)
                rows.append({"signal": signal_name, "hold_days": hold_days, "top_n": top_n, **m})

    df = pd.DataFrame(rows)
    ranked = df[df["n_trades"] >= MIN_TRADES_FOR_RANKING].sort_values("sharpe", ascending=False)
    for _, r in ranked.head(25).iterrows():
        print(_fmt_row(r["signal"], int(r["hold_days"]), int(r["top_n"]), r.to_dict()))

    n_profitable = (df["total_return"] > 0).sum()
    print(f"\n{len(df)} 組合中有 {n_profitable} 組總報酬為正（{n_profitable / len(df):.1%}）")

    if ranked.empty:
        print("\n沒有任何組合達到最低成交筆數門檻，無法排名。")

    print(
        "\n（RR = 風報比；EV% = 勝率加權後單筆期望報酬率；PF = 獲利因子；calmar = CAGR / MDD；"
        "hold_d = 調倉週期（近似持有天數）；ascending=True 買排名最低（RSI最低/z-score最負，"
        "也就是最超賣）的股票；台股版對照：36 組只有 5 組正報酬，最佳 Sharpe 0.41）"
    )


if __name__ == "__main__":
    main()
