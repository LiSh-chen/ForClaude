"""RSI/布林通道均值回歸在美股上的意外好結果（見
scripts/test_us_rsi_bollinger_reversion_from_db.py 和
docs/research_findings.md 第10.3節）需要樣本外驗證才能信任——
原本只有 2023-09~2026-09 這一段 3 年的資料，無法排除「這段特定
市場環境巧合」的可能。

做法：回填程式（scripts/ingest_us_daily_data.py）拉長到 8 年後，把
2023-09-18 之前的部分當成「樣本外」（之前從沒看過、也沒用來挑
參數的區間），用跟原本完全相同的 36 組參數網格（沒有重新調參）重跑一次。
重用 test_us_rsi_bollinger_reversion_from_db.py 的排名函數，避免邏輯重複。

反未來函數議题：run_factor_backtest 的 start_date/end_date 只限制「哪些日期允許
實際調倉」，魚池篩選與排名指標的計算永遠用傳進來的完整 prices 算（不會因為
切片而重新累積暨期），所以這裡永遠傳完整 us_prices，只用 end_date 限制交易只發生在
樣本外區間。

用法：
    python scripts/test_us_rsi_bollinger_out_of_sample_from_db.py
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_us_rsi_bollinger_reversion_from_db import (  # noqa: E402
    make_bollinger_ranking_signal_fn,
    rsi_ranking_signal_fn,
)

IN_SAMPLE_START = pd.Timestamp("2023-09-18")  # 之前已經測過、拿來找紐好組合的區間起點
REBALANCE_FREQ_GRID = (3, 5, 10, 21)
TOP_N_GRID = (10, 20, 30)
MIN_TRADES_FOR_RANKING = 15

HEADER = (
    f"{'signal':<20} {'hold_d':>7} {'top_n':>6}  {'total_ret':>10} {'cagr':>8} {'max_dd':>8} "
    f"{'sharpe':>7} {'calmar':>7}  {'n_trd':>6} {'win%':>7}"
)


def _fmt_row(label: str, hold_days: int, top_n: int, m: dict) -> str:
    return (
        f"{label:<20} {hold_days:>7} {top_n:>6}  "
        f"{m['total_return']:>9.2%} {m['cagr']:>7.2%} {m['max_dd']:>7.2%} {m['sharpe']:>7.2f} {m['calmar']:>7.2f}  "
        f"{m['n_trades']:>6.0f} {m['win_rate']:>6.1%}"
    )


def main() -> None:
    store = get_data_store()
    us_prices = store.load_us_prices()

    if us_prices.empty:
        print("資料庫裡沒有任何美股價量資料。", file=sys.stderr)
        sys.exit(1)

    earliest = us_prices["date"].min()
    if earliest >= IN_SAMPLE_START - pd.Timedelta(days=180):
        print(
            f"[warn] 資料庫最早日期只到 {earliest.date()}，離樣本內起點 {IN_SAMPLE_START.date()} "
            "不到半年，樣本外區間太短，結果不具參考價值。請先用 force_backfill=true 觸發 "
            "us_daily_data_ingest.yml 拉長歷史再重跑。",
            file=sys.stderr,
        )

    n_stocks = us_prices["stock_id"].nunique()
    print(
        f"讀到 {n_stocks} 檔美股的資料（{earliest.date()} ~ {us_prices['date'].max().date()}）\n"
        f"樣本外測試區間：{earliest.date()} ~ {(IN_SAMPLE_START - pd.Timedelta(days=1)).date()}\n"
        f"（樣本內區間 {IN_SAMPLE_START.date()} ~ 2026-09-17 的結果已在正式報告中，這裡完全不重複、"
        "不重新調參，用同一組固定參數網格套在沒看過的更早期間）\n"
    )

    base_cfg = build_us_config()

    print("=== RSI/布林通道均值回歸 樣本外驗證（美股 S&P 500，完整交易成本）===")
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
                    us_prices,
                    base_cfg,
                    factor_cfg,
                    end_date=IN_SAMPLE_START - pd.Timedelta(days=1),
                    signal_fn=fn,
                    cost_module=us_costs,
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
        "\n（對照：同一組參數網格在樣本內區間 2023-09~2026-09 的結果是 36/36 組全部正報酬、"
        "最佳 Sharpe 1.55——這裡是完全沒調過參數、沒看過的更早期間，是否還維持這個一致性，"
        "直接決定這個發現是真的市場結構優勢還是特定期間的巧合）"
    )


if __name__ == "__main__":
    main()
