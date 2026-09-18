"""RSI/布林通道均值回歸策略單獨在 2022 熊市窗格的表現。

背景：樣本外驗證（2018-09~2023-09，見
test_us_rsi_bollinger_out_of_sample_from_db.py）已經否證這個策略的
「美股結構性優勢」解讀——36 組只剩 5 組正報酬，最佳 Sharpe 0.39，幾乎
跟台股原版否證數字一樣。但那次驗證涵蓋整整 5 年，把 2022 熊市稀釋在
更長的期間裡看不出單獨表現。這裡把窗格縮小到 S&P 500 公認的 2022 熊市
本身（2022-01-03 高點 ~ 2022-10-12 低點，跌約 25%），測試「均值回歸
策略在熊市裡可能因為『跌深就反彈』邏輯而相對抗跌」這個假設——如果
成立，應該會看到虧損幅度明顯小於大盤同期跌幅，甚至轉正。

用跟正式報告、樣本外驗證完全相同的參數網格，不重新調參。

反未來函數：永遠傳完整 us_prices（不切片），用 start_date/end_date
限制交易只發生在熊市窗格內，排名指標計算永遠用完整歷史。

用法：
    python scripts/test_us_rsi_bollinger_bear_market_from_db.py
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

BEAR_START = pd.Timestamp("2022-01-03")  # S&P 500 這輪熊市公認高點
BEAR_END = pd.Timestamp("2022-10-12")  # S&P 500 這輪熊市公認低點（跌約25%）
REBALANCE_FREQ_GRID = (3, 5, 10, 21)
TOP_N_GRID = (10, 20, 30)
MIN_TRADES_FOR_RANKING = 8  # 熊市窗格只有約9個月，調低門檻避免組合被濾光

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
    if earliest > BEAR_START:
        print(
            f"[warn] 資料庫最早日期只到 {earliest.date()}，晚於熊市窗格起點 {BEAR_START.date()}，"
            "結果會缺開頭一段，不完整。",
            file=sys.stderr,
        )

    n_stocks = us_prices["stock_id"].nunique()
    print(
        f"讀到 {n_stocks} 檔美股的資料（{earliest.date()} ~ {us_prices['date'].max().date()}）\n"
        f"熊市測試區間：{BEAR_START.date()} ~ {BEAR_END.date()}（S&P 500 公認 2022 熊市高點到低點）\n"
    )

    base_cfg = build_us_config()

    print("=== RSI/布林通道均值回歸 2022 熊市窗格表現（美股 S&P 500，完整交易成本）===")
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
                    us_prices, base_cfg, factor_cfg,
                    start_date=BEAR_START, end_date=BEAR_END,
                    signal_fn=fn, cost_module=us_costs,
                )
                m = metrics_from_result(result, base_cfg.initial_capital, prices=us_prices)
                rows.append({"signal": signal_name, "hold_days": hold_days, "top_n": top_n, **m})

    df = pd.DataFrame(rows)
    ranked = df[df["n_trades"] >= MIN_TRADES_FOR_RANKING].sort_values("sharpe", ascending=False)
    for _, r in ranked.head(15).iterrows():
        print(_fmt_row(r["signal"], int(r["hold_days"]), int(r["top_n"]), r.to_dict()))

    n_profitable = (df["total_return"] > 0).sum()
    print(f"\n{len(df)} 組合中有 {n_profitable} 組總報酬為正（{n_profitable / len(df):.1%}）")

    if ranked.empty:
        print("\n沒有任何組合達到最低成交筆數門檻，無法排名，以下是全部組合：")
        print(df.to_string(index=False))

    print(
        "\n（這組參數網格跟正式報告/樣本外驗證完全相同，沒有為了熊市窗格重新調參；"
        "對照：S&P 500 這段期間大盤本身跌約25%）"
    )


if __name__ == "__main__":
    main()
