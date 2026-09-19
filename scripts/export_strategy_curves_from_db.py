"""把 10.5~10.16 節探討過的代表性策略/組合，統一算出逐日資產淨值曲線
+ 成交紀錄，寫成兩份長格式 parquet（data/strategy_equity_curves.parquet、
data/strategy_trades.parquet），供前端圖表把不同策略疊在同一張圖上比較。

使用者明確要求「所有探討成果」要有真實的總資產變化、持股/股價變化、
交易紀錄，不能只用摘要指標湊——這裡不是重新設計新策略，是把前面每支
_from_db.py 腳本「本來就已經算出來但沒有存下來」的 BacktestResult.
equity_curve / trades（見 tw_quant/backtest.py）跟 ETF 組合模擬的市值
序列，用 tw_quant/curve_export.py 收斂成同一種格式。

不是每一個測過的網格組合都在這裡重跑——像 10.5 的 24 組、10.8 的 7 種
調倉頻率、10.15 的 25 組交叉網格，全部重新產生逐日曲線意義不大（線太多、
彼此高度相似），這裡只挑每一節的代表性設定（跟先前報告給使用者的數字
用同一組參數，不是另外重新調參）：

  - QQQ/VOO/VTI/VT 買進持有（10.14）
  - 四檔等額不再平衡、QQQ90/10、QQQ50/50（不再平衡 + 每年再平衡對照）（10.16）
  - 10.10 等權重全魚池被動參照組
  - 10.11 top_n 網格全部 8 個值（1/2/3/5/10/20/30/50）——這節是使用者
    最關心的核心發現，全部保留
  - 10.7 雙動能最佳濾網（trend_ma=100）
  - 10.8 最佳調倉頻率（42天，top_n=20）
  - 10.9 最佳低波動窗格（126天）

網格類細節（例如 10.5 的完整 24 組、10.15 的完整 25 組交叉網格）留在
docs/research_findings.md 的表格跟前端的「參數比較」表格裡，不是這裡的
逐日曲線範圍——那些是給「掃一遍所有參數敏感度」用的，逐日曲線是給
「看這幾個關鍵候選人的資產變化長什麼樣子」用的，兩者互補、不是重複。

反未來函數：全部沿用各腳本原本的 run_factor_backtest 呼叫方式（永遠傳
完整 us_prices，只用 start_date/end_date 限制交易日期），沒有新增任何
切片邏輯。

用法：
    python scripts/export_strategy_curves_from_db.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tw_quant import us_costs
from tw_quant.curve_export import (
    buyhold_trade_row,
    equity_rows_from_result,
    equity_rows_from_series,
    trade_rows_from_fills,
    trade_rows_from_result,
)
from tw_quant.data_snapshot import load_us_index_membership_snapshot, load_us_prices_snapshot
from tw_quant.etf_combo import simulate_weighted_portfolio
from tw_quant.factor_backtest import FactorConfig, run_factor_backtest
from tw_quant.us_config import build_us_config
from tw_quant.us_data_provider import YFinanceUSDataProvider
from tw_quant.us_universe import filter_prices_by_index_membership

from scripts.test_us_dual_momentum_from_db import make_trend_filtered_momentum_signal_fn
from scripts.test_us_lowvol_factor_from_db import make_low_vol_signal_fn

MOM_WINDOW = 126
REBALANCE_FREQ_DAYS = 21
TOP_N_GRID = (1, 2, 3, 5, 10, 20, 30, 50)
BEST_TREND_MA = 100
BEST_REBALANCE_FREQ_DAYS = 42
BEST_VOL_WINDOW = 126

ETF_TICKERS = ["QQQ", "VOO", "VTI", "VT"]

IN_SAMPLE_START = "2023-09-19"
OOS_END = pd.Timestamp(IN_SAMPLE_START) - pd.Timedelta(days=1)
PERIODS = [("樣本內", IN_SAMPLE_START, None), ("樣本外", None, OOS_END)]

EQUITY_PATH = Path(__file__).resolve().parents[1] / "data" / "strategy_equity_curves.parquet"
TRADES_PATH = Path(__file__).resolve().parents[1] / "data" / "strategy_trades.parquet"


def _run_factor(label, us_prices, base_cfg, factor_cfg, signal_fn, start_date, end_date, period_label, equity_rows, trade_rows):
    result = run_factor_backtest(
        us_prices, base_cfg, factor_cfg, start_date=start_date, end_date=end_date,
        signal_fn=signal_fn, cost_module=us_costs,
    )
    equity_rows.append(equity_rows_from_result(period_label, label, result))
    trade_rows.append(trade_rows_from_result(period_label, label, result, prices=us_prices))
    print(f"  {label}：{len(result.equity_curve)} 個交易日、{len(result.trades)} 筆已平倉交易")


def _run_etf_combo(label, prices, weights, cost_cfg, initial_capital, rebalance_freq_days, period_label, equity_rows, trade_rows):
    values, n_rebal, fills = simulate_weighted_portfolio(prices, weights, cost_cfg, initial_capital, rebalance_freq_days)
    equity_rows.append(equity_rows_from_series(period_label, label, values))
    last_prices = {t: float(prices[t].iloc[-1]) for t in weights}
    trade_rows.append(trade_rows_from_fills(period_label, label, fills, prices.index[-1], last_prices))
    print(f"  {label}：{len(values)} 個交易日、再平衡 {n_rebal} 次、{len(fills)} 筆成交")


def main() -> None:
    us_prices = load_us_prices_snapshot()
    membership = load_us_index_membership_snapshot()
    if us_prices.empty:
        print("快照裡沒有任何美股價量資料（data/us_prices_snapshot.parquet 是空的）。", file=sys.stderr)
        sys.exit(1)

    us_prices = filter_prices_by_index_membership(us_prices, membership)
    earliest, latest = us_prices["date"].min(), us_prices["date"].max()
    print(f"讀到 {us_prices['stock_id'].nunique()} 檔美股的資料（{earliest.date()} ~ {latest.date()}）")

    provider = YFinanceUSDataProvider()
    etf_closes = {}
    for ticker in ETF_TICKERS:
        print(f"抓取 {ticker} 價格歷史...")
        df = provider.fetch_price(
            ticker, start_date=str(earliest.date()), end_date=str((latest + pd.Timedelta(days=1)).date()), industry="ETF"
        )
        if df.empty:
            print(f"{ticker} 資料抓取失敗，中止。", file=sys.stderr)
            sys.exit(1)
        etf_closes[ticker] = df.sort_values("date").reset_index(drop=True)

    base_cfg = build_us_config()
    equity_rows: list[pd.DataFrame] = []
    trade_rows: list[pd.DataFrame] = []

    for period_label, start_date, end_date in PERIODS:
        print(f"\n=== {period_label} ===")

        for ticker in ETF_TICKERS:
            df = etf_closes[ticker]
            lo = pd.Timestamp(start_date) if start_date is not None else df["date"].min()
            hi = pd.Timestamp(end_date) if end_date is not None else df["date"].max()
            clipped = df[(df["date"] >= lo) & (df["date"] <= hi)].sort_values("date").reset_index(drop=True)
            equity_rows.append(equity_rows_from_series(period_label, ticker, clipped.set_index("date")["close"]))
            trade_rows.append(buyhold_trade_row(period_label, ticker, ticker, clipped))
            print(f"  {ticker}（買進持有）：{len(clipped)} 個交易日")

        aligned = pd.concat({t: etf_closes[t].set_index("date")["close"] for t in ETF_TICKERS}, axis=1).dropna()
        lo = pd.Timestamp(start_date) if start_date is not None else aligned.index.min()
        hi = pd.Timestamp(end_date) if end_date is not None else aligned.index.max()
        combo_prices = aligned[(aligned.index >= lo) & (aligned.index <= hi)]

        combo_defs = [
            ("四檔等額不再平衡", {"QQQ": 0.25, "VOO": 0.25, "VTI": 0.25, "VT": 0.25}, None),
            ("QQQ90%/VOO10%不再平衡", {"QQQ": 0.9, "VOO": 0.1}, None),
            ("QQQ50%/VOO50%不再平衡", {"QQQ": 0.5, "VOO": 0.5}, None),
            ("QQQ50%/VOO50%每年再平衡", {"QQQ": 0.5, "VOO": 0.5}, 252),
        ]
        for label, weights, freq in combo_defs:
            cols = list(weights.keys())
            _run_etf_combo(
                label, combo_prices[cols], weights, base_cfg.costs, base_cfg.initial_capital, freq,
                period_label, equity_rows, trade_rows,
            )

        _run_factor(
            "等權重全魚池(10.10)", us_prices, base_cfg,
            FactorConfig(momentum_window=21, rebalance_freq_days=21, top_n=999, ascending=False),
            None, start_date, end_date, period_label, equity_rows, trade_rows,
        )

        for top_n in TOP_N_GRID:
            _run_factor(
                f"top_n={top_n}", us_prices, base_cfg,
                FactorConfig(momentum_window=MOM_WINDOW, rebalance_freq_days=REBALANCE_FREQ_DAYS, top_n=top_n, ascending=False),
                None, start_date, end_date, period_label, equity_rows, trade_rows,
            )

        qqq = etf_closes["QQQ"]
        _run_factor(
            f"雙動能(濾網MA{BEST_TREND_MA})", us_prices, base_cfg,
            FactorConfig(momentum_window=MOM_WINDOW, rebalance_freq_days=REBALANCE_FREQ_DAYS, top_n=20, ascending=False),
            make_trend_filtered_momentum_signal_fn(qqq, BEST_TREND_MA), start_date, end_date,
            period_label, equity_rows, trade_rows,
        )

        _run_factor(
            f"調倉{BEST_REBALANCE_FREQ_DAYS}天(top_n=20)", us_prices, base_cfg,
            FactorConfig(momentum_window=MOM_WINDOW, rebalance_freq_days=BEST_REBALANCE_FREQ_DAYS, top_n=20, ascending=False),
            None, start_date, end_date, period_label, equity_rows, trade_rows,
        )

        _run_factor(
            f"低波動因子(vol_win={BEST_VOL_WINDOW})", us_prices, base_cfg,
            FactorConfig(rebalance_freq_days=REBALANCE_FREQ_DAYS, top_n=20, ascending=True),
            make_low_vol_signal_fn(BEST_VOL_WINDOW), start_date, end_date,
            period_label, equity_rows, trade_rows,
        )

    equity_df = pd.concat(equity_rows, ignore_index=True)
    trades_df = pd.concat(trade_rows, ignore_index=True)

    EQUITY_PATH.parent.mkdir(parents=True, exist_ok=True)
    equity_df.to_parquet(EQUITY_PATH, index=False)
    trades_df.to_parquet(TRADES_PATH, index=False)

    print(f"\n已匯出 {len(equity_df)} 列權益曲線 -> {EQUITY_PATH}")
    print(f"已匯出 {len(trades_df)} 列成交紀錄 -> {TRADES_PATH}")
    print(f"策略數：{equity_df['strategy'].nunique()}，期間數：{equity_df['period'].nunique()}")


if __name__ == "__main__":
    main()
