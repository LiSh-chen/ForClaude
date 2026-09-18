"""一次性診斷：QQQ 買進持有在三段不同期間的表現，一次抓齊，供分別跟
動量策略、RSI/布林通道策略的回測結果對照：
  1. 完整 8 年（2018-09-20~2026-09-17）——跟動量策略「換到完整8年初步檢查」對照
  2. 動量策略樣本外驗證期間（2018-09-20~2023-09-17，前5年）——跟
     test_us_momentum_out_of_sample_from_db.py 對照
  3. 2022 熊市（2022-01-03~2022-10-12）——跟
     test_us_rsi_bollinger_bear_market_from_db.py 對照

只抓一次完整歷史再切片，不對同一檔股票重複打 API。用跟
tw_quant/us_data_provider.py 一致的 auto_adjust=True。

用法：
    python scripts/debug_qqq_multi_period_buyhold.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf

PERIODS = {
    "完整8年": ("2018-09-20", "2026-09-17"),
    "動量策略樣本外期間（前5年）": ("2018-09-20", "2023-09-17"),
    "2022熊市（S&P500公認高點~低點）": ("2022-01-03", "2022-10-12"),
}


def compute_stats(close: pd.Series) -> dict:
    total_return = close.iloc[-1] / close.iloc[0] - 1
    n_days = len(close)
    years = max(n_days / 252, 1e-9)
    cagr = (close.iloc[-1] / close.iloc[0]) ** (1 / years) - 1

    running_peak = close.cummax()
    drawdown = (running_peak - close) / running_peak
    max_dd = drawdown.max()

    daily_ret = close.pct_change().dropna()
    sharpe = (daily_ret.mean() / daily_ret.std()) * np.sqrt(252) if daily_ret.std() > 0 else 0.0
    calmar = (cagr / max_dd) if max_dd > 0 else 0.0

    return {
        "total_return": total_return, "cagr": cagr, "max_dd": max_dd,
        "sharpe": sharpe, "calmar": calmar, "n_days": n_days,
    }


def main() -> None:
    ticker = yf.Ticker("QQQ")
    full_hist = ticker.history(start="2018-09-20", end="2026-09-18", auto_adjust=True)
    if full_hist.empty:
        print("沒有抓到 QQQ 資料")
        return
    close_full = full_hist["Close"]
    close_full.index = pd.to_datetime(close_full.index).tz_localize(None)

    for label, (start, end) in PERIODS.items():
        mask = (close_full.index >= pd.Timestamp(start)) & (close_full.index <= pd.Timestamp(end))
        sub = close_full[mask]
        if sub.empty:
            print(f"=== {label}：沒有資料 ===\n")
            continue
        s = compute_stats(sub)
        print(f"=== QQQ 買進持有：{label}（{sub.index[0].date()} ~ {sub.index[-1].date()}，{s['n_days']} 個交易日）===")
        print(f"起始價: {sub.iloc[0]:.2f}  結束價: {sub.iloc[-1]:.2f}")
        print(
            f"總報酬: {s['total_return']:.2%}  CAGR: {s['cagr']:.2%}  MDD: {s['max_dd']:.2%}  "
            f"Sharpe: {s['sharpe']:.2f}  Calmar: {s['calmar']:.2f}"
        )
        print()


if __name__ == "__main__":
    main()
