"""一次性診斷腳本：算 QQQ（Nasdaq-100 ETF）在跟 RSI/布林通道美股回測
完全相同的日期區間（2023-09-19 ~ 2026-09-17，樣本內）單純買進持有的
報酬，拿來跟策略的 39.96% 總報酬對照，看策略有沒有真的打敗大盤，還是
只是搭了這段期間本來就很強的美股大盤順風車。

用跟 tw_quant/us_data_provider.py 完全一樣的 auto_adjust=True 設定抓
QQQ 歷史價格（股息/分割都已調整），跟回測引擎讀到的個股價格用同一種
基準，才是公平對照，不是隨便抓個報價比一比。

用法：
    python scripts/debug_qqq_buyhold_comparison.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf

START = "2023-09-19"
END = "2026-09-17"


def main() -> None:
    ticker = yf.Ticker("QQQ")
    hist = ticker.history(start=START, end=END, auto_adjust=True)
    if hist.empty:
        print("沒有抓到 QQQ 資料")
        return

    close = hist["Close"]
    close.index = pd.to_datetime(close.index).tz_localize(None)

    total_return = close.iloc[-1] / close.iloc[0] - 1
    n_days = len(close)
    years = n_days / 252
    cagr = (close.iloc[-1] / close.iloc[0]) ** (1 / years) - 1

    running_peak = close.cummax()
    drawdown = (running_peak - close) / running_peak
    max_dd = drawdown.max()

    daily_ret = close.pct_change().dropna()
    sharpe = (daily_ret.mean() / daily_ret.std()) * np.sqrt(252) if daily_ret.std() > 0 else 0.0

    print(f"QQQ 買進持有（{close.index[0].date()} ~ {close.index[-1].date()}，{n_days} 個交易日）")
    print(f"起始價: {close.iloc[0]:.2f}  結束價: {close.iloc[-1]:.2f}")
    print(f"總報酬: {total_return:.2%}")
    print(f"CAGR: {cagr:.2%}")
    print(f"MDD: {max_dd:.2%}")
    print(f"Sharpe: {sharpe:.2f}")
    print(f"Calmar: {(cagr / max_dd) if max_dd > 0 else 0.0:.2f}")
    print(
        "\n（對照：RSI/布林通道美股策略同一段期間 BB_zscore+vol/hold=5/top_n=10 "
        "總報酬 39.96%、CAGR 11.92%、MDD 15.39%、Sharpe 0.93、Calmar 0.77）"
    )


if __name__ == "__main__":
    main()
