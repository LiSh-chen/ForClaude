"""把不同回測引擎（tw_quant.backtest 的 BacktestResult / tw_quant.etf_combo
的加權組合模擬 / 單純買進持有 ETF）的結果，統一整理成兩份長格式表格：

  - 權益曲線（date, period, strategy, equity）
  - 成交紀錄（period, strategy, stock_id, entry_date, entry_price, exit_date,
    exit_price, shares, pnl, pnl_pct, still_open）

用途：使用者要求把「所有探討成果」畫成圖表比較不同策略，且明確要求要有
真實的逐日資產淨值曲線、持股/交易紀錄，不能用摘要指標湊——這裡把各腳本
already 算出來的 BacktestResult（本來就內建 equity_curve/trades，只是
沒有被存下來）跟 ETF 組合模擬的市值序列，收斂成同一種長格式，方便
scripts/export_strategy_curves_from_db.py 統一收集、寫成 parquet 給前端
圖表讀。

「still_open」代表回測結束時還沒平倉的部位，用最後一天的價格做記帳用
標記價（mark-to-market），不是真的賣出，跟 tw_quant.backtest_stats
的 build_combined_trades 邏輯一致（避免「贏家還沒平倉所以看不到」）。
"""

from __future__ import annotations

import pandas as pd

from tw_quant.backtest import BacktestResult

EQUITY_COLS = ["period", "strategy", "date", "equity"]
TRADE_COLS = [
    "period", "strategy", "stock_id", "industry", "entry_date", "entry_price",
    "exit_date", "exit_price", "shares", "pnl", "pnl_pct", "still_open",
]


def equity_rows_from_series(period: str, strategy: str, values: pd.Series) -> pd.DataFrame:
    """values：index 是日期、值是資產淨值（美元），可以是 BacktestResult
    .equity_curve['equity']，也可以是 etf_combo.simulate_weighted_portfolio
    回傳的市值序列，或單純買進持有 ETF 的收盤價序列。"""
    s = values.copy()
    s.index = pd.to_datetime(s.index)
    return pd.DataFrame(
        {"period": period, "strategy": strategy, "date": s.index, "equity": s.to_numpy()}
    )[EQUITY_COLS]


def equity_rows_from_result(period: str, strategy: str, result: BacktestResult) -> pd.DataFrame:
    return equity_rows_from_series(period, strategy, result.equity_curve["equity"])


def trade_rows_from_result(
    period: str, strategy: str, result: BacktestResult, prices: pd.DataFrame | None = None
) -> pd.DataFrame:
    """跟 tw_quant.backtest_stats.build_combined_trades 邏輯一致：已平倉的
    交易直接列出，回測結束時還持有中的部位額外用最後收盤價標記一筆
    still_open=True 的列，讓「持股狀態」在圖表上不會突然消失。"""
    rows = []
    for row in result.trades.itertuples(index=False):
        rows.append(
            {
                "period": period, "strategy": strategy, "stock_id": row.stock_id,
                "industry": row.industry, "entry_date": row.entry_date, "entry_price": row.entry_price,
                "exit_date": row.exit_date, "exit_price": row.exit_price, "shares": row.shares,
                "pnl": row.pnl, "pnl_pct": row.pnl_pct, "still_open": False,
            }
        )

    if result.open_positions and prices is not None and not prices.empty:
        last_date = prices["date"].max()
        last_prices = prices.sort_values("date").groupby("stock_id")["close"].last()
        for stock_id, pos in result.open_positions.items():
            if stock_id not in last_prices.index:
                continue
            mark_price = float(last_prices.loc[stock_id])
            pnl = pos.shares * mark_price - pos.cost_basis
            rows.append(
                {
                    "period": period, "strategy": strategy, "stock_id": stock_id,
                    "industry": pos.industry, "entry_date": pos.entry_date, "entry_price": pos.entry_price,
                    "exit_date": last_date, "exit_price": mark_price, "shares": pos.shares,
                    "pnl": pnl, "pnl_pct": pnl / pos.cost_basis if pos.cost_basis else 0.0, "still_open": True,
                }
            )

    if not rows:
        return pd.DataFrame(columns=TRADE_COLS)
    return pd.DataFrame(rows)[TRADE_COLS]


def trade_rows_from_fills(period: str, strategy: str, fills: list[dict], last_date, last_prices: dict[str, float]) -> pd.DataFrame:
    """把 tw_quant.etf_combo.simulate_weighted_portfolio 回傳的成交紀錄
    （逐筆買/賣，不是配對好的進出場）轉成跟 trade_rows_from_result 同樣的
    長格式：每個 ticker 用 FIFO 把 buy 配對到後續的 sell（或最後一天）當
    exit，方便跟 factor 策略的交易紀錄放進同一張表比較。"""
    rows = []
    open_lots: dict[str, list[dict]] = {}
    for fill in fills:
        ticker = fill["ticker"]
        open_lots.setdefault(ticker, [])
        if fill["action"] == "buy":
            open_lots[ticker].append({"date": fill["date"], "price": fill["price"], "shares": fill["shares"]})
        else:
            remaining = fill["shares"]
            sell_price = fill["price"]
            while remaining > 0 and open_lots[ticker]:
                lot = open_lots[ticker][0]
                matched = min(lot["shares"], remaining)
                pnl = matched * (sell_price - lot["price"])
                rows.append(
                    {
                        "period": period, "strategy": strategy, "stock_id": ticker, "industry": "ETF",
                        "entry_date": lot["date"], "entry_price": lot["price"], "exit_date": fill["date"],
                        "exit_price": sell_price, "shares": matched, "pnl": pnl,
                        "pnl_pct": pnl / (matched * lot["price"]) if lot["price"] else 0.0, "still_open": False,
                    }
                )
                lot["shares"] -= matched
                remaining -= matched
                if lot["shares"] <= 0:
                    open_lots[ticker].pop(0)

    for ticker, lots in open_lots.items():
        mark_price = last_prices.get(ticker)
        if mark_price is None:
            continue
        for lot in lots:
            if lot["shares"] <= 0:
                continue
            pnl = lot["shares"] * (mark_price - lot["price"])
            rows.append(
                {
                    "period": period, "strategy": strategy, "stock_id": ticker, "industry": "ETF",
                    "entry_date": lot["date"], "entry_price": lot["price"], "exit_date": last_date,
                    "exit_price": mark_price, "shares": lot["shares"], "pnl": pnl,
                    "pnl_pct": pnl / (lot["shares"] * lot["price"]) if lot["price"] else 0.0, "still_open": True,
                }
            )

    if not rows:
        return pd.DataFrame(columns=TRADE_COLS)
    return pd.DataFrame(rows)[TRADE_COLS]


def buyhold_trade_row(period: str, strategy: str, ticker: str, close: pd.DataFrame) -> pd.DataFrame:
    """單純買進持有一檔 ETF（QQQ/VOO/VTI/VT）：只有期初一筆進場，回測結束
    時永遠是還持有中，用最後收盤價標記。close 需有 date/close 兩欄，已經
    排序、已經限定期間。"""
    first = close.iloc[0]
    last = close.iloc[-1]
    shares = 1.0  # 買進持有比較看的是報酬率曲線本身，股數用 1 股正規化即可
    pnl = shares * (last["close"] - first["close"])
    row = {
        "period": period, "strategy": strategy, "stock_id": ticker, "industry": "ETF",
        "entry_date": first["date"], "entry_price": first["close"], "exit_date": last["date"],
        "exit_price": last["close"], "shares": shares, "pnl": pnl,
        "pnl_pct": pnl / first["close"] if first["close"] else 0.0, "still_open": True,
    }
    return pd.DataFrame([row])[TRADE_COLS]
