"""共用的交易層級統計工具（勝率/風報比/EV/獲利因子），給 scripts/explore_*_from_db.py
系列探索腳本共用，避免每支腳本各自複製一份、容易顧此失彼——factor 策略的
EV 記帳偏誤（已平倉交易樣本排除掉還沒被換掉的贏家）就是先在一份複製裡修好、
但這件事本身就說明了重複程式碼的風險，所以抽成共用模組。
"""

from __future__ import annotations

import pandas as pd

from tw_quant.backtest import BacktestResult, summarize_performance


def trade_stats(trades: pd.DataFrame) -> dict:
    if trades.empty:
        return {
            "avg_win_pct": 0.0, "avg_loss_pct": 0.0, "risk_reward_ratio": 0.0,
            "ev_pct": 0.0, "profit_factor": 0.0, "avg_holding_days": 0.0,
        }
    wins = trades[trades["pnl"] > 0]
    losses = trades[trades["pnl"] <= 0]
    win_rate = len(wins) / len(trades)
    avg_win_pct = wins["pnl_pct"].mean() if not wins.empty else 0.0
    avg_loss_pct = losses["pnl_pct"].mean() if not losses.empty else 0.0
    gross_profit = wins["pnl"].sum()
    gross_loss = -losses["pnl"].sum()
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")
    risk_reward_ratio = (avg_win_pct / abs(avg_loss_pct)) if avg_loss_pct < 0 else float("inf")
    ev_pct = win_rate * avg_win_pct + (1 - win_rate) * avg_loss_pct
    holding_days = (pd.to_datetime(trades["exit_date"]) - pd.to_datetime(trades["entry_date"])).dt.days
    return {
        "avg_win_pct": avg_win_pct, "avg_loss_pct": avg_loss_pct,
        "risk_reward_ratio": risk_reward_ratio, "ev_pct": ev_pct,
        "profit_factor": profit_factor, "avg_holding_days": holding_days.mean(),
    }


def build_combined_trades(result: BacktestResult, prices: pd.DataFrame) -> pd.DataFrame:
    """把回測結束時還持有中的部位，用最後一天收盤價算「虛擬平倉」損益，
    併入已實現交易一起算統計指標，避免「贏家還沒平倉所以不算」的偏誤
    （total_return 不受影響，那是用完整權益曲線算的）。
    """
    trades = result.trades.copy()
    if not result.open_positions or prices.empty:
        return trades

    last_date = prices["date"].max()
    last_prices = prices.sort_values("date").groupby("stock_id")["close"].last()

    pseudo_rows = []
    for stock_id, pos in result.open_positions.items():
        if stock_id not in last_prices.index:
            continue
        mark_price = last_prices.loc[stock_id]
        pnl = pos.shares * mark_price - pos.cost_basis
        pseudo_rows.append(
            {
                "stock_id": stock_id, "industry": pos.industry, "strategy": pos.strategy,
                "shares": pos.shares, "entry_date": pos.entry_date, "entry_price": pos.entry_price,
                "exit_date": last_date, "exit_price": mark_price, "pnl": pnl,
                "pnl_pct": pnl / pos.cost_basis if pos.cost_basis else 0.0,
            }
        )
    if pseudo_rows:
        trades = pd.concat([trades, pd.DataFrame(pseudo_rows)], ignore_index=True)
    return trades


def metrics_from_result(result: BacktestResult, initial_capital: float, prices: pd.DataFrame | None = None) -> dict:
    """prices 有給的話，會把還未平倉的部位用最後收盤價一併算進 EV/勝率/風報比/
    獲利因子/n_trades/annual_trades；total_return/cagr/max_dd/sharpe/calmar 本來
    就是用完整權益曲線算的，不受影響。
    """
    m = summarize_performance(result, initial_capital)
    m["calmar"] = (m["cagr"] / m["max_dd"]) if m["max_dd"] > 0 else 0.0

    stats_source = build_combined_trades(result, prices) if prices is not None else result.trades
    m.update(trade_stats(stats_source))
    if prices is not None:
        n_days = len(result.equity_curve)
        years = max(n_days / 252, 1e-9)
        m["n_trades"] = len(stats_source)
        m["annual_trades"] = len(stats_source) / years
        m["n_open_positions"] = len(result.open_positions)
    return m
