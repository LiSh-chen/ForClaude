"""測試 tw_quant/curve_export.py 的長格式轉換邏輯（合成資料，不連資料庫）。"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tw_quant.backtest import BacktestResult, Position, TradeRecord
from tw_quant.curve_export import (
    EQUITY_COLS,
    TRADE_COLS,
    buyhold_trade_row,
    equity_rows_from_result,
    equity_rows_from_series,
    trade_rows_from_fills,
    trade_rows_from_result,
)


def test_equity_rows_from_series_has_expected_columns_and_values():
    idx = pd.date_range("2020-01-01", periods=3, freq="B")
    values = pd.Series([100.0, 110.0, 105.0], index=idx)
    rows = equity_rows_from_series("樣本外", "策略A", values)

    assert list(rows.columns) == EQUITY_COLS
    assert len(rows) == 3
    assert (rows["strategy"] == "策略A").all()
    assert (rows["period"] == "樣本外").all()
    assert rows["equity"].tolist() == [100.0, 110.0, 105.0]


def _make_result(with_open_position: bool) -> BacktestResult:
    idx = pd.date_range("2020-01-01", periods=5, freq="B")
    equity = pd.DataFrame({"equity": [100.0, 101.0, 102.0, 103.0, 104.0]}, index=idx)
    trade = TradeRecord(
        stock_id="AAA", industry="Tech", strategy="FACTOR", shares=10,
        entry_date=idx[0], entry_price=10.0, exit_date=idx[2], exit_price=11.0,
        pnl=10.0, pnl_pct=0.10,
    )
    trades_df = pd.DataFrame([trade.__dict__])
    open_positions = {}
    if with_open_position:
        open_positions["BBB"] = Position(
            stock_id="BBB", industry="Tech", strategy="FACTOR", shares=5,
            entry_price=20.0, entry_date=idx[3], stop_price=0.0, cost_basis=100.0,
        )
    return BacktestResult(equity_curve=equity, trades=trades_df, open_positions=open_positions)


def test_equity_rows_from_result_reads_equity_curve():
    result = _make_result(with_open_position=False)
    rows = equity_rows_from_result("樣本內", "top_n=2", result)
    assert len(rows) == 5
    assert rows["equity"].iloc[-1] == 104.0


def test_trade_rows_from_result_includes_closed_trade():
    result = _make_result(with_open_position=False)
    rows = trade_rows_from_result("樣本內", "top_n=2", result)
    assert list(rows.columns) == TRADE_COLS
    assert len(rows) == 1
    assert rows.iloc[0]["stock_id"] == "AAA"
    assert rows.iloc[0]["still_open"] == False  # noqa: E712


def test_trade_rows_from_result_marks_open_position_with_last_price():
    result = _make_result(with_open_position=True)
    prices = pd.DataFrame(
        {"date": pd.date_range("2020-01-01", periods=5, freq="B"), "stock_id": "BBB", "close": [20.0, 21.0, 22.0, 23.0, 25.0]}
    )
    rows = trade_rows_from_result("樣本內", "top_n=2", result, prices=prices)

    assert len(rows) == 2
    open_row = rows[rows["still_open"]].iloc[0]
    assert open_row["stock_id"] == "BBB"
    assert open_row["exit_price"] == 25.0  # 最後一天收盤價當標記價
    assert open_row["pnl"] == pytest.approx(5 * 25.0 - 100.0)


def test_trade_rows_from_result_without_prices_skips_open_positions():
    result = _make_result(with_open_position=True)
    rows = trade_rows_from_result("樣本內", "top_n=2", result, prices=None)
    # 沒給 prices 就沒辦法標記未平倉部位的市值，只回傳已平倉交易
    assert len(rows) == 1


def test_trade_rows_from_fills_pairs_buy_then_sell_fifo():
    dates = pd.date_range("2020-01-01", periods=3, freq="B")
    fills = [
        {"date": dates[0], "ticker": "QQQ", "action": "buy", "price": 100.0, "shares": 10},
        {"date": dates[1], "ticker": "QQQ", "action": "sell", "price": 110.0, "shares": 4},
    ]
    rows = trade_rows_from_fills("樣本外", "QQQ90/VOO10", fills, last_date=dates[2], last_prices={"QQQ": 120.0})

    assert len(rows) == 2  # 一筆已平倉（賣掉的4股）+ 一筆未平倉（剩下的6股標記到最後價）
    closed = rows[~rows["still_open"]].iloc[0]
    assert closed["shares"] == 4
    assert closed["exit_price"] == 110.0
    assert closed["pnl"] == pytest.approx(4 * (110.0 - 100.0))

    still_open = rows[rows["still_open"]].iloc[0]
    assert still_open["shares"] == 6
    assert still_open["exit_price"] == 120.0
    assert still_open["pnl"] == pytest.approx(6 * (120.0 - 100.0))


def test_trade_rows_from_fills_empty_returns_empty_frame_with_columns():
    rows = trade_rows_from_fills("樣本外", "x", [], last_date=pd.Timestamp("2020-01-01"), last_prices={})
    assert list(rows.columns) == TRADE_COLS
    assert rows.empty


def test_buyhold_trade_row_marks_single_open_position():
    close = pd.DataFrame(
        {"date": pd.date_range("2020-01-01", periods=3, freq="B"), "close": [100.0, 105.0, 110.0]}
    )
    row = buyhold_trade_row("樣本外", "QQQ", "QQQ", close)
    assert len(row) == 1
    assert row.iloc[0]["still_open"] == True  # noqa: E712
    assert row.iloc[0]["entry_price"] == 100.0
    assert row.iloc[0]["exit_price"] == 110.0
    assert row.iloc[0]["pnl_pct"] == pytest.approx(0.10)
