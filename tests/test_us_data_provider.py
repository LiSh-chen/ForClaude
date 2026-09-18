"""測試美股資料轉換邏輯本身（tw_quant/us_data_provider.py），不連網——
只測「維基百科表格轉欄位」「yfinance 歷史資料轉長格式」「代號格式轉換」
這幾個純函式對不對。
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tw_quant.us_data_provider import (
    US_PRICE_COLUMNS,
    normalize_yfinance_ticker,
    parse_sp500_wikipedia_table,
    parse_yfinance_history,
)


def test_normalize_yfinance_ticker_converts_dot_to_dash():
    assert normalize_yfinance_ticker("BRK.B") == "BRK-B"
    assert normalize_yfinance_ticker("AAPL") == "AAPL"


def test_parse_sp500_wikipedia_table_renames_and_normalizes():
    raw = pd.DataFrame(
        {
            "Symbol": ["AAPL", "BRK.B", "AAPL"],  # 重複的 AAPL 該被去重
            "Security": ["Apple Inc.", "Berkshire Hathaway", "Apple Inc."],
            "GICS Sector": ["Information Technology", "Financials", "Information Technology"],
            "GICS Sub-Industry": ["x", "y", "x"],  # 不需要的欄位該被丟掉
        }
    )
    result = parse_sp500_wikipedia_table(raw)

    assert list(result.columns) == ["stock_id", "name", "industry"]
    assert len(result) == 2  # 去重後只剩 2 檔
    assert set(result["stock_id"]) == {"AAPL", "BRK-B"}  # BRK.B 該被轉成 BRK-B


def test_parse_yfinance_history_converts_to_long_format():
    idx = pd.date_range("2024-01-01", periods=2, tz="America/New_York")  # yfinance 通常回傳帶時區的索引
    idx.name = "Date"
    raw = pd.DataFrame(
        {"Open": [100.0, 101.0], "High": [101.0, 102.0], "Low": [99.0, 100.0], "Close": [100.5, 101.5], "Volume": [1000, 2000]},
        index=idx,
    )
    result = parse_yfinance_history(raw, "AAPL", "Information Technology")

    assert list(result.columns) == US_PRICE_COLUMNS
    assert len(result) == 2
    assert (result["stock_id"] == "AAPL").all()
    assert (result["industry"] == "Information Technology").all()
    assert result["turnover_value"].iloc[0] == 100.5 * 1000
    assert result["date"].dt.tz is None  # 時區該被去掉，跟其他資料的 naive datetime 一致


def test_parse_yfinance_history_handles_empty_result():
    result = parse_yfinance_history(pd.DataFrame(), "AAPL", "Information Technology")
    assert result.empty
    assert list(result.columns) == US_PRICE_COLUMNS
