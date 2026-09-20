"""測試美股資料轉換邏輯本身（tw_quant/us_data_provider.py），不連網——
只測「維基百科表格轉欄位」「yfinance 歷史資料轉長格式」「代號格式轉換」
這幾個純函式對不對。
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tw_quant.us_data_provider import (
    US_EARNINGS_COLUMNS,
    US_PRICE_COLUMNS,
    normalize_yfinance_ticker,
    parse_sp500_wikipedia_table,
    parse_yfinance_earnings_dates,
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
            "Date added": ["1980-12-12", "2010-02-16", "1980-12-12"],
        }
    )
    result = parse_sp500_wikipedia_table(raw)

    assert list(result.columns) == ["stock_id", "name", "industry", "date_added"]
    assert len(result) == 2  # 去重後只剩 2 檔
    assert set(result["stock_id"]) == {"AAPL", "BRK-B"}  # BRK.B 該被轉成 BRK-B
    added = result.set_index("stock_id")["date_added"]
    assert added["AAPL"] == pd.Timestamp("1980-12-12")
    assert added["BRK-B"] == pd.Timestamp("2010-02-16")


def test_parse_sp500_wikipedia_table_handles_missing_date_added_column():
    """維基百科頁面格式偶爾會變動，"Date added" 欄位萬一被拿掉，不該直接
    炸掉——存活者偏差修正是加分項，不是回測能不能跑的必要條件。
    """
    raw = pd.DataFrame(
        {"Symbol": ["AAPL"], "Security": ["Apple Inc."], "GICS Sector": ["Information Technology"]}
    )
    result = parse_sp500_wikipedia_table(raw)

    assert list(result.columns) == ["stock_id", "name", "industry", "date_added"]
    assert result["date_added"].isna().all()


def test_parse_sp500_wikipedia_table_coerces_unparseable_dates_to_nat():
    raw = pd.DataFrame(
        {
            "Symbol": ["AAPL"],
            "Security": ["Apple Inc."],
            "GICS Sector": ["Information Technology"],
            "Date added": ["not a date"],
        }
    )
    result = parse_sp500_wikipedia_table(raw)

    assert result["date_added"].isna().all()


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


def test_parse_yfinance_earnings_dates_converts_to_long_format():
    idx = pd.DatetimeIndex(["2024-01-25", "2024-04-25"], tz="America/New_York")
    raw = pd.DataFrame(
        {"EPS Estimate": [1.0, 1.1], "Reported EPS": [1.05, 1.2], "Surprise(%)": [5.0, 9.1]},
        index=idx,
    )
    result = parse_yfinance_earnings_dates(raw, "AAPL")

    assert list(result.columns) == US_EARNINGS_COLUMNS
    assert len(result) == 2
    assert (result["stock_id"] == "AAPL").all()
    assert result["date"].dt.tz is None  # 時區該被去掉，跟其他資料的 naive datetime 一致
    assert result["eps_estimate"].tolist() == [1.0, 1.1]
    assert result["eps_actual"].tolist() == [1.05, 1.2]
    assert result["surprise_pct"].tolist() == [5.0, 9.1]


def test_parse_yfinance_earnings_dates_matches_columns_case_and_space_insensitively():
    """2026-09-19 探路時發現不能假設欄位名稱完全固定，模糊比對要真的有效。"""
    idx = pd.DatetimeIndex(["2024-01-25"])
    raw = pd.DataFrame({"epsestimate": [1.0], "REPORTED eps": [1.1]}, index=idx)
    result = parse_yfinance_earnings_dates(raw, "AAPL")

    assert result["eps_estimate"].iloc[0] == 1.0
    assert result["eps_actual"].iloc[0] == 1.1
    assert result["surprise_pct"].isna().all()  # 沒有 surprise 欄位時該是 NaN，不是報錯


def test_parse_yfinance_earnings_dates_handles_empty_result():
    result = parse_yfinance_earnings_dates(pd.DataFrame(), "AAPL")
    assert result.empty
    assert list(result.columns) == US_EARNINGS_COLUMNS


def test_parse_yfinance_earnings_dates_handles_none():
    result = parse_yfinance_earnings_dates(None, "AAPL")
    assert result.empty
    assert list(result.columns) == US_EARNINGS_COLUMNS


def test_parse_yfinance_earnings_dates_dedupes_same_day_keeping_more_complete_row():
    """2026-09-20 全量回填時實測發現：約 13% 的股票 yfinance 對同一天會
    回傳重複列，寫進 Postgres 的 ON CONFLICT DO UPDATE 在同一批次裡遇到
    重複主鍵會直接報錯，整批股票的資料都寫不進去。這裡驗證去重邏輯：
    同一天有多筆時，保留欄位比較完整（非空值較多）的那一筆。
    """
    idx = pd.DatetimeIndex(["2024-01-25", "2024-01-25"])  # 同一天重複兩筆
    raw = pd.DataFrame(
        {"EPS Estimate": [1.0, 1.0], "Reported EPS": [None, 1.1], "Surprise(%)": [None, 10.0]},
        index=idx,
    )
    result = parse_yfinance_earnings_dates(raw, "AAPL")

    assert len(result) == 1  # 去重成只剩一筆
    assert result["eps_actual"].iloc[0] == 1.1  # 保留比較完整（有實際EPS）的那筆
    assert result["surprise_pct"].iloc[0] == 10.0
