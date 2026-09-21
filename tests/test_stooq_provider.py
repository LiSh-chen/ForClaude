"""測試 tw_quant/stooq_provider.py 的 CSV 解析與代號轉換邏輯（不連網）。"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tw_quant.stooq_provider import normalize_stooq_ticker, parse_stooq_csv


def test_normalize_stooq_ticker_lowercases_and_appends_us_suffix():
    assert normalize_stooq_ticker("AAPL") == "aapl.us"
    assert normalize_stooq_ticker("LEHMQ") == "lehmq.us"
    assert normalize_stooq_ticker("BRK-B") == "brk-b.us"


def test_parse_stooq_csv_parses_valid_data():
    csv_text = "Date,Open,High,Low,Close,Volume\n2008-09-12,16.20,16.50,15.80,16.00,50000000\n2008-09-15,3.65,3.70,3.10,3.65,120000000\n"

    df = parse_stooq_csv(csv_text, "LEHMQ", industry="Financials")

    assert len(df) == 2
    assert list(df.columns) == ["date", "stock_id", "industry", "open", "high", "low", "close", "volume", "turnover_value"]
    assert df["date"].iloc[0] == pd.Timestamp("2008-09-12")
    assert df["stock_id"].unique().tolist() == ["LEHMQ"]
    assert df["turnover_value"].iloc[1] == 3.65 * 120000000


def test_parse_stooq_csv_returns_empty_on_no_data_response():
    df = parse_stooq_csv("No data", "WAMUQ")
    assert df.empty
    assert list(df.columns) == ["date", "stock_id", "industry", "open", "high", "low", "close", "volume", "turnover_value"]


def test_parse_stooq_csv_returns_empty_on_blank_response():
    df = parse_stooq_csv("", "BSC")
    assert df.empty
