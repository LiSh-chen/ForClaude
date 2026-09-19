"""測試 scripts/test_yfinance_delisted_tickers.py 的摘要格式化邏輯本身
（不連網）：空資料跟有資料兩種情況統計得對不對。
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from test_yfinance_delisted_tickers import summarize_fetch  # noqa: E402


def test_summarize_fetch_empty_dataframe():
    summary = summarize_fetch("GHOST", pd.DataFrame(columns=["date", "close"]))
    assert summary == {"ticker": "GHOST", "rows": 0, "first": None, "last": None}


def test_summarize_fetch_reports_row_count_and_date_range():
    df = pd.DataFrame({"date": pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06"]), "close": [1.0, 2.0, 3.0]})
    summary = summarize_fetch("AAPL", df)
    assert summary["ticker"] == "AAPL"
    assert summary["rows"] == 3
    assert summary["first"] == pd.Timestamp("2020-01-02").date()
    assert summary["last"] == pd.Timestamp("2020-01-06").date()
