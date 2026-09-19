"""測試 scripts/test_yfinance_full_backfill_scan.py 的 fetch_one 摘要/例外
處理邏輯本身（不連網，用假 provider 替換掉真的 yfinance 呼叫）。
"""

import sys
from pathlib import Path
from unittest import mock

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from test_yfinance_full_backfill_scan import fetch_one  # noqa: E402


def test_fetch_one_reports_rows_and_date_range_on_success():
    provider = mock.Mock()
    provider.fetch_price.return_value = pd.DataFrame(
        {"date": pd.to_datetime(["2020-01-02", "2020-01-03"]), "close": [1.0, 2.0]}
    )
    result = fetch_one(provider, "AAPL", "2020-01-01", "2020-01-05")
    assert result["stock_id"] == "AAPL"
    assert result["rows"] == 2
    assert result["error"] is None
    assert result["first"] == pd.Timestamp("2020-01-02").date()
    assert result["last"] == pd.Timestamp("2020-01-03").date()


def test_fetch_one_reports_zero_rows_on_empty_dataframe():
    provider = mock.Mock()
    provider.fetch_price.return_value = pd.DataFrame(columns=["date", "close"])
    result = fetch_one(provider, "GHOST", "2020-01-01", "2020-01-05")
    assert result == {"stock_id": "GHOST", "rows": 0, "error": None}


def test_fetch_one_catches_exception_and_reports_it_instead_of_raising():
    provider = mock.Mock()
    provider.fetch_price.side_effect = RuntimeError("boom")
    result = fetch_one(provider, "BROKEN", "2020-01-01", "2020-01-05")
    assert result["stock_id"] == "BROKEN"
    assert result["rows"] == 0
    assert "RuntimeError" in result["error"]
    assert "boom" in result["error"]
