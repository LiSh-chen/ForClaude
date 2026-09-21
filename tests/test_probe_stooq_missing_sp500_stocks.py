"""測試 scripts/probe_stooq_missing_sp500_stocks.py 的 probe_one 判讀邏輯
（不連網，傳假的 fetch_fn 替換掉）。
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from probe_stooq_missing_sp500_stocks import probe_one  # noqa: E402


def _df(dates):
    return pd.DataFrame({"date": pd.to_datetime(dates), "stock_id": ["LEHMQ"] * len(dates), "close": [1.0] * len(dates)})


def test_probe_one_reports_full_coverage_when_data_overlaps_window():
    fetch_fn = lambda stock_id: _df(["2007-01-01", "2008-09-15", "2009-01-01"])  # noqa: E731

    result = probe_one("LEHMQ", pd.Timestamp("2008-01-01"), pd.Timestamp("2008-12-31"), fetch_fn=fetch_fn)

    assert result["covers_window"] is True
    assert result["rows"] == 3
    assert result["rows_in_window"] == 1


def test_probe_one_reports_no_coverage_when_data_outside_window():
    fetch_fn = lambda stock_id: _df(["2015-01-01", "2016-01-01"])  # noqa: E731

    result = probe_one("LEHMQ", pd.Timestamp("2008-01-01"), pd.Timestamp("2008-12-31"), fetch_fn=fetch_fn)

    assert result["covers_window"] is False
    assert result["rows"] == 2


def test_probe_one_reports_empty_when_no_data():
    fetch_fn = lambda stock_id: pd.DataFrame(columns=["date", "stock_id", "close"])  # noqa: E731

    result = probe_one("WAMUQ", pd.Timestamp("2008-01-01"), pd.Timestamp("2008-12-31"), fetch_fn=fetch_fn)

    assert result == {"stock_id": "WAMUQ", "rows": 0, "covers_window": False, "error": None}


def test_probe_one_catches_exception():
    def fetch_fn(stock_id):
        raise RuntimeError("connection refused")

    result = probe_one("BSC", pd.Timestamp("2008-01-01"), pd.Timestamp("2008-12-31"), fetch_fn=fetch_fn)

    assert result["rows"] == 0
    assert result["covers_window"] is False
    assert "RuntimeError" in result["error"]
