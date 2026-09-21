"""測試 scripts/retry_missing_sp500_stocks_from_db.py 的 retry_one 寫入
邏輯與錯誤分類本身（不連網、不碰真的資料庫，用假 store/provider 替換掉）。
"""

import sys
from pathlib import Path
from unittest import mock

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from retry_missing_sp500_stocks_from_db import _error_category, retry_one  # noqa: E402


def _row(stock_id="XTO", start="2006-09-25", end="2010-06-28", clipped_start="2006-09-25", clipped_end="2010-06-28"):
    return mock.Mock(
        stock_id=stock_id,
        start_date=pd.Timestamp(start),
        end_date=pd.Timestamp(end),
        clipped_start=pd.Timestamp(clipped_start),
        clipped_end=pd.Timestamp(clipped_end),
    )


def test_retry_one_writes_only_rows_within_clipped_window():
    store = mock.Mock()
    provider = mock.Mock()
    # period="max" 抓到比 clipped 窗口更寬的歷史（含窗口外的資料）
    provider.fetch_price_full_history.return_value = pd.DataFrame(
        {
            "date": pd.to_datetime(["2005-01-01", "2007-01-01", "2010-01-01", "2015-01-01"]),
            "stock_id": ["XTO"] * 4,
            "close": [1.0, 2.0, 3.0, 4.0],
        }
    )

    result = retry_one(store, provider, _row())

    assert result == {"stock_id": "XTO", "written": True, "rows": 2, "error": None}
    written_df = store.upsert_us_prices.call_args[0][0]
    assert list(written_df["date"]) == [pd.Timestamp("2007-01-01"), pd.Timestamp("2010-01-01")]
    store.upsert_us_index_membership.assert_called_once()


def test_retry_one_reports_data_outside_window_when_nothing_overlaps():
    store = mock.Mock()
    provider = mock.Mock()
    provider.fetch_price_full_history.return_value = pd.DataFrame(
        {"date": pd.to_datetime(["2020-01-01"]), "stock_id": ["XTO"], "close": [1.0]}
    )

    result = retry_one(store, provider, _row())

    assert result["written"] is False
    assert "落在查詢窗口外" in result["error"]
    store.upsert_us_prices.assert_not_called()


def test_retry_one_skips_write_when_empty():
    store = mock.Mock()
    provider = mock.Mock()
    provider.fetch_price_full_history.return_value = pd.DataFrame(columns=["date", "stock_id", "close"])

    result = retry_one(store, provider, _row(stock_id="LEHMQ"))

    assert result == {"stock_id": "LEHMQ", "written": False, "rows": 0, "error": None}
    store.upsert_us_prices.assert_not_called()


def test_retry_one_catches_exception():
    store = mock.Mock()
    provider = mock.Mock()
    provider.fetch_price_full_history.side_effect = RuntimeError("possibly delisted; no timezone found")

    result = retry_one(store, provider, _row(stock_id="WAMUQ"))

    assert result["written"] is False
    assert "RuntimeError" in result["error"]


def test_error_category_classifies_known_patterns():
    assert _error_category(None) == "查無資料（無例外，空表）"
    assert _error_category("抓到資料但都落在查詢窗口外（2006-09-25~2010-06-28）") == "抓到資料但不在需要的日期範圍內"
    assert _error_category("RuntimeError: possibly delisted; no timezone found") == "yfinance 完全不認得這個代號"
    assert _error_category("YFPricesMissingError: possibly no price data found (1d 2006-09-25 -> 2010-06-28)") == "yfinance 認得但該窗口無資料"
    assert _error_category("Data doesn't exist for startDate = 1159156800, endDate = 1231218000") == "yfinance 認得但該窗口無資料"
    assert _error_category("KeyError: something unexpected") == "其他例外"
