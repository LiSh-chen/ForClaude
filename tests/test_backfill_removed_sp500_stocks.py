"""測試 scripts/backfill_removed_sp500_stocks.py 的 backfill_one 寫入邏輯
本身（不連網、不碰真的資料庫，用假 store/provider 替換掉）。
"""

import sys
from pathlib import Path
from unittest import mock

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from backfill_removed_sp500_stocks import backfill_one  # noqa: E402


def _row(stock_id="CELG", start="2005-01-01", end="2019-11-21", clipped_start="2018-09-20", clipped_end="2019-11-21"):
    return mock.Mock(
        stock_id=stock_id,
        start_date=pd.Timestamp(start),
        end_date=pd.Timestamp(end),
        clipped_start=pd.Timestamp(clipped_start),
        clipped_end=pd.Timestamp(clipped_end),
    )


def test_backfill_one_writes_prices_and_membership_on_success():
    store = mock.Mock()
    provider = mock.Mock()
    provider.fetch_price.return_value = pd.DataFrame(
        {"date": pd.to_datetime(["2018-09-20", "2018-09-21"]), "stock_id": ["CELG", "CELG"], "close": [1.0, 2.0]}
    )

    result = backfill_one(store, provider, _row())

    assert result == {"stock_id": "CELG", "written": True, "rows": 2, "error": None}
    store.upsert_us_prices.assert_called_once()
    store.upsert_us_index_membership.assert_called_once()
    membership_arg = store.upsert_us_index_membership.call_args[0][0]
    assert membership_arg["stock_id"].iloc[0] == "CELG"
    assert membership_arg["start_date"].iloc[0] == pd.Timestamp("2005-01-01")
    assert membership_arg["end_date"].iloc[0] == pd.Timestamp("2019-11-21")


def test_backfill_one_skips_write_when_no_data():
    store = mock.Mock()
    provider = mock.Mock()
    provider.fetch_price.return_value = pd.DataFrame(columns=["date", "stock_id", "close"])

    result = backfill_one(store, provider, _row(stock_id="SIVB"))

    assert result == {"stock_id": "SIVB", "written": False, "rows": 0, "error": None}
    store.upsert_us_prices.assert_not_called()
    store.upsert_us_index_membership.assert_not_called()


def test_backfill_one_catches_exception_and_skips_write():
    store = mock.Mock()
    provider = mock.Mock()
    provider.fetch_price.side_effect = RuntimeError("possibly delisted; no timezone found")

    result = backfill_one(store, provider, _row(stock_id="TWTR"))

    assert result["stock_id"] == "TWTR"
    assert result["written"] is False
    assert "RuntimeError" in result["error"]
    store.upsert_us_prices.assert_not_called()
    store.upsert_us_index_membership.assert_not_called()
