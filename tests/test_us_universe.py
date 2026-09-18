"""測試 tw_quant/us_universe.py 的存活者偏差部分修正邏輯。"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tw_quant.us_universe import filter_prices_by_index_membership


def test_filters_out_rows_before_date_added():
    prices = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-01", "2020-06-01", "2021-01-01"]),
            "stock_id": ["A", "A", "A"],
            "close": [1.0, 2.0, 3.0],
        }
    )
    membership = pd.DataFrame({"stock_id": ["A"], "date_added": pd.to_datetime(["2020-05-01"])})

    result = filter_prices_by_index_membership(prices, membership)

    assert result["date"].tolist() == pd.to_datetime(["2020-06-01", "2021-01-01"]).tolist()


def test_date_added_row_itself_is_kept():
    prices = pd.DataFrame({"date": pd.to_datetime(["2020-05-01"]), "stock_id": ["A"], "close": [1.0]})
    membership = pd.DataFrame({"stock_id": ["A"], "date_added": pd.to_datetime(["2020-05-01"])})

    result = filter_prices_by_index_membership(prices, membership)

    assert len(result) == 1


def test_stock_without_membership_record_is_not_filtered():
    """membership 表裡完全沒有這檔股票（例如尚未跑過最新的 ingest），
    寧可保守不誤殺，不是當成「永遠不合格」全部丟掉。
    """
    prices = pd.DataFrame({"date": pd.to_datetime(["2020-01-01"]), "stock_id": ["B"], "close": [1.0]})
    membership = pd.DataFrame({"stock_id": ["A"], "date_added": pd.to_datetime(["2020-05-01"])})

    result = filter_prices_by_index_membership(prices, membership)

    assert len(result) == 1


def test_missing_date_added_value_is_not_filtered():
    prices = pd.DataFrame({"date": pd.to_datetime(["2020-01-01"]), "stock_id": ["A"], "close": [1.0]})
    membership = pd.DataFrame({"stock_id": ["A"], "date_added": [pd.NaT]})

    result = filter_prices_by_index_membership(prices, membership)

    assert len(result) == 1


def test_empty_membership_returns_prices_unchanged():
    prices = pd.DataFrame({"date": pd.to_datetime(["2020-01-01"]), "stock_id": ["A"], "close": [1.0]})
    membership = pd.DataFrame(columns=["stock_id", "date_added"])

    result = filter_prices_by_index_membership(prices, membership)

    pd.testing.assert_frame_equal(result, prices)


def test_only_filters_the_matching_stock_not_the_whole_universe():
    prices = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-01", "2020-01-01"]),
            "stock_id": ["A", "B"],
            "close": [1.0, 2.0],
        }
    )
    membership = pd.DataFrame({"stock_id": ["A"], "date_added": pd.to_datetime(["2020-06-01"])})

    result = filter_prices_by_index_membership(prices, membership)

    assert result["stock_id"].tolist() == ["B"]
