"""測試 tw_quant/us_universe.py 的存活者偏差修正邏輯（新進戶跟被剔除
兩半都要覆蓋，membership 現在是 (stock_id, start_date, end_date) 區間表）。
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tw_quant.us_universe import filter_prices_by_index_membership


def test_filters_out_rows_before_start_date():
    prices = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-01", "2020-06-01", "2021-01-01"]),
            "stock_id": ["A", "A", "A"],
            "close": [1.0, 2.0, 3.0],
        }
    )
    membership = pd.DataFrame({"stock_id": ["A"], "start_date": pd.to_datetime(["2020-05-01"]), "end_date": [pd.NaT]})

    result = filter_prices_by_index_membership(prices, membership)

    assert result["date"].tolist() == pd.to_datetime(["2020-06-01", "2021-01-01"]).tolist()


def test_start_date_row_itself_is_kept():
    prices = pd.DataFrame({"date": pd.to_datetime(["2020-05-01"]), "stock_id": ["A"], "close": [1.0]})
    membership = pd.DataFrame({"stock_id": ["A"], "start_date": pd.to_datetime(["2020-05-01"]), "end_date": [pd.NaT]})

    result = filter_prices_by_index_membership(prices, membership)

    assert len(result) == 1


def test_stock_without_membership_record_is_not_filtered():
    """membership 表裡完全沒有這檔股票（例如尚未跑過最新的 ingest 或
    backfill），寧可保守不誤殺，不是當成「永遠不合格」全部丟掉。
    """
    prices = pd.DataFrame({"date": pd.to_datetime(["2020-01-01"]), "stock_id": ["B"], "close": [1.0]})
    membership = pd.DataFrame({"stock_id": ["A"], "start_date": pd.to_datetime(["2020-05-01"]), "end_date": [pd.NaT]})

    result = filter_prices_by_index_membership(prices, membership)

    assert len(result) == 1


def test_missing_start_date_value_is_not_filtered():
    prices = pd.DataFrame({"date": pd.to_datetime(["2020-01-01"]), "stock_id": ["A"], "close": [1.0]})
    membership = pd.DataFrame({"stock_id": ["A"], "start_date": [pd.NaT], "end_date": [pd.NaT]})

    result = filter_prices_by_index_membership(prices, membership)

    assert len(result) == 1


def test_empty_membership_returns_prices_unchanged():
    prices = pd.DataFrame({"date": pd.to_datetime(["2020-01-01"]), "stock_id": ["A"], "close": [1.0]})
    membership = pd.DataFrame(columns=["stock_id", "start_date", "end_date"])

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
    membership = pd.DataFrame({"stock_id": ["A"], "start_date": pd.to_datetime(["2020-06-01"]), "end_date": [pd.NaT]})

    result = filter_prices_by_index_membership(prices, membership)

    assert result["stock_id"].tolist() == ["B"]


def test_filters_out_rows_on_or_after_end_date():
    """被剔除股票：end_date 當天（含）之後不再合格——區間語意是
    [start_date, end_date)，跟 tw_quant.sp500_history.build_membership_intervals
    的區間定義一致。
    """
    prices = pd.DataFrame(
        {
            "date": pd.to_datetime(["2019-12-20", "2019-12-21", "2019-12-22"]),
            "stock_id": ["CELG", "CELG", "CELG"],
            "close": [1.0, 2.0, 3.0],
        }
    )
    membership = pd.DataFrame(
        {"stock_id": ["CELG"], "start_date": pd.to_datetime(["2005-01-01"]), "end_date": pd.to_datetime(["2019-12-21"])}
    )

    result = filter_prices_by_index_membership(prices, membership)

    assert result["date"].tolist() == [pd.Timestamp("2019-12-20")]


def test_open_ended_end_date_keeps_rows_through_latest_data():
    prices = pd.DataFrame(
        {"date": pd.to_datetime(["2020-01-01", "2026-01-01"]), "stock_id": ["A", "A"], "close": [1.0, 2.0]}
    )
    membership = pd.DataFrame({"stock_id": ["A"], "start_date": pd.to_datetime(["2018-01-01"]), "end_date": [pd.NaT]})

    result = filter_prices_by_index_membership(prices, membership)

    assert len(result) == 2


def test_multiple_intervals_keeps_rows_in_either_interval():
    """中途被剔除又重新加入的股票：兩段區間之間的日子要被丟掉，
    落在任一段區間內的日子都該留著。
    """
    prices = pd.DataFrame(
        {
            "date": pd.to_datetime(["2019-06-01", "2020-06-01", "2022-06-01"]),
            "stock_id": ["FLIP", "FLIP", "FLIP"],
            "close": [1.0, 2.0, 3.0],
        }
    )
    membership = pd.DataFrame(
        {
            "stock_id": ["FLIP", "FLIP"],
            "start_date": pd.to_datetime(["2018-01-01", "2021-01-01"]),
            "end_date": pd.to_datetime(["2020-01-01", pd.NaT]),
        }
    )

    result = filter_prices_by_index_membership(prices, membership)

    assert result["date"].tolist() == pd.to_datetime(["2019-06-01", "2022-06-01"]).tolist()
