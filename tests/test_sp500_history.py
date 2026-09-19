"""測試 tw_quant/sp500_history.py 的快照解析跟區間重建邏輯（不連網）：
代號正規化對不對、稀疏快照能不能正確重建成連續區間、中途被剔除又重新
加入的股票會不會被拆成兩段區間、目前仍在指數裡的股票 end_date 是不是
NaT。
"""

import pandas as pd

from tw_quant.sp500_history import build_membership_intervals, parse_snapshot_table


def test_parse_snapshot_table_normalizes_tickers_and_sorts_by_date():
    csv_text = "date,tickers\n" '2020-01-02,"BRK.B,AAPL"\n' '2019-01-02,"BF.B"\n'
    df = parse_snapshot_table(csv_text)

    assert list(df["date"]) == [pd.Timestamp("2019-01-02"), pd.Timestamp("2020-01-02")]
    assert df.iloc[0]["tickers"] == ["BF-B"]
    assert df.iloc[1]["tickers"] == ["AAPL", "BRK-B"]


def test_parse_snapshot_table_dedupes_same_date_keeping_last():
    csv_text = "date,tickers\n" '2020-01-02,"AAPL"\n' '2020-01-02,"AAPL,MSFT"\n'
    df = parse_snapshot_table(csv_text)

    assert len(df) == 1
    assert df.iloc[0]["tickers"] == ["AAPL", "MSFT"]


def test_build_membership_intervals_open_ended_for_stock_still_present():
    snapshot = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-01", "2020-02-01", "2020-03-01"]),
            "tickers": [["AAPL", "MSFT"], ["AAPL", "MSFT"], ["AAPL", "MSFT"]],
        }
    )
    intervals = build_membership_intervals(snapshot)

    aapl = intervals[intervals["stock_id"] == "AAPL"]
    assert len(aapl) == 1
    assert aapl.iloc[0]["start_date"] == pd.Timestamp("2020-01-01")
    assert pd.isna(aapl.iloc[0]["end_date"])


def test_build_membership_intervals_closes_interval_on_removal():
    snapshot = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-01", "2020-02-01", "2020-03-01"]),
            "tickers": [["AAPL", "GONE"], ["AAPL"], ["AAPL"]],
        }
    )
    intervals = build_membership_intervals(snapshot)

    gone = intervals[intervals["stock_id"] == "GONE"]
    assert len(gone) == 1
    assert gone.iloc[0]["start_date"] == pd.Timestamp("2020-01-01")
    assert gone.iloc[0]["end_date"] == pd.Timestamp("2020-02-01")


def test_build_membership_intervals_splits_readded_stock_into_two_intervals():
    snapshot = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-01", "2020-02-01", "2020-03-01", "2020-04-01"]),
            "tickers": [["FLIP"], [], ["FLIP"], ["FLIP"]],
        }
    )
    intervals = build_membership_intervals(snapshot)

    flip = intervals[intervals["stock_id"] == "FLIP"].sort_values("start_date").reset_index(drop=True)
    assert len(flip) == 2
    assert flip.iloc[0]["start_date"] == pd.Timestamp("2020-01-01")
    assert flip.iloc[0]["end_date"] == pd.Timestamp("2020-02-01")
    assert flip.iloc[1]["start_date"] == pd.Timestamp("2020-03-01")
    assert pd.isna(flip.iloc[1]["end_date"])
