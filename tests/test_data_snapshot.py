"""測試 tw_quant/data_snapshot.py 的匯出/讀取邏輯本身（不連資料庫，用假
store + tmp_path 做完整的寫入→讀回往返測試）。"""

import sys
from pathlib import Path
from unittest import mock

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tw_quant.data_snapshot import (
    export_us_earnings_snapshot,
    export_us_snapshot,
    load_us_earnings_snapshot,
    load_us_index_membership_snapshot,
    load_us_prices_snapshot,
)


def _fake_store(us_prices: pd.DataFrame, membership: pd.DataFrame):
    store = mock.Mock()
    store.load_us_prices.return_value = us_prices
    store.load_us_index_membership.return_value = membership
    return store


def test_export_then_load_round_trips_prices_and_membership(tmp_path):
    us_prices = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-02", "2020-01-03"]),
            "stock_id": ["AAPL", "AAPL"],
            "industry": ["Tech", "Tech"],
            "open": [1.0, 2.0],
            "high": [1.5, 2.5],
            "low": [0.5, 1.5],
            "close": [1.2, 2.2],
            "volume": [1000, 2000],
            "turnover_value": [1200.0, 4400.0],
        }
    )
    membership = pd.DataFrame(
        {
            "stock_id": ["AAPL", "CELG"],
            "start_date": pd.to_datetime(["1980-01-01", "2005-01-01"]),
            "end_date": [pd.NaT, pd.Timestamp("2019-11-21")],
        }
    )
    store = _fake_store(us_prices, membership)

    prices_path = tmp_path / "prices.parquet"
    membership_path = tmp_path / "membership.parquet"
    n_prices, n_membership = export_us_snapshot(store, prices_path=prices_path, membership_path=membership_path)

    assert n_prices == 2
    assert n_membership == 2
    assert prices_path.exists()
    assert membership_path.exists()

    loaded_prices = load_us_prices_snapshot(prices_path)
    loaded_membership = load_us_index_membership_snapshot(membership_path)

    pd.testing.assert_frame_equal(loaded_prices, us_prices)
    assert loaded_membership["start_date"].iloc[0] == pd.Timestamp("1980-01-01")
    assert pd.isna(loaded_membership["end_date"].iloc[0])
    assert loaded_membership["end_date"].iloc[1] == pd.Timestamp("2019-11-21")


def test_export_splits_us_prices_into_part_files_above_chunk_size(tmp_path):
    """2026-09-20：資料量拉大到 20 年時單一 us_prices_snapshot.parquet
    超過 GitHub 100MB 單檔上限被擋下，改成依列數切成多檔（見
    tw_quant/data_snapshot.py 檔頭「us_prices 分檔」說明）。這裡用很小的
    max_rows_per_chunk（不用真的生出百萬列資料）驗證切檔 + 讀回 concat
    這條路徑本身是對的。
    """
    us_prices = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06", "2020-01-07", "2020-01-08"]),
            "stock_id": ["AAPL"] * 5,
            "industry": ["Tech"] * 5,
            "open": [1.0, 2.0, 3.0, 4.0, 5.0],
            "high": [1.5, 2.5, 3.5, 4.5, 5.5],
            "low": [0.5, 1.5, 2.5, 3.5, 4.5],
            "close": [1.2, 2.2, 3.2, 4.2, 5.2],
            "volume": [1000, 2000, 3000, 4000, 5000],
            "turnover_value": [1200.0, 4400.0, 9600.0, 16800.0, 26000.0],
        }
    )
    membership = pd.DataFrame({"stock_id": ["AAPL"], "start_date": pd.to_datetime(["1980-01-01"]), "end_date": [pd.NaT]})
    store = _fake_store(us_prices, membership)

    prices_path = tmp_path / "prices.parquet"
    membership_path = tmp_path / "membership.parquet"
    n_prices, _ = export_us_snapshot(store, prices_path=prices_path, membership_path=membership_path, max_rows_per_chunk=2)

    assert n_prices == 5
    assert prices_path.exists()
    part_files = sorted(tmp_path.glob("prices.part*.parquet"))
    assert [p.name for p in part_files] == ["prices.part2.parquet", "prices.part3.parquet"]

    loaded = load_us_prices_snapshot(prices_path)
    pd.testing.assert_frame_equal(loaded, us_prices)


def test_export_removes_stale_part_files_when_row_count_shrinks(tmp_path):
    """重新匯出時資料量變少、需要的 part 檔案數也變少——舊的多餘 part
    檔案要被清掉，不然讀取端 glob 進來會把已經過期的資料當成額外列
    concat 進去，變成重複資料。
    """
    big = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06", "2020-01-07"]),
            "stock_id": ["AAPL"] * 4,
            "industry": ["Tech"] * 4,
            "open": [1.0, 2.0, 3.0, 4.0],
            "high": [1.5, 2.5, 3.5, 4.5],
            "low": [0.5, 1.5, 2.5, 3.5],
            "close": [1.2, 2.2, 3.2, 4.2],
            "volume": [1000, 2000, 3000, 4000],
            "turnover_value": [1200.0, 4400.0, 9600.0, 16800.0],
        }
    )
    membership = pd.DataFrame({"stock_id": ["AAPL"], "start_date": pd.to_datetime(["1980-01-01"]), "end_date": [pd.NaT]})
    prices_path = tmp_path / "prices.parquet"
    membership_path = tmp_path / "membership.parquet"

    export_us_snapshot(_fake_store(big, membership), prices_path=prices_path, membership_path=membership_path, max_rows_per_chunk=1)
    assert len(list(tmp_path.glob("prices.part*.parquet"))) == 3  # 4 列切成 4 檔：base + part2/3/4

    small = big.iloc[:1].reset_index(drop=True)
    n_prices, _ = export_us_snapshot(
        _fake_store(small, membership), prices_path=prices_path, membership_path=membership_path, max_rows_per_chunk=1
    )

    assert n_prices == 1
    assert list(tmp_path.glob("prices.part*.parquet")) == []
    loaded = load_us_prices_snapshot(prices_path)
    pd.testing.assert_frame_equal(loaded, small)


def test_load_us_prices_snapshot_raises_clear_error_when_missing(tmp_path):
    missing_path = tmp_path / "does_not_exist.parquet"
    with pytest.raises(FileNotFoundError, match="找不到美股價量快照檔"):
        load_us_prices_snapshot(missing_path)


def test_load_us_index_membership_snapshot_raises_clear_error_when_missing(tmp_path):
    missing_path = tmp_path / "does_not_exist.parquet"
    with pytest.raises(FileNotFoundError, match="找不到美股指數成分股快照檔"):
        load_us_index_membership_snapshot(missing_path)


def test_export_creates_parent_directory_if_missing(tmp_path):
    store = _fake_store(pd.DataFrame(columns=["date", "stock_id"]), pd.DataFrame(columns=["stock_id", "start_date", "end_date"]))
    nested_prices = tmp_path / "nested" / "prices.parquet"
    nested_membership = tmp_path / "nested" / "membership.parquet"

    export_us_snapshot(store, prices_path=nested_prices, membership_path=nested_membership)

    assert nested_prices.exists()
    assert nested_membership.exists()


def test_export_then_load_round_trips_earnings(tmp_path):
    earnings = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-25", "2024-04-25"]),
            "stock_id": ["AAPL", "AAPL"],
            "eps_estimate": [1.0, 1.1],
            "eps_actual": [1.05, 1.2],
            "surprise_pct": [5.0, 9.1],
        }
    )
    store = mock.Mock()
    store.load_us_earnings.return_value = earnings

    path = tmp_path / "earnings.parquet"
    n_rows = export_us_earnings_snapshot(store, path=path)

    assert n_rows == 2
    assert path.exists()

    loaded = load_us_earnings_snapshot(path)
    pd.testing.assert_frame_equal(loaded, earnings)


def test_load_us_earnings_snapshot_raises_clear_error_when_missing(tmp_path):
    missing_path = tmp_path / "does_not_exist.parquet"
    with pytest.raises(FileNotFoundError, match="找不到美股財報公布快照檔"):
        load_us_earnings_snapshot(missing_path)


def test_export_us_earnings_snapshot_creates_parent_directory_if_missing(tmp_path):
    store = mock.Mock()
    store.load_us_earnings.return_value = pd.DataFrame(columns=["date", "stock_id"])
    nested_path = tmp_path / "nested" / "earnings.parquet"

    export_us_earnings_snapshot(store, path=nested_path)

    assert nested_path.exists()
