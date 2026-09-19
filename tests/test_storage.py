import pandas as pd
import pytest

from tw_quant.storage import SQLiteDataStore
from tw_quant.storage import _membership_rows as build_membership_rows


@pytest.fixture
def store(tmp_path):
    return SQLiteDataStore(db_path=tmp_path / "test.db")


def _price_rows(dates, stock_id="2330"):
    return pd.DataFrame(
        {
            "date": dates,
            "stock_id": [stock_id] * len(dates),
            "industry": ["半導體"] * len(dates),
            "open": [100.0] * len(dates),
            "high": [101.0] * len(dates),
            "low": [99.0] * len(dates),
            "close": [100.5] * len(dates),
            "volume": [1_000_000] * len(dates),
            "turnover_value": [100_500_000.0] * len(dates),
        }
    )


def test_upsert_and_load_prices_round_trip(store):
    dates = pd.bdate_range("2024-01-01", periods=5)
    store.upsert_prices(_price_rows(dates))

    loaded = store.load_prices()
    assert len(loaded) == 5
    assert set(loaded["stock_id"]) == {"2330"}
    assert loaded["close"].iloc[0] == 100.5


def test_upsert_is_idempotent_no_duplicates(store):
    dates = pd.bdate_range("2024-01-01", periods=3)
    store.upsert_prices(_price_rows(dates))
    store.upsert_prices(_price_rows(dates))  # 重複寫入同一批資料

    loaded = store.load_prices()
    assert len(loaded) == 3  # 主鍵 (date, stock_id) 防止重複


def test_upsert_overwrites_changed_values():
    from tw_quant.storage import SQLiteDataStore
    import tempfile, os

    with tempfile.TemporaryDirectory() as d:
        store = SQLiteDataStore(db_path=os.path.join(d, "test.db"))
        dates = pd.bdate_range("2024-01-01", periods=1)
        store.upsert_prices(_price_rows(dates))

        revised = _price_rows(dates)
        revised["close"] = 999.0
        store.upsert_prices(revised)

        loaded = store.load_prices()
        assert len(loaded) == 1
        assert loaded["close"].iloc[0] == 999.0


def test_load_prices_filters_by_date_range(store):
    dates = pd.bdate_range("2024-01-01", periods=10)
    store.upsert_prices(_price_rows(dates))

    loaded = store.load_prices(start_date="2024-01-04", end_date="2024-01-08")
    assert loaded["date"].min() >= pd.Timestamp("2024-01-04")
    assert loaded["date"].max() <= pd.Timestamp("2024-01-08")


def test_load_prices_filters_by_stock_ids(store):
    dates = pd.bdate_range("2024-01-01", periods=3)
    store.upsert_prices(_price_rows(dates, "2330"))
    store.upsert_prices(_price_rows(dates, "2317"))

    loaded = store.load_prices(stock_ids=["2317"])
    assert set(loaded["stock_id"]) == {"2317"}


def test_margin_short_round_trip(store):
    dates = pd.bdate_range("2024-01-01", periods=4)
    df = pd.DataFrame(
        {
            "date": dates,
            "stock_id": ["2330"] * len(dates),
            "margin_purchase_balance": [1000.0] * len(dates),
            "short_balance": [50.0] * len(dates),
        }
    )
    store.upsert_margin_short(df)
    loaded = store.load_margin_short()
    assert len(loaded) == 4
    assert loaded["short_balance"].iloc[0] == 50.0


def test_month_revenue_round_trip(store):
    dates = pd.to_datetime(["2024-01-01", "2024-02-01", "2024-03-01"])
    df = pd.DataFrame(
        {
            "date": dates,
            "stock_id": ["2330"] * len(dates),
            "revenue": [1e11, 1.1e11, 1.2e11],
            "revenue_year": [2024, 2024, 2024],
            "revenue_month": [1, 2, 3],
        }
    )
    store.upsert_month_revenue(df)
    loaded = store.load_month_revenue()
    assert len(loaded) == 3
    assert loaded["revenue"].iloc[0] == 1e11


def test_month_revenue_is_idempotent_no_duplicates(store):
    dates = pd.to_datetime(["2024-01-01"])
    df = pd.DataFrame(
        {"date": dates, "stock_id": ["2330"], "revenue": [1e11], "revenue_year": [2024], "revenue_month": [1]}
    )
    store.upsert_month_revenue(df)
    store.upsert_month_revenue(df)
    assert len(store.load_month_revenue()) == 1


def test_shares_issued_round_trip(store):
    dates = pd.to_datetime(["2024-01-01", "2024-01-02"])
    df = pd.DataFrame({"date": dates, "stock_id": ["2330"] * 2, "shares_issued": [25932070992.0, 25932070992.0]})
    store.upsert_shares_issued(df)
    loaded = store.load_shares_issued()
    assert len(loaded) == 2
    assert loaded["shares_issued"].iloc[0] == 25932070992.0


def test_shares_issued_is_idempotent_no_duplicates(store):
    dates = pd.to_datetime(["2024-01-01"])
    df = pd.DataFrame({"date": dates, "stock_id": ["2330"], "shares_issued": [25932070992.0]})
    store.upsert_shares_issued(df)
    store.upsert_shares_issued(df)
    assert len(store.load_shares_issued()) == 1


def test_us_prices_round_trip_and_stays_separate_from_tw_prices(store):
    dates = pd.bdate_range("2024-01-01", periods=3)
    tw_df = _price_rows(dates, "2330")
    us_df = pd.DataFrame(
        {
            "date": dates, "stock_id": ["AAPL"] * len(dates), "industry": ["Information Technology"] * len(dates),
            "open": [190.0] * len(dates), "high": [191.0] * len(dates), "low": [189.0] * len(dates),
            "close": [190.5] * len(dates), "volume": [50_000_000] * len(dates), "turnover_value": [9.5e9] * len(dates),
        }
    )
    store.upsert_prices(tw_df)
    store.upsert_us_prices(us_df)

    loaded_us = store.load_us_prices()
    assert len(loaded_us) == 3
    assert set(loaded_us["stock_id"]) == {"AAPL"}
    assert loaded_us["close"].iloc[0] == 190.5

    # 兩個市場的資料要各自獨立，不會互相污染
    loaded_tw = store.load_prices()
    assert set(loaded_tw["stock_id"]) == {"2330"}


def test_us_prices_is_idempotent_no_duplicates(store):
    dates = pd.bdate_range("2024-01-01", periods=2)
    df = pd.DataFrame(
        {
            "date": dates, "stock_id": ["AAPL"] * len(dates), "industry": ["Information Technology"] * len(dates),
            "open": [190.0] * len(dates), "high": [191.0] * len(dates), "low": [189.0] * len(dates),
            "close": [190.5] * len(dates), "volume": [50_000_000] * len(dates), "turnover_value": [9.5e9] * len(dates),
        }
    )
    store.upsert_us_prices(df)
    store.upsert_us_prices(df)
    assert len(store.load_us_prices()) == 2


def _membership_rows(stock_ids, start_dates, end_dates=None):
    return pd.DataFrame(
        {
            "stock_id": stock_ids,
            "start_date": pd.to_datetime(start_dates),
            "end_date": pd.to_datetime(end_dates) if end_dates is not None else [pd.NaT] * len(stock_ids),
        }
    )


def test_membership_rows_uses_real_none_not_float_nan_for_missing_end_date():
    """回歸測試：曾經在正式 Postgres 資料庫上炸掉的 bug——`series.where(cond,
    None) 賦值回 DataFrame 欄位時，在 pandas 3.x 的新版字串 dtype 下會把
    None 悄悄轉成 float('nan')，SQLite 對型別不敏感所以本地測試一直沒抓到，
    直到全新的 Postgres 資料庫執行 upsert 才用 DatatypeMismatch 報錯
    （'NaN'::float 塞進 DATE 欄位）。這裡直接測 _membership_rows 這個組
    tuple 的函式本身，確保缺值真的是 Python 的 None，不是 float nan。
    """
    df = pd.DataFrame(
        {"stock_id": ["MMM", "CELG"], "start_date": pd.to_datetime(["1957-03-04", "2005-01-01"])}
    )
    df["end_date"] = pd.NaT
    df.loc[1, "end_date"] = pd.Timestamp("2019-11-21")

    rows = build_membership_rows(df)

    assert rows[0] == ("MMM", "1957-03-04", None)
    assert rows[0][2] is None
    assert rows[1] == ("CELG", "2005-01-01", "2019-11-21")


def test_us_index_membership_round_trip(store):
    df = _membership_rows(["AAPL", "NVDA"], ["1982-11-30", "2001-06-08"])
    store.upsert_us_index_membership(df)

    loaded = store.load_us_index_membership()
    assert len(loaded) == 2
    started = loaded.set_index("stock_id")["start_date"]
    assert started["AAPL"] == pd.Timestamp("1982-11-30")
    assert started["NVDA"] == pd.Timestamp("2001-06-08")
    assert loaded["end_date"].isna().all()


def test_us_index_membership_round_trip_with_end_date(store):
    df = _membership_rows(["CELG"], ["2005-01-01"], ["2019-11-21"])
    store.upsert_us_index_membership(df)

    loaded = store.load_us_index_membership()
    assert loaded["end_date"].iloc[0] == pd.Timestamp("2019-11-21")


def test_us_index_membership_upsert_overwrites_same_start_date(store):
    store.upsert_us_index_membership(_membership_rows(["AAPL"], ["1982-11-30"]))
    store.upsert_us_index_membership(_membership_rows(["AAPL"], ["1982-11-30"], ["2000-01-01"]))

    loaded = store.load_us_index_membership()
    assert len(loaded) == 1
    assert loaded["end_date"].iloc[0] == pd.Timestamp("2000-01-01")


def test_us_index_membership_upsert_adds_second_interval_for_different_start_date(store):
    """同一檔股票中途被剔除又重新加入：兩段不同 start_date 的區間該是兩列，
    不是互相覆蓋——這是跟舊版（PK 只有 stock_id）行為不同的地方。
    """
    store.upsert_us_index_membership(_membership_rows(["FLIP"], ["2010-01-01"], ["2015-01-01"]))
    store.upsert_us_index_membership(_membership_rows(["FLIP"], ["2018-01-01"]))

    loaded = store.load_us_index_membership()
    assert len(loaded) == 2
    assert set(loaded["start_date"]) == {pd.Timestamp("2010-01-01"), pd.Timestamp("2018-01-01")}


def test_us_index_membership_load_returns_empty_frame_when_no_data(store):
    loaded = store.load_us_index_membership()
    assert loaded.empty
    assert list(loaded.columns) == ["stock_id", "start_date", "end_date"]


def test_latest_date_returns_none_when_empty(store):
    assert store.latest_date("prices") is None


def test_latest_date_returns_max_date(store):
    dates = pd.bdate_range("2024-01-01", periods=5)
    store.upsert_prices(_price_rows(dates))
    assert store.latest_date("prices") == pd.Timestamp(dates.max())


def test_latest_date_is_per_stock_not_global(store):
    """一檔股票已同步到最新、另一檔完全沒資料時，兩者的 latest_date 不該互相污染

    ——這是實測接 Neon 時發現的真實 bug：ingest 腳本原本只查「全市場最新
    日期」當同步起點，若某一批裡有一檔已經同步過，其餘從未同步過的股票會
    被誤判成「已經有資料」，之後永遠只抓 lookback_days 天，補不回完整歷史。
    """
    old_dates = pd.bdate_range("2024-01-01", periods=3)
    recent_dates = pd.bdate_range("2024-06-01", periods=3)
    store.upsert_prices(_price_rows(old_dates, stock_id="2330"))
    store.upsert_prices(_price_rows(recent_dates, stock_id="2317"))

    assert store.latest_date("prices", stock_id="2317") == pd.Timestamp(recent_dates.max())
    assert store.latest_date("prices", stock_id="2330") == pd.Timestamp(old_dates.max())
    assert store.latest_date("prices", stock_id="0000") is None  # 從未同步過的股票
    assert store.latest_date("prices") == pd.Timestamp(recent_dates.max())  # 不帶 stock_id 仍是全域行為


def test_get_data_store_defaults_to_sqlite(tmp_path, monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setenv("SQLITE_DB_PATH", str(tmp_path / "market.db"))
    from tw_quant.storage import get_data_store, SQLiteDataStore

    store = get_data_store()
    assert isinstance(store, SQLiteDataStore)
    assert (tmp_path / "market.db").exists()
