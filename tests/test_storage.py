import pandas as pd
import pytest

from tw_quant.storage import SQLiteDataStore


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


def test_latest_date_returns_none_when_empty(store):
    assert store.latest_date("prices") is None


def test_latest_date_returns_max_date(store):
    dates = pd.bdate_range("2024-01-01", periods=5)
    store.upsert_prices(_price_rows(dates))
    assert store.latest_date("prices") == pd.Timestamp(dates.max())


def test_get_data_store_defaults_to_sqlite(tmp_path, monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setenv("SQLITE_DB_PATH", str(tmp_path / "market.db"))
    from tw_quant.storage import get_data_store, SQLiteDataStore

    store = get_data_store()
    assert isinstance(store, SQLiteDataStore)
    assert (tmp_path / "market.db").exists()
