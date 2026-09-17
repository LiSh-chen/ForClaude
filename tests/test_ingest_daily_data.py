"""測試 ingest_daily_data.py 的邏輯（不打真的網路，FinMind 被 monkeypatch 掉）。

這個沙盒環境連不到 FinMind，所以這裡只驗證「假設 FinMind 回傳這些資料，
腳本會不會正確寫進資料庫」，不驗證 FinMind 本身的欄位格式是否正確
（那個假設寫在 tw_quant/data_provider.py 檔頭註解裡，需要使用者在有網路的
環境下自行驗證）。
"""

import sys
import time
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import ingest_daily_data


def test_call_with_timeout_returns_result_when_fast():
    assert ingest_daily_data._call_with_timeout(lambda: 42, timeout_s=1.0) == 42


def test_call_with_timeout_raises_on_slow_call():
    """requests 的 timeout 參數對「持續慢速吐資料」的連線可能完全不生效
    （實測 FORCE_BACKFILL 那次真的卡了 15 分鐘），這裡驗證備援的
    wall-clock 逾時保護確實能在期限內放棄，不會被卡住的呼叫拖住。
    """

    def _slow():
        time.sleep(5)
        return "too late"

    with pytest.raises(TimeoutError):
        ingest_daily_data._call_with_timeout(_slow, timeout_s=0.2)


def test_call_with_timeout_propagates_exceptions():
    def _boom():
        raise ValueError("網路壞了")

    with pytest.raises(ValueError, match="網路壞了"):
        ingest_daily_data._call_with_timeout(_boom, timeout_s=1.0)


class FakeProvider:
    def __init__(self, token=None):
        self.token = token

    def fetch_stock_info(self):
        return pd.DataFrame({"stock_id": ["2330"], "stock_name": ["台積電"], "industry": ["半導體"], "type": ["twse"]})

    def fetch_price(self, stock_id, start_date, end_date):
        dates = pd.bdate_range(start_date, end_date)
        return pd.DataFrame(
            {
                "date": dates,
                "stock_id": [stock_id] * len(dates),
                "open": [500.0] * len(dates),
                "high": [505.0] * len(dates),
                "low": [495.0] * len(dates),
                "close": [502.0] * len(dates),
                "volume": [10_000_000] * len(dates),
                "turnover_value": [5_020_000_000.0] * len(dates),
            }
        )

    def fetch_margin_short(self, stock_id, start_date, end_date):
        dates = pd.bdate_range(start_date, end_date)
        return pd.DataFrame(
            {
                "date": dates,
                "stock_id": [stock_id] * len(dates),
                "margin_purchase_balance": [1000.0] * len(dates),
                "short_balance": [30.0] * len(dates),
            }
        )


def test_ingest_writes_prices_and_margin_short(tmp_path, monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setenv("SQLITE_DB_PATH", str(tmp_path / "market.db"))
    monkeypatch.setenv("STOCK_UNIVERSE", "2330")
    monkeypatch.setenv("REQUEST_SLEEP_SECONDS", "0")
    monkeypatch.setattr(ingest_daily_data, "FinMindDataProvider", FakeProvider)

    ingest_daily_data.main()

    from tw_quant.storage import get_data_store

    store = get_data_store()
    prices = store.load_prices()
    margin = store.load_margin_short()

    assert not prices.empty
    assert (prices["industry"] == "半導體").all()
    assert not margin.empty


def test_backfill_decision_is_per_stock(tmp_path, monkeypatch):
    """一檔股票已經同步過、另一檔從沒同步過時，前者只做短期增量、
    後者仍要拿到完整 3 年回填——不能被前者的「最近日期」誤判帶偏。
    """
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setenv("SQLITE_DB_PATH", str(tmp_path / "market.db"))
    monkeypatch.setenv("STOCK_UNIVERSE", "2330,2317")
    monkeypatch.setenv("REQUEST_SLEEP_SECONDS", "0")
    monkeypatch.setenv("LOOKBACK_DAYS", "10")
    monkeypatch.setattr(ingest_daily_data, "FinMindDataProvider", FakeProvider)

    from tw_quant.storage import get_data_store

    # 預先讓 2330 已經有「最近」的資料，2317 完全沒有
    store = get_data_store()
    store.upsert_prices(
        pd.DataFrame(
            {
                "date": [pd.Timestamp.today().normalize()],
                "stock_id": ["2330"],
                "industry": ["半導體"],
                "open": [500.0],
                "high": [505.0],
                "low": [495.0],
                "close": [502.0],
                "volume": [10_000_000],
                "turnover_value": [5_020_000_000.0],
            }
        )
    )
    store.close()

    requested_ranges: dict[str, tuple[str, str]] = {}
    original_fetch_price = FakeProvider.fetch_price

    def tracking_fetch_price(self, stock_id, start_date, end_date):
        requested_ranges[stock_id] = (start_date, end_date)
        return original_fetch_price(self, stock_id, start_date, end_date)

    monkeypatch.setattr(FakeProvider, "fetch_price", tracking_fetch_price)

    ingest_daily_data.main()

    start_2330 = pd.Timestamp(requested_ranges["2330"][0])
    start_2317 = pd.Timestamp(requested_ranges["2317"][0])

    # 2330 已同步過 -> 只往前抓 lookback_days 天左右
    assert (pd.Timestamp.today().normalize() - start_2330).days <= 15
    # 2317 從沒同步過 -> 必須拿到接近 3 年的回填起點，不能被 2330 的「最近日期」帶偏
    assert (pd.Timestamp.today().normalize() - start_2317).days >= 365 * 2


def test_force_backfill_ignores_existing_data(tmp_path, monkeypatch):
    """FORCE_BACKFILL=true 時，即使股票已經有資料，也要強制拿完整 3 年回填。

    這是用來修復「某次執行中途被打斷，部分股票只寫進一小段資料，之後被
    per-stock 判斷誤認為已經同步過」這種情況的手動修復開關（實測第一次接
    Neon 時，10 檔demo股票裡有 8 檔就是卡在這個狀態）。
    """
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setenv("SQLITE_DB_PATH", str(tmp_path / "market.db"))
    monkeypatch.setenv("STOCK_UNIVERSE", "2330")
    monkeypatch.setenv("REQUEST_SLEEP_SECONDS", "0")
    monkeypatch.setenv("FORCE_BACKFILL", "true")
    monkeypatch.setattr(ingest_daily_data, "FinMindDataProvider", FakeProvider)

    from tw_quant.storage import get_data_store

    # 2330 只有一小段「最近」的資料（模擬被打斷後的殘缺狀態）
    store = get_data_store()
    store.upsert_prices(
        pd.DataFrame(
            {
                "date": [pd.Timestamp.today().normalize()],
                "stock_id": ["2330"],
                "industry": ["半導體"],
                "open": [500.0],
                "high": [505.0],
                "low": [495.0],
                "close": [502.0],
                "volume": [10_000_000],
                "turnover_value": [5_020_000_000.0],
            }
        )
    )
    store.close()

    requested_ranges: dict[str, tuple[str, str]] = {}
    original_fetch_price = FakeProvider.fetch_price

    def tracking_fetch_price(self, stock_id, start_date, end_date):
        requested_ranges[stock_id] = (start_date, end_date)
        return original_fetch_price(self, stock_id, start_date, end_date)

    monkeypatch.setattr(FakeProvider, "fetch_price", tracking_fetch_price)

    ingest_daily_data.main()

    start_2330 = pd.Timestamp(requested_ranges["2330"][0])
    assert (pd.Timestamp.today().normalize() - start_2330).days >= 365 * 2


def test_load_universe_prefers_env_list(monkeypatch):
    monkeypatch.setenv("STOCK_UNIVERSE", "2330, 2317 ,2454")
    assert ingest_daily_data._load_universe() == ["2330", "2317", "2454"]


def test_load_universe_falls_back_to_default(monkeypatch):
    monkeypatch.delenv("STOCK_UNIVERSE", raising=False)
    monkeypatch.delenv("STOCK_UNIVERSE_FILE", raising=False)
    assert ingest_daily_data._load_universe() == ingest_daily_data.DEFAULT_UNIVERSE
