"""測試 ingest_daily_data.py 的邏輯（不打真的網路，FinMind 被 monkeypatch 掉）。

這個沙盒環境連不到 FinMind，所以這裡只驗證「假設 FinMind 回傳這些資料，
腳本會不會正確寫進資料庫」，不驗證 FinMind 本身的欄位格式是否正確
（那個假設寫在 tw_quant/data_provider.py 檔頭註解裡，需要使用者在有網路的
環境下自行驗證）。
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import ingest_daily_data


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


def test_load_universe_prefers_env_list(monkeypatch):
    monkeypatch.setenv("STOCK_UNIVERSE", "2330, 2317 ,2454")
    assert ingest_daily_data._load_universe() == ["2330", "2317", "2454"]


def test_load_universe_falls_back_to_default(monkeypatch):
    monkeypatch.delenv("STOCK_UNIVERSE", raising=False)
    monkeypatch.delenv("STOCK_UNIVERSE_FILE", raising=False)
    assert ingest_daily_data._load_universe() == ingest_daily_data.DEFAULT_UNIVERSE
