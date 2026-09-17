"""PostgresDataStore 的連線行為測試（用 mock psycopg2，不碰真實網路）。

這裡存在的原因：第一次接上真實 Neon 資料庫時，GitHub Actions 的
ingest workflow 卡在 "Run ingestion" 步驟超過 8 分鐘沒有任何輸出或錯誤，
根本原因是 psycopg2.connect() 沒有帶 connect_timeout，遇到連線異常時會
無限期卡住而不是快速失敗。這裡釘住「一定要帶 connect_timeout /
statement_timeout」與「同一個 store 物件要重用同一條連線」這兩個行為，
防止回歸。
"""

import sys
import types
from unittest.mock import MagicMock

import pytest


def _install_fake_psycopg2(monkeypatch):
    """psycopg2 不是這個沙盒環境的必要依賴，用假模組避免測試需要真的裝它、
    也避免測試意外連真的網路。
    """
    fake_conn = MagicMock()
    fake_conn.closed = False
    fake_cursor_cm = MagicMock()
    fake_cursor_cm.__enter__.return_value = MagicMock()
    fake_cursor_cm.__exit__.return_value = False
    fake_conn.cursor.return_value = fake_cursor_cm
    fake_conn.__enter__.return_value = fake_conn
    fake_conn.__exit__.return_value = False

    connect_mock = MagicMock(return_value=fake_conn)
    fake_module = types.SimpleNamespace(connect=connect_mock)
    monkeypatch.setitem(sys.modules, "psycopg2", fake_module)
    return connect_mock, fake_conn


def test_connect_passes_timeout_kwargs(monkeypatch):
    connect_mock, _ = _install_fake_psycopg2(monkeypatch)
    from tw_quant.storage import PostgresDataStore

    PostgresDataStore("postgresql://u:p@host/db", connect_timeout=7, statement_timeout_ms=15_000)

    assert connect_mock.call_count == 1
    _, kwargs = connect_mock.call_args
    assert kwargs["connect_timeout"] == 7
    assert "statement_timeout=15000" in kwargs["options"]


def test_connection_is_reused_across_calls(monkeypatch):
    connect_mock, _ = _install_fake_psycopg2(monkeypatch)
    from tw_quant.storage import PostgresDataStore
    import pandas as pd

    store = PostgresDataStore("postgresql://u:p@host/db")
    df = pd.DataFrame(
        {
            "date": pd.bdate_range("2024-01-01", periods=1),
            "stock_id": ["2330"],
            "industry": ["半導體業"],
            "open": [100.0],
            "high": [101.0],
            "low": [99.0],
            "close": [100.5],
            "volume": [1000],
            "turnover_value": [100500.0],
        }
    )
    store.upsert_prices(df)
    store.upsert_prices(df)

    # __init__ 的 _init_schema 加上兩次 upsert，只應該真正 connect() 一次
    assert connect_mock.call_count == 1


def test_default_timeouts_are_sane(monkeypatch):
    connect_mock, _ = _install_fake_psycopg2(monkeypatch)
    from tw_quant.storage import PostgresDataStore

    PostgresDataStore("postgresql://u:p@host/db")
    _, kwargs = connect_mock.call_args
    assert kwargs["connect_timeout"] > 0
    assert kwargs["connect_timeout"] <= 30  # 太長就失去「快速失敗」的意義
