"""PostgresDataStore 的連線行為測試（用 mock psycopg2，不碰真實網路）。

這裡存在的原因：第一次接上真實 Neon 資料庫時，一路踩過幾個只有實測才會
暴露的問題：
1. psycopg2.connect() 沒有帶 connect_timeout，遇到連線異常會無限期卡住。
2. 把 statement_timeout 塞進 connect() 的 options 參數，Neon 的 pooler
   endpoint 直接拒絕連線，改成連線建立後另外執行 SET 指令才行。
3. 全量回填一檔股票 3 年資料（~729 列）用 cursor.executemany() 要將近
   6 分鐘——executemany 對 psycopg2 來說不會真的打包成一次網路往返，是
   每一列各自送一次，729 列就是 729 次往返。這才是每次「全量回填」的
   GitHub Actions 執行一路卡到 15 分鐘逾時上限的真正原因，不是任何一次
   之前懷疑的連線/速率限制問題。改用 psycopg2.extras.execute_values()
   把整批資料打包成一條多列 INSERT。

這裡的測試釘住這幾個行為，防止回歸。
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
    fake_cursor = MagicMock()
    fake_cursor_cm = MagicMock()
    fake_cursor_cm.__enter__.return_value = fake_cursor
    fake_cursor_cm.__exit__.return_value = False
    fake_conn.cursor.return_value = fake_cursor_cm
    fake_conn.__enter__.return_value = fake_conn
    fake_conn.__exit__.return_value = False

    connect_mock = MagicMock(return_value=fake_conn)
    execute_values_mock = MagicMock()
    fake_extras_module = types.SimpleNamespace(execute_values=execute_values_mock)
    fake_module = types.SimpleNamespace(connect=connect_mock, extras=fake_extras_module)
    monkeypatch.setitem(sys.modules, "psycopg2", fake_module)
    monkeypatch.setitem(sys.modules, "psycopg2.extras", fake_extras_module)
    return connect_mock, fake_conn, fake_cursor, execute_values_mock


def test_connect_passes_timeout_kwargs(monkeypatch):
    connect_mock, _, _, _ = _install_fake_psycopg2(monkeypatch)
    from tw_quant.storage import PostgresDataStore

    PostgresDataStore("postgresql://u:p@host/db", connect_timeout=7, statement_timeout_ms=15_000)

    assert connect_mock.call_count == 1
    _, kwargs = connect_mock.call_args
    assert kwargs["connect_timeout"] == 7
    assert "options" not in kwargs  # 不能塞進 startup packet，pooler 會直接拒絕連線


def test_statement_timeout_set_via_sql_not_startup_options(monkeypatch):
    _, _, fake_cursor, _ = _install_fake_psycopg2(monkeypatch)
    from tw_quant.storage import PostgresDataStore

    PostgresDataStore("postgresql://u:p@host/db", statement_timeout_ms=15_000)

    executed_sql = [call.args[0] for call in fake_cursor.execute.call_args_list]
    assert any("SET statement_timeout = 15000" in sql for sql in executed_sql)


def test_connection_is_reused_across_calls(monkeypatch):
    connect_mock, _, _, _ = _install_fake_psycopg2(monkeypatch)
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


def test_latest_date_filters_by_stock_id(monkeypatch):
    _, fake_conn, fake_cursor, _ = _install_fake_psycopg2(monkeypatch)
    from tw_quant.storage import PostgresDataStore

    fake_cursor.fetchone.return_value = ("2024-06-01",)
    store = PostgresDataStore("postgresql://u:p@host/db")
    store.latest_date("prices", stock_id="2330")

    last_call_sql, last_call_params = fake_cursor.execute.call_args_list[-1].args
    assert "WHERE stock_id = %s" in last_call_sql
    assert last_call_params == ("2330",)


def test_upsert_prices_uses_execute_values_not_row_by_row(monkeypatch):
    _, _, fake_cursor, execute_values_mock = _install_fake_psycopg2(monkeypatch)
    from tw_quant.storage import PostgresDataStore
    import pandas as pd

    store = PostgresDataStore("postgresql://u:p@host/db")
    df = pd.DataFrame(
        {
            "date": pd.bdate_range("2024-01-01", periods=3),
            "stock_id": ["2330"] * 3,
            "industry": ["半導體業"] * 3,
            "open": [100.0] * 3,
            "high": [101.0] * 3,
            "low": [99.0] * 3,
            "close": [100.5] * 3,
            "volume": [1000] * 3,
            "turnover_value": [100500.0] * 3,
        }
    )
    store.upsert_prices(df)

    assert execute_values_mock.call_count == 1
    _, sql, rows = execute_values_mock.call_args.args
    assert "VALUES %s" in sql
    assert len(rows) == 3
    fake_cursor.executemany.assert_not_called()


def test_upsert_margin_short_uses_execute_values(monkeypatch):
    _, _, fake_cursor, execute_values_mock = _install_fake_psycopg2(monkeypatch)
    from tw_quant.storage import PostgresDataStore
    import pandas as pd

    store = PostgresDataStore("postgresql://u:p@host/db")
    df = pd.DataFrame(
        {
            "date": pd.bdate_range("2024-01-01", periods=2),
            "stock_id": ["2330"] * 2,
            "margin_purchase_balance": [1000.0] * 2,
            "short_balance": [30.0] * 2,
        }
    )
    store.upsert_margin_short(df)

    assert execute_values_mock.call_count == 1
    _, sql, rows = execute_values_mock.call_args.args
    assert "VALUES %s" in sql
    assert len(rows) == 2
    fake_cursor.executemany.assert_not_called()


def test_default_timeouts_are_sane(monkeypatch):
    connect_mock, _, _, _ = _install_fake_psycopg2(monkeypatch)
    from tw_quant.storage import PostgresDataStore

    PostgresDataStore("postgresql://u:p@host/db")
    _, kwargs = connect_mock.call_args
    assert kwargs["connect_timeout"] > 0
    assert kwargs["connect_timeout"] <= 30  # 太長就失去「快速失敗」的意義
