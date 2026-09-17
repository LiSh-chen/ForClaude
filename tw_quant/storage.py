"""資料落地層：把抓下來的價量 / 融資券資料寫進資料庫，供回測直接讀取。

設計目標：**今天就能動、以後換雲端資料庫不用改任何程式碼**。

- 預設（沒有設定 `DATABASE_URL`）：寫進本機 SQLite 檔案
  （預設路徑 `data/tw_market.db`）。在 GitHub Actions 排程情境下，這個檔案
  由 workflow 直接 commit 回 repo，等於用 git 當免費、零設定的暫時資料庫。
- 一旦你申請好雲端 Postgres（RDS / Supabase / Neon 等皆可），只要在 repo
  設定 `DATABASE_URL` 這個 GitHub Secret（例如
  `postgresql://user:pass@host:5432/dbname`），`get_data_store()` 會自動
  切換到 `PostgresDataStore`，ingest 腳本與回測程式碼完全不用改。

兩種實作都是「以 (date, stock_id) 為主鍵的 upsert」，重複抓同一天資料不會
產生重複列，可以放心每天無腦全量重抓最近 N 天做補資料。
"""

from __future__ import annotations

import os
import sqlite3
from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd

PRICE_COLS = ["date", "stock_id", "industry", "open", "high", "low", "close", "volume", "turnover_value"]
MARGIN_COLS = ["date", "stock_id", "margin_purchase_balance", "short_balance"]


class DataStore(ABC):
    @abstractmethod
    def upsert_prices(self, df: pd.DataFrame) -> None: ...

    @abstractmethod
    def upsert_margin_short(self, df: pd.DataFrame) -> None: ...

    @abstractmethod
    def load_prices(
        self, start_date: str | None = None, end_date: str | None = None, stock_ids: list[str] | None = None
    ) -> pd.DataFrame: ...

    @abstractmethod
    def load_margin_short(
        self, start_date: str | None = None, end_date: str | None = None, stock_ids: list[str] | None = None
    ) -> pd.DataFrame: ...

    @abstractmethod
    def latest_date(self, table: str) -> pd.Timestamp | None: ...


def _normalize_dates(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["date"] = pd.to_datetime(d["date"]).dt.strftime("%Y-%m-%d")
    return d


class SQLiteDataStore(DataStore):
    def __init__(self, db_path: str | Path = "data/tw_market.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS prices (
                    date TEXT NOT NULL, stock_id TEXT NOT NULL, industry TEXT,
                    open REAL, high REAL, low REAL, close REAL,
                    volume INTEGER, turnover_value REAL,
                    PRIMARY KEY (date, stock_id)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS margin_short (
                    date TEXT NOT NULL, stock_id TEXT NOT NULL,
                    margin_purchase_balance REAL, short_balance REAL,
                    PRIMARY KEY (date, stock_id)
                )
                """
            )

    def upsert_prices(self, df: pd.DataFrame) -> None:
        if df.empty:
            return
        d = _normalize_dates(df)
        rows = list(d[PRICE_COLS].itertuples(index=False, name=None))
        with self._connect() as conn:
            conn.executemany(
                "INSERT OR REPLACE INTO prices "
                "(date, stock_id, industry, open, high, low, close, volume, turnover_value) "
                "VALUES (?,?,?,?,?,?,?,?,?)",
                rows,
            )

    def upsert_margin_short(self, df: pd.DataFrame) -> None:
        if df.empty:
            return
        d = _normalize_dates(df)
        rows = list(d[MARGIN_COLS].itertuples(index=False, name=None))
        with self._connect() as conn:
            conn.executemany(
                "INSERT OR REPLACE INTO margin_short "
                "(date, stock_id, margin_purchase_balance, short_balance) VALUES (?,?,?,?)",
                rows,
            )

    def _load(self, table: str, columns: list[str], start_date, end_date, stock_ids) -> pd.DataFrame:
        query = f"SELECT * FROM {table} WHERE 1=1"
        params: list = []
        if start_date:
            query += " AND date >= ?"
            params.append(str(start_date))
        if end_date:
            query += " AND date <= ?"
            params.append(str(end_date))
        if stock_ids:
            query += f" AND stock_id IN ({','.join('?' * len(stock_ids))})"
            params.extend(stock_ids)
        with self._connect() as conn:
            df = pd.read_sql_query(query, conn, params=params, parse_dates=["date"])
        if df.empty:
            return pd.DataFrame(columns=columns)
        return df[columns].sort_values(["stock_id", "date"]).reset_index(drop=True)

    def load_prices(self, start_date=None, end_date=None, stock_ids=None) -> pd.DataFrame:
        return self._load("prices", PRICE_COLS, start_date, end_date, stock_ids)

    def load_margin_short(self, start_date=None, end_date=None, stock_ids=None) -> pd.DataFrame:
        return self._load("margin_short", MARGIN_COLS, start_date, end_date, stock_ids)

    def latest_date(self, table: str) -> pd.Timestamp | None:
        with self._connect() as conn:
            row = conn.execute(f"SELECT MAX(date) FROM {table}").fetchone()
        return pd.Timestamp(row[0]) if row and row[0] else None


class PostgresDataStore(DataStore):
    """雲端 Postgres 版本。需要 `pip install psycopg2-binary`（見 pyproject.toml
    的 `cloud` extra），延遲匯入以免沒用到雲端資料庫的人也被迫安裝它。
    """

    def __init__(self, dsn: str):
        import psycopg2  # noqa: PLC0415

        self._psycopg2 = psycopg2
        self.dsn = dsn
        self._init_schema()

    def _connect(self):
        return self._psycopg2.connect(self.dsn)

    def _init_schema(self) -> None:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS prices (
                    date DATE NOT NULL, stock_id TEXT NOT NULL, industry TEXT,
                    open DOUBLE PRECISION, high DOUBLE PRECISION, low DOUBLE PRECISION,
                    close DOUBLE PRECISION, volume BIGINT, turnover_value DOUBLE PRECISION,
                    PRIMARY KEY (date, stock_id)
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS margin_short (
                    date DATE NOT NULL, stock_id TEXT NOT NULL,
                    margin_purchase_balance DOUBLE PRECISION, short_balance DOUBLE PRECISION,
                    PRIMARY KEY (date, stock_id)
                )
                """
            )
            conn.commit()

    def upsert_prices(self, df: pd.DataFrame) -> None:
        if df.empty:
            return
        d = _normalize_dates(df)
        rows = list(d[PRICE_COLS].itertuples(index=False, name=None))
        with self._connect() as conn, conn.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO prices (date, stock_id, industry, open, high, low, close, volume, turnover_value)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
                ON CONFLICT (date, stock_id) DO UPDATE SET
                    industry = EXCLUDED.industry, open = EXCLUDED.open, high = EXCLUDED.high,
                    low = EXCLUDED.low, close = EXCLUDED.close, volume = EXCLUDED.volume,
                    turnover_value = EXCLUDED.turnover_value
                """,
                rows,
            )
            conn.commit()

    def upsert_margin_short(self, df: pd.DataFrame) -> None:
        if df.empty:
            return
        d = _normalize_dates(df)
        rows = list(d[MARGIN_COLS].itertuples(index=False, name=None))
        with self._connect() as conn, conn.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO margin_short (date, stock_id, margin_purchase_balance, short_balance)
                VALUES (%s,%s,%s,%s)
                ON CONFLICT (date, stock_id) DO UPDATE SET
                    margin_purchase_balance = EXCLUDED.margin_purchase_balance,
                    short_balance = EXCLUDED.short_balance
                """,
                rows,
            )
            conn.commit()

    def _load(self, table: str, columns: list[str], start_date, end_date, stock_ids) -> pd.DataFrame:
        query = f"SELECT * FROM {table} WHERE 1=1"
        params: list = []
        if start_date:
            query += " AND date >= %s"
            params.append(str(start_date))
        if end_date:
            query += " AND date <= %s"
            params.append(str(end_date))
        if stock_ids:
            placeholders = ",".join(["%s"] * len(stock_ids))
            query += f" AND stock_id IN ({placeholders})"
            params.extend(stock_ids)
        with self._connect() as conn:
            df = pd.read_sql_query(query, conn, params=params, parse_dates=["date"])
        if df.empty:
            return pd.DataFrame(columns=columns)
        return df[columns].sort_values(["stock_id", "date"]).reset_index(drop=True)

    def load_prices(self, start_date=None, end_date=None, stock_ids=None) -> pd.DataFrame:
        return self._load("prices", PRICE_COLS, start_date, end_date, stock_ids)

    def load_margin_short(self, start_date=None, end_date=None, stock_ids=None) -> pd.DataFrame:
        return self._load("margin_short", MARGIN_COLS, start_date, end_date, stock_ids)

    def latest_date(self, table: str) -> pd.Timestamp | None:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(f"SELECT MAX(date) FROM {table}")
            row = cur.fetchone()
        return pd.Timestamp(row[0]) if row and row[0] else None


def get_data_store() -> DataStore:
    """依環境變數決定要用雲端 Postgres 還是本機 SQLite，全系統只有這裡需要判斷。"""
    dsn = os.environ.get("DATABASE_URL")
    if dsn:
        return PostgresDataStore(dsn)
    return SQLiteDataStore(os.environ.get("SQLITE_DB_PATH", "data/tw_market.db"))
