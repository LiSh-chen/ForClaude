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
MONTH_REVENUE_COLS = ["date", "stock_id", "revenue", "revenue_year", "revenue_month"]
SHARES_ISSUED_COLS = ["date", "stock_id", "shares_issued"]
US_INDEX_MEMBERSHIP_COLS = ["stock_id", "start_date", "end_date"]
US_EARNINGS_COLS = ["date", "stock_id", "eps_estimate", "eps_actual", "surprise_pct"]  # date = 財報公布日


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
    def upsert_month_revenue(self, df: pd.DataFrame) -> None: ...

    @abstractmethod
    def load_month_revenue(
        self, start_date: str | None = None, end_date: str | None = None, stock_ids: list[str] | None = None
    ) -> pd.DataFrame: ...

    @abstractmethod
    def upsert_shares_issued(self, df: pd.DataFrame) -> None: ...

    @abstractmethod
    def load_shares_issued(
        self, start_date: str | None = None, end_date: str | None = None, stock_ids: list[str] | None = None
    ) -> pd.DataFrame: ...

    @abstractmethod
    def upsert_us_prices(self, df: pd.DataFrame) -> None: ...

    @abstractmethod
    def load_us_prices(
        self, start_date: str | None = None, end_date: str | None = None, stock_ids: list[str] | None = None
    ) -> pd.DataFrame: ...

    @abstractmethod
    def upsert_us_index_membership(self, df: pd.DataFrame) -> None: ...

    @abstractmethod
    def load_us_index_membership(self) -> pd.DataFrame: ...

    @abstractmethod
    def upsert_us_earnings(self, df: pd.DataFrame) -> None: ...

    @abstractmethod
    def load_us_earnings(
        self, start_date: str | None = None, end_date: str | None = None, stock_ids: list[str] | None = None
    ) -> pd.DataFrame: ...

    @abstractmethod
    def latest_date(self, table: str, stock_id: str | None = None) -> pd.Timestamp | None: ...


def _normalize_dates(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["date"] = pd.to_datetime(d["date"]).dt.strftime("%Y-%m-%d")
    return d


def _membership_rows(df: pd.DataFrame) -> list[tuple]:
    """把 upsert_us_index_membership 的輸入轉成 (stock_id, start_date字串,
    end_date字串或None) 的 tuple list，直接用 list comprehension 組、不透過
    中間的 pandas Series/欄位賦值。

    原本用 `series.where(cond, None)` 想把 NaT 換成 None 再寫回 DataFrame
    欄位，結果在 pandas 3.x 的新版字串 dtype 下，None 賦值回欄位時會被
    悄悄轉型成 float('nan')，不是真的 None——SQLite 對這個不敏感（NaN 被
    當成一般值存進去），但 Postgres 的 DATE 欄位型別檢查會直接報
    `DatatypeMismatch`（新的 Neon 資料庫從空的開始 upsert 時才第一次真的
    踩到，SQLite 開發環境一直沒發現）。改成逐列組 tuple、None 就是
    Python 的 None，不會被 pandas 重新推斷型別動過。
    """
    starts = pd.to_datetime(df["start_date"])
    ends = pd.to_datetime(df["end_date"])
    return [
        (stock_id, s.strftime("%Y-%m-%d"), None if pd.isna(e) else e.strftime("%Y-%m-%d"))
        for stock_id, s, e in zip(df["stock_id"], starts, ends)
    ]


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
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS month_revenue (
                    date TEXT NOT NULL, stock_id TEXT NOT NULL,
                    revenue REAL, revenue_year INTEGER, revenue_month INTEGER,
                    PRIMARY KEY (date, stock_id)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS shares_issued (
                    date TEXT NOT NULL, stock_id TEXT NOT NULL,
                    shares_issued REAL,
                    PRIMARY KEY (date, stock_id)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS us_prices (
                    date TEXT NOT NULL, stock_id TEXT NOT NULL, industry TEXT,
                    open REAL, high REAL, low REAL, close REAL,
                    volume INTEGER, turnover_value REAL,
                    PRIMARY KEY (date, stock_id)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS us_earnings (
                    date TEXT NOT NULL, stock_id TEXT NOT NULL,
                    eps_estimate REAL, eps_actual REAL, surprise_pct REAL,
                    PRIMARY KEY (date, stock_id)
                )
                """
            )
            self._migrate_us_index_membership(conn)

    def _migrate_us_index_membership(self, conn: sqlite3.Connection) -> None:
        """`us_index_membership` 原本是「一檔股票一列、只存加入日期」
        （PK 是 stock_id），只夠表示「這檔股票現在還在指數裡、哪天加入的」。
        補回被剔除股票的歷史資料需要同一檔股票可能有多段區間（加入/剔除
        日期都要存，中途被剔除又重新加入的話會有不只一段），改成
        `(stock_id, start_date, end_date)`、PK 是 `(stock_id, start_date)`。

        正式環境（Neon Postgres）已經有舊表格跑了一段時間，不能單純
        `CREATE TABLE IF NOT EXISTS` 了事——那樣舊表格會原封不動留著、
        新程式碼卻預期新欄位名稱，直接炸掉。這裡偵測舊表格存不存在、
        有沒有 `start_date` 欄位，沒有的話才搬資料、換表格；已經是新結構
        就直接 no-op，可以每次啟動都安全重跑。
        """
        cols = [row[1] for row in conn.execute("PRAGMA table_info(us_index_membership)").fetchall()]
        if cols and "start_date" not in cols:
            conn.execute("ALTER TABLE us_index_membership RENAME TO us_index_membership_old")
            conn.execute(
                """
                CREATE TABLE us_index_membership (
                    stock_id TEXT NOT NULL, start_date TEXT NOT NULL, end_date TEXT,
                    PRIMARY KEY (stock_id, start_date)
                )
                """
            )
            conn.execute(
                "INSERT INTO us_index_membership (stock_id, start_date, end_date) "
                "SELECT stock_id, date_added, NULL FROM us_index_membership_old WHERE date_added IS NOT NULL"
            )
            conn.execute("DROP TABLE us_index_membership_old")
        else:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS us_index_membership (
                    stock_id TEXT NOT NULL, start_date TEXT NOT NULL, end_date TEXT,
                    PRIMARY KEY (stock_id, start_date)
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

    def upsert_month_revenue(self, df: pd.DataFrame) -> None:
        if df.empty:
            return
        d = _normalize_dates(df)
        rows = list(d[MONTH_REVENUE_COLS].itertuples(index=False, name=None))
        with self._connect() as conn:
            conn.executemany(
                "INSERT OR REPLACE INTO month_revenue "
                "(date, stock_id, revenue, revenue_year, revenue_month) VALUES (?,?,?,?,?)",
                rows,
            )

    def load_month_revenue(self, start_date=None, end_date=None, stock_ids=None) -> pd.DataFrame:
        return self._load("month_revenue", MONTH_REVENUE_COLS, start_date, end_date, stock_ids)

    def upsert_shares_issued(self, df: pd.DataFrame) -> None:
        if df.empty:
            return
        d = _normalize_dates(df)
        rows = list(d[SHARES_ISSUED_COLS].itertuples(index=False, name=None))
        with self._connect() as conn:
            conn.executemany(
                "INSERT OR REPLACE INTO shares_issued (date, stock_id, shares_issued) VALUES (?,?,?)",
                rows,
            )

    def load_shares_issued(self, start_date=None, end_date=None, stock_ids=None) -> pd.DataFrame:
        return self._load("shares_issued", SHARES_ISSUED_COLS, start_date, end_date, stock_ids)

    def upsert_us_prices(self, df: pd.DataFrame) -> None:
        if df.empty:
            return
        d = _normalize_dates(df)
        rows = list(d[PRICE_COLS].itertuples(index=False, name=None))
        with self._connect() as conn:
            conn.executemany(
                "INSERT OR REPLACE INTO us_prices "
                "(date, stock_id, industry, open, high, low, close, volume, turnover_value) "
                "VALUES (?,?,?,?,?,?,?,?,?)",
                rows,
            )

    def load_us_prices(self, start_date=None, end_date=None, stock_ids=None) -> pd.DataFrame:
        return self._load("us_prices", PRICE_COLS, start_date, end_date, stock_ids)

    def upsert_us_index_membership(self, df: pd.DataFrame) -> None:
        if df.empty:
            return
        rows = _membership_rows(df)
        with self._connect() as conn:
            conn.executemany(
                "INSERT OR REPLACE INTO us_index_membership (stock_id, start_date, end_date) VALUES (?,?,?)", rows
            )

    def load_us_index_membership(self) -> pd.DataFrame:
        with self._connect() as conn:
            df = pd.read_sql_query(
                "SELECT * FROM us_index_membership", conn, parse_dates=["start_date", "end_date"]
            )
        if df.empty:
            return pd.DataFrame(columns=US_INDEX_MEMBERSHIP_COLS)
        return df[US_INDEX_MEMBERSHIP_COLS]

    def upsert_us_earnings(self, df: pd.DataFrame) -> None:
        if df.empty:
            return
        d = _normalize_dates(df)
        rows = list(d[US_EARNINGS_COLS].itertuples(index=False, name=None))
        with self._connect() as conn:
            conn.executemany(
                "INSERT OR REPLACE INTO us_earnings "
                "(date, stock_id, eps_estimate, eps_actual, surprise_pct) VALUES (?,?,?,?,?)",
                rows,
            )

    def load_us_earnings(self, start_date=None, end_date=None, stock_ids=None) -> pd.DataFrame:
        return self._load("us_earnings", US_EARNINGS_COLS, start_date, end_date, stock_ids)

    def latest_date(self, table: str, stock_id: str | None = None) -> pd.Timestamp | None:
        query = f"SELECT MAX(date) FROM {table}"
        params: tuple = ()
        if stock_id is not None:
            query += " WHERE stock_id = ?"
            params = (stock_id,)
        with self._connect() as conn:
            row = conn.execute(query, params).fetchone()
        return pd.Timestamp(row[0]) if row and row[0] else None

    def close(self) -> None:
        """每次呼叫都各自開關連線，這裡是 no-op，純粹跟 PostgresDataStore 對稱。"""


class PostgresDataStore(DataStore):
    """雲端 Postgres 版本。需要 `pip install psycopg2-binary`（見 pyproject.toml
    的 `cloud` extra），延遲匯入以免沒用到雲端資料庫的人也被迫安裝它。
    """

    def __init__(self, dsn: str, connect_timeout: int = 10, statement_timeout_ms: int = 30_000):
        import psycopg2  # noqa: PLC0415
        import psycopg2.extras  # noqa: PLC0415

        self._psycopg2 = psycopg2
        self._execute_values = psycopg2.extras.execute_values
        self.dsn = dsn
        self.connect_timeout = connect_timeout
        self.statement_timeout_ms = statement_timeout_ms
        self._conn = None
        self._init_schema()

    def _connect(self):
        # 重用同一條連線（psycopg2 的 `with conn:` 只管交易 commit/rollback，
        # 不會關閉連線，所以整個 store 生命週期內可以安全重複用同一條）。
        # connect_timeout 一定要帶：沒有它，遇到連線異常 psycopg2 會無限期
        # 卡住而不是拋出清楚的錯誤（實測 GitHub Actions 上跑 ingest 卡了
        # 8 分鐘以上，就是這裡沒設 timeout 的直接後果）。
        #
        # statement_timeout 不能透過 connect() 的 options 參數在 startup
        # packet 裡帶（實測對 Neon 的 pooler endpoint 這樣做會被直接拒絕：
        # "unsupported startup parameter in options: statement_timeout.
        # Please use unpooled connection or remove this parameter" ——
        # PgBouncer 交易池接模式不轉發任意 GUC），改成連線建立後另外執行
        # SET 指令，兩種 pooler/非 pooler 連線都吃得下。
        if self._conn is None or self._conn.closed:
            self._conn = self._psycopg2.connect(self.dsn, connect_timeout=self.connect_timeout)
            with self._conn.cursor() as cur:
                cur.execute(f"SET statement_timeout = {int(self.statement_timeout_ms)}")
            self._conn.commit()
        return self._conn

    def close(self) -> None:
        if self._conn is not None and not self._conn.closed:
            self._conn.close()

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
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS month_revenue (
                    date DATE NOT NULL, stock_id TEXT NOT NULL,
                    revenue DOUBLE PRECISION, revenue_year INTEGER, revenue_month INTEGER,
                    PRIMARY KEY (date, stock_id)
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS shares_issued (
                    date DATE NOT NULL, stock_id TEXT NOT NULL,
                    shares_issued DOUBLE PRECISION,
                    PRIMARY KEY (date, stock_id)
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS us_prices (
                    date DATE NOT NULL, stock_id TEXT NOT NULL, industry TEXT,
                    open DOUBLE PRECISION, high DOUBLE PRECISION, low DOUBLE PRECISION,
                    close DOUBLE PRECISION, volume BIGINT, turnover_value DOUBLE PRECISION,
                    PRIMARY KEY (date, stock_id)
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS us_earnings (
                    date DATE NOT NULL, stock_id TEXT NOT NULL,
                    eps_estimate DOUBLE PRECISION, eps_actual DOUBLE PRECISION, surprise_pct DOUBLE PRECISION,
                    PRIMARY KEY (date, stock_id)
                )
                """
            )
            self._migrate_us_index_membership(cur)
            conn.commit()

    def _migrate_us_index_membership(self, cur) -> None:
        """見 SQLiteDataStore._migrate_us_index_membership 的完整說明：把舊的
        「一檔股票一列、PK 是 stock_id、只存加入日期」表格，遷移成
        `(stock_id, start_date, end_date)`、PK 是 `(stock_id, start_date)`
        的多區間表格，才能表示被剔除股票的剔除日期、以及中途被剔除又
        重新加入的多段區間。跟 SQLite 版一樣偵測欄位存不存在，冪等、
        可以每次啟動都安全重跑，不會对已經是新結構的表格重複搬資料。
        """
        cur.execute("SELECT column_name FROM information_schema.columns WHERE table_name = 'us_index_membership'")
        cols = [row[0] for row in cur.fetchall()]
        if cols and "start_date" not in cols:
            cur.execute("ALTER TABLE us_index_membership RENAME TO us_index_membership_old")
            cur.execute(
                """
                CREATE TABLE us_index_membership (
                    stock_id TEXT NOT NULL, start_date DATE NOT NULL, end_date DATE,
                    PRIMARY KEY (stock_id, start_date)
                )
                """
            )
            cur.execute(
                "INSERT INTO us_index_membership (stock_id, start_date, end_date) "
                "SELECT stock_id, date_added, NULL FROM us_index_membership_old WHERE date_added IS NOT NULL"
            )
            cur.execute("DROP TABLE us_index_membership_old")
        else:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS us_index_membership (
                    stock_id TEXT NOT NULL, start_date DATE NOT NULL, end_date DATE,
                    PRIMARY KEY (stock_id, start_date)
                )
                """
            )

    def upsert_prices(self, df: pd.DataFrame) -> None:
        if df.empty:
            return
        d = _normalize_dates(df)
        rows = list(d[PRICE_COLS].itertuples(index=False, name=None))
        with self._connect() as conn, conn.cursor() as cur:
            # execute_values 把整批資料打包成一條多列 INSERT，而不是像
            # cursor.executemany() 那樣每一列各自送一次網路往返——實測用
            # executemany 全量回填一檔股票 3 年資料（~729 列）要將近 6 分鐘，
            # 換成 execute_values 後同樣的資料量應該是幾百毫秒等級，這才是
            # 之前每次全量回填 workflow 卡到超過 15 分鐘上限的真正原因
            # （不是連線卡住、不是速率限制，是逐列寫入太慢）。
            self._execute_values(
                cur,
                """
                INSERT INTO prices (date, stock_id, industry, open, high, low, close, volume, turnover_value)
                VALUES %s
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
            self._execute_values(
                cur,
                """
                INSERT INTO margin_short (date, stock_id, margin_purchase_balance, short_balance)
                VALUES %s
                ON CONFLICT (date, stock_id) DO UPDATE SET
                    margin_purchase_balance = EXCLUDED.margin_purchase_balance,
                    short_balance = EXCLUDED.short_balance
                """,
                rows,
            )
            conn.commit()

    def upsert_month_revenue(self, df: pd.DataFrame) -> None:
        if df.empty:
            return
        d = _normalize_dates(df)
        rows = list(d[MONTH_REVENUE_COLS].itertuples(index=False, name=None))
        with self._connect() as conn, conn.cursor() as cur:
            self._execute_values(
                cur,
                """
                INSERT INTO month_revenue (date, stock_id, revenue, revenue_year, revenue_month)
                VALUES %s
                ON CONFLICT (date, stock_id) DO UPDATE SET
                    revenue = EXCLUDED.revenue, revenue_year = EXCLUDED.revenue_year,
                    revenue_month = EXCLUDED.revenue_month
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

    def load_month_revenue(self, start_date=None, end_date=None, stock_ids=None) -> pd.DataFrame:
        return self._load("month_revenue", MONTH_REVENUE_COLS, start_date, end_date, stock_ids)

    def upsert_shares_issued(self, df: pd.DataFrame) -> None:
        if df.empty:
            return
        d = _normalize_dates(df)
        rows = list(d[SHARES_ISSUED_COLS].itertuples(index=False, name=None))
        with self._connect() as conn, conn.cursor() as cur:
            self._execute_values(
                cur,
                """
                INSERT INTO shares_issued (date, stock_id, shares_issued)
                VALUES %s
                ON CONFLICT (date, stock_id) DO UPDATE SET shares_issued = EXCLUDED.shares_issued
                """,
                rows,
            )
            conn.commit()

    def load_shares_issued(self, start_date=None, end_date=None, stock_ids=None) -> pd.DataFrame:
        return self._load("shares_issued", SHARES_ISSUED_COLS, start_date, end_date, stock_ids)

    def upsert_us_prices(self, df: pd.DataFrame) -> None:
        if df.empty:
            return
        d = _normalize_dates(df)
        rows = list(d[PRICE_COLS].itertuples(index=False, name=None))
        with self._connect() as conn, conn.cursor() as cur:
            self._execute_values(
                cur,
                """
                INSERT INTO us_prices (date, stock_id, industry, open, high, low, close, volume, turnover_value)
                VALUES %s
                ON CONFLICT (date, stock_id) DO UPDATE SET
                    industry = EXCLUDED.industry, open = EXCLUDED.open, high = EXCLUDED.high,
                    low = EXCLUDED.low, close = EXCLUDED.close, volume = EXCLUDED.volume,
                    turnover_value = EXCLUDED.turnover_value
                """,
                rows,
            )
            conn.commit()

    def load_us_prices(self, start_date=None, end_date=None, stock_ids=None) -> pd.DataFrame:
        return self._load("us_prices", PRICE_COLS, start_date, end_date, stock_ids)

    def upsert_us_index_membership(self, df: pd.DataFrame) -> None:
        if df.empty:
            return
        rows = _membership_rows(df)
        with self._connect() as conn, conn.cursor() as cur:
            self._execute_values(
                cur,
                """
                INSERT INTO us_index_membership (stock_id, start_date, end_date)
                VALUES %s
                ON CONFLICT (stock_id, start_date) DO UPDATE SET end_date = EXCLUDED.end_date
                """,
                rows,
            )
            conn.commit()

    def load_us_index_membership(self) -> pd.DataFrame:
        with self._connect() as conn:
            df = pd.read_sql_query(
                "SELECT * FROM us_index_membership", conn, parse_dates=["start_date", "end_date"]
            )
        if df.empty:
            return pd.DataFrame(columns=US_INDEX_MEMBERSHIP_COLS)
        return df[US_INDEX_MEMBERSHIP_COLS]

    def upsert_us_earnings(self, df: pd.DataFrame) -> None:
        if df.empty:
            return
        d = _normalize_dates(df)
        rows = list(d[US_EARNINGS_COLS].itertuples(index=False, name=None))
        with self._connect() as conn, conn.cursor() as cur:
            self._execute_values(
                cur,
                """
                INSERT INTO us_earnings (date, stock_id, eps_estimate, eps_actual, surprise_pct)
                VALUES %s
                ON CONFLICT (date, stock_id) DO UPDATE SET
                    eps_estimate = EXCLUDED.eps_estimate, eps_actual = EXCLUDED.eps_actual,
                    surprise_pct = EXCLUDED.surprise_pct
                """,
                rows,
            )
            conn.commit()

    def load_us_earnings(self, start_date=None, end_date=None, stock_ids=None) -> pd.DataFrame:
        return self._load("us_earnings", US_EARNINGS_COLS, start_date, end_date, stock_ids)

    def latest_date(self, table: str, stock_id: str | None = None) -> pd.Timestamp | None:
        query = f"SELECT MAX(date) FROM {table}"
        params: tuple = ()
        if stock_id is not None:
            query += " WHERE stock_id = %s"
            params = (stock_id,)
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(query, params)
            row = cur.fetchone()
        return pd.Timestamp(row[0]) if row and row[0] else None


def get_data_store() -> DataStore:
    """依環境變數決定要用雲端 Postgres 還是本機 SQLite，全系統只有這裡需要判斷。"""
    dsn = os.environ.get("DATABASE_URL")
    if dsn:
        return PostgresDataStore(dsn)
    return SQLiteDataStore(os.environ.get("SQLITE_DB_PATH", "data/tw_market.db"))
