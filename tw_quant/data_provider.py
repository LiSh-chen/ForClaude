"""資料層：定義本系統要求的資料 schema，並提供

1. CSV / Parquet 落地檔案的讀取與欄位驗證
2. 合成假資料產生器，供開發環境沒有市場資料來源時，仍可跑通全流程、寫測試
3. FinMind 開放資料 API 的介接骨架（未在本沙盒環境下實測，見下方說明）

★★★ 重要限制 ★★★
本開發環境的對外網路走白名單代理，無法連線 FinMind / TWSE OpenAPI / 券商 API
等台股資料來源（已實測 api.finmindtrade.com 連線被代理拒絕）。因此本模組
提供的是「正確的資料介接骨架 + schema 規範」，真正落地仍需要使用者在有網路
權限的環境中：
  (a) 自行申請 FinMind token 並啟用 FinMindPriceProvider / FinMindMarginShortProvider，或
  (b) 使用 TEJ / 券商 API（如永豐 Shioaji）匯出歷史資料存成本模組要求的 CSV schema，或
  (c) 直接呼叫 load_price_csv / load_margin_short_csv 讀入既有資料。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

PRICE_COLUMNS = [
    "date",
    "stock_id",
    "industry",
    "open",
    "high",
    "low",
    "close",
    "volume",  # 股數
    "turnover_value",  # 成交金額（元），流動性排序 tie-breaker 用
]

MARGIN_SHORT_COLUMNS = [
    "date",
    "stock_id",
    "margin_purchase_balance",  # 融資餘額（張）
    "short_balance",  # 融券餘額（張）
]


class SchemaError(ValueError):
    pass


def _validate_columns(df: pd.DataFrame, required: list[str], name: str) -> None:
    missing = set(required) - set(df.columns)
    if missing:
        raise SchemaError(f"{name} 缺少必要欄位: {sorted(missing)}")


def load_price_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    _validate_columns(df, PRICE_COLUMNS, "price panel")
    df = df.sort_values(["stock_id", "date"]).reset_index(drop=True)
    return df


def load_margin_short_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    _validate_columns(df, MARGIN_SHORT_COLUMNS, "margin/short panel")
    df = df.sort_values(["stock_id", "date"]).reset_index(drop=True)
    return df


def compute_margin_short_ratio(margin_short_df: pd.DataFrame) -> pd.Series:
    """券資比 = 融券餘額 / 融資餘額。融資餘額為 0 時視為缺值（避免除以 0 產生 inf）。"""
    denom = margin_short_df["margin_purchase_balance"].replace(0, np.nan)
    return margin_short_df["short_balance"] / denom


@dataclass
class SyntheticUniverseConfig:
    n_stocks: int = 60
    n_days: int = 900  # 約 3.5 個交易年，足以跑 WFA 3+1 年窗
    n_industries: int = 6
    start_date: str = "2019-01-02"
    seed: int = 42
    new_listing_stock_ids: tuple[str, ...] = ()  # 這些股票故意晚上市，測試 min_history_days 排除


def generate_synthetic_universe(cfg: SyntheticUniverseConfig | None = None) -> dict[str, pd.DataFrame]:
    """產生 GBM 隨機漫步 + 週期性動能事件的多股假資料，僅供開發/測試流程使用，
    不代表任何真實台股標的之價格行為，嚴禁用於實際績效評估或投資決策。
    """
    cfg = cfg or SyntheticUniverseConfig()
    rng = np.random.default_rng(cfg.seed)

    dates = pd.bdate_range(cfg.start_date, periods=cfg.n_days)
    stock_ids = [f"S{1000 + i}" for i in range(cfg.n_stocks)]
    industries = [f"IND{i % cfg.n_industries}" for i in range(cfg.n_stocks)]

    price_rows = []
    margin_rows = []

    for si, (stock_id, industry) in enumerate(zip(stock_ids, industries)):
        local_rng = np.random.default_rng(cfg.seed + si)
        n = cfg.n_days
        start_idx = 0
        if stock_id in cfg.new_listing_stock_ids:
            start_idx = n - 150  # 只給 150 天資料，模擬新股

        drift = local_rng.normal(0.0003, 0.0002)
        vol = local_rng.uniform(0.012, 0.03)
        # 動能事件：隨機挑幾段時間給予正向漂移+放量，製造可被策略偵測到的突破
        momentum_mask = np.zeros(n)
        n_events = local_rng.integers(2, 5)
        for _ in range(n_events):
            evt_start = local_rng.integers(260, n - 30)
            evt_len = local_rng.integers(10, 25)
            momentum_mask[evt_start : evt_start + evt_len] = 1.0

        returns = local_rng.normal(drift, vol, n) + momentum_mask * local_rng.uniform(0.01, 0.03, n)
        price = 20 * np.exp(np.cumsum(returns))

        close = price
        high = close * (1 + np.abs(local_rng.normal(0, 0.006, n)))
        low = close * (1 - np.abs(local_rng.normal(0, 0.006, n)))
        open_ = low + (high - low) * local_rng.uniform(0.2, 0.8, n)

        base_vol_lots = local_rng.uniform(1500, 6000)
        volume_lots = base_vol_lots * (1 + local_rng.normal(0, 0.3, n).clip(-0.8, 3))
        volume_lots = volume_lots * (1 + momentum_mask * local_rng.uniform(0.8, 2.0, n))
        volume_lots = np.clip(volume_lots, 50, None)
        volume_shares = (volume_lots * 1000).astype(int)
        turnover_value = volume_shares * close

        margin_base = local_rng.uniform(500, 3000)
        margin_bal = np.clip(margin_base * (1 + np.abs(local_rng.normal(0, 0.2, n))), 10, None)
        # 軋空事件：在動能事件末端拉高券資比（融券回補前的異常堆積）
        short_bal = margin_bal * local_rng.uniform(0.05, 0.15, n)
        short_bal = short_bal + momentum_mask * margin_bal * local_rng.uniform(0.3, 0.9, n)

        for i in range(start_idx, n):
            price_rows.append(
                (dates[i], stock_id, industry, open_[i], high[i], low[i], close[i], volume_shares[i], turnover_value[i])
            )
            margin_rows.append((dates[i], stock_id, margin_bal[i], short_bal[i]))

    price_df = pd.DataFrame(price_rows, columns=PRICE_COLUMNS)
    margin_df = pd.DataFrame(margin_rows, columns=MARGIN_SHORT_COLUMNS)
    return {"prices": price_df, "margin_short": margin_df}


# ---------------------------------------------------------------------------
# FinMind 介接骨架（未在此沙盒環境測試，需使用者在有網路權限環境下驗證）
# ---------------------------------------------------------------------------


class FinMindDataProvider:
    """FinMind 開放資料 API 介接骨架。

    使用方式（需自行 `pip install requests` 並取得 FinMind token）：

        provider = FinMindDataProvider(token="你的token")
        prices = provider.fetch_price("2330", "2019-01-01", "2024-12-31")

    本類別在本次開發沙盒中無法連上外網驗證，僅依 FinMind 官方文件公開的
    dataset 名稱與參數格式撰寫，串接前請務必先以小範圍資料驗證欄位對應。
    """

    BASE_URL = "https://api.finmindtrade.com/api/v4/data"

    def __init__(self, token: str | None = None):
        self.token = token

    def _get(self, dataset: str, data_id: str, start_date: str, end_date: str) -> pd.DataFrame:
        import requests  # 延遲匯入，避免無此套件時整個模組載入失敗

        params = {
            "dataset": dataset,
            "data_id": data_id,
            "start_date": start_date,
            "end_date": end_date,
        }
        if self.token:
            params["token"] = self.token
        resp = requests.get(self.BASE_URL, params=params, timeout=30)
        resp.raise_for_status()
        payload = resp.json()
        return pd.DataFrame(payload.get("data", []))

    def fetch_price(self, stock_id: str, start_date: str, end_date: str) -> pd.DataFrame:
        raw = self._get("TaiwanStockPrice", stock_id, start_date, end_date)
        if raw.empty:
            return raw
        df = raw.rename(
            columns={
                "Trading_Volume": "volume",
                "Trading_money": "turnover_value",
                "open": "open",
                "max": "high",
                "min": "low",
                "close": "close",
            }
        )
        df["date"] = pd.to_datetime(df["date"])
        df["stock_id"] = stock_id
        return df[[c for c in PRICE_COLUMNS if c in df.columns] + ["date", "stock_id"]].drop_duplicates()

    def fetch_margin_short(self, stock_id: str, start_date: str, end_date: str) -> pd.DataFrame:
        raw = self._get("TaiwanStockMarginPurchaseShortSale", stock_id, start_date, end_date)
        if raw.empty:
            return raw
        df = raw.rename(
            columns={
                "MarginPurchaseTodayBalance": "margin_purchase_balance",
                "ShortSaleTodayBalance": "short_balance",
            }
        )
        df["date"] = pd.to_datetime(df["date"])
        df["stock_id"] = stock_id
        return df[MARGIN_SHORT_COLUMNS]
