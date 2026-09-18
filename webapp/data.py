"""資料存取層：用 st.cache_data 包 tw_quant.storage 的 load_* 方法。

資料庫本身由 GitHub Actions 排程每日更新，使用者在頁面上調參數重跑回測
不需要每次都重新打資料庫——快取 10 分鐘，同一次瀏覽器工作階段內切換
參數不會重複打資料庫，但也不會久到看不到當天排程更新的新資料。

DATABASE_URL 沒設定時 get_data_store() 會退回本機 SQLite（見
tw_quant/storage.py），可以用來在本機用合成/測試資料驗證這個網頁本身
能不能動，不需要真的連到雲端 Postgres。
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from tw_quant.storage import get_data_store

CACHE_TTL_SECONDS = 600


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner="讀取台股價量資料...")
def load_tw_prices() -> pd.DataFrame:
    return get_data_store().load_prices()


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner="讀取台股融資券資料...")
def load_tw_margin_short() -> pd.DataFrame:
    return get_data_store().load_margin_short()


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner="讀取台股月營收資料...")
def load_tw_month_revenue() -> pd.DataFrame:
    return get_data_store().load_month_revenue()


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner="讀取台股已發行股數資料...")
def load_tw_shares_issued() -> pd.DataFrame:
    return get_data_store().load_shares_issued()


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner="讀取美股價量資料...")
def load_us_prices() -> pd.DataFrame:
    return get_data_store().load_us_prices()


def describe_coverage(df: pd.DataFrame) -> str:
    if df.empty:
        return "沒有資料"
    return f"{df['stock_id'].nunique()} 檔，{df['date'].min().date()} ~ {df['date'].max().date()}"
