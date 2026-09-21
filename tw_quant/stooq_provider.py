"""Stooq 歷史股價介接——備用資料源，用來測試 yfinance 完全查無資料的
2008 年代被剔除 S&P 500 成分股（雷曼兄弟、Bear Stearns、Washington
Mutual 等），Stooq 是否有更完整的收錄。

背景：2026-09-21，yfinance 的兩輪嘗試（指定日期窗口 + period="max"
全歷史重試）加總只從 364 檔缺漏股票裡補回 54 檔，剩下 318 檔（含全部
2008 危機主角）完全查無資料或資料窗口對不上。Stooq（stooq.com）是另一個
免費、不需 API key 的歷史股價來源，資料整合自多個交易所/資料商，社群
經驗上對已下市證券的收錄有時候比 yfinance 更完整——但這只是假設，需要
實測才知道真的有沒有用，見 scripts/probe_stooq_missing_sp500_stocks.py。

這裡只做「探路」這一層（不連網也能單元測試的 CSV 解析部分 + 需要連網的
抓取部分分開），跟 tw_quant/us_data_provider.py 的既有慣例一致：先測
覆蓋率，確認有實質幫助後才考慮正式整合進 us_prices 資料表。

跟 yfinance 一樣，這個開發沙盒連不了外部網站，串接前務必先以小範圍資料
驗證欄位對應（Stooq 偶爾會調整回傳格式，且免費端點沒有官方文件保證
穩定）。
"""

from __future__ import annotations

import io

import pandas as pd

from tw_quant.us_data_provider import US_PRICE_COLUMNS

STOOQ_CSV_URL = "https://stooq.com/q/d/l/?s={ticker}&i=d"


def normalize_stooq_ticker(ticker: str) -> str:
    """S&P 500/yfinance 慣用的代號格式（例如 BRK-B、LEHMQ）轉成 Stooq
    的美股代號格式——小寫、加上 .us 後綴。Stooq 的連字號/特殊代號處理
    方式沒有官方文件，這裡先用最基本的轉換規則（照抄 ticker、只轉小寫），
    實際能不能查到要靠 probe 腳本的實測結果驗證，查不到不代表轉換規則
    本身有問題，也可能是 Stooq 真的沒收錄這檔。
    """
    return f"{ticker.lower()}.us"


def parse_stooq_csv(csv_text: str, stock_id: str, industry: str = "") -> pd.DataFrame:
    """把 Stooq CSV 端點回傳的原始文字解析成跟 us_prices 資料表一致的
    欄位（US_PRICE_COLUMNS）。Stooq 查無資料時回傳的內容是純文字
    "No data"（不是合法 CSV），這裡明確判斷這個情況回傳空表，而不是讓
    pandas.read_csv 拋出難以理解的解析錯誤。

    Stooq CSV 欄位是 Date,Open,High,Low,Close,Volume（沒有台股/yfinance
    版本裡的成交金額，這裡用 close*volume 概算 turnover_value，跟
    tw_quant.us_data_provider.parse_yfinance_history 對成交金額缺欄位
    情況的處理方式一致，僅供參考、不是真正的逐筆成交金額）。
    """
    if not csv_text or csv_text.strip().lower().startswith("no data"):
        return pd.DataFrame(columns=US_PRICE_COLUMNS)

    df = pd.read_csv(io.StringIO(csv_text))
    if df.empty or "Date" not in df.columns:
        return pd.DataFrame(columns=US_PRICE_COLUMNS)

    df = df.rename(columns={"Date": "date", "Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"})
    df["date"] = pd.to_datetime(df["date"])
    df["stock_id"] = stock_id
    df["industry"] = industry
    df["turnover_value"] = df["close"] * df["volume"]
    return df[US_PRICE_COLUMNS].sort_values("date").reset_index(drop=True)


def fetch_stooq_price(stock_id: str, industry: str = "", timeout: float = 30.0) -> pd.DataFrame:
    """對單一股票代號呼叫 Stooq CSV 端點，回傳解析過的價量資料。跟
    tw_quant/us_data_provider.py 的 fetch_sp500_constituents 一樣自己帶
    User-Agent（部分網站對沒有 User-Agent 的請求會回應異常內容）。
    """
    import requests

    ticker = normalize_stooq_ticker(stock_id)
    url = STOOQ_CSV_URL.format(ticker=ticker)
    headers = {"User-Agent": "Mozilla/5.0 (compatible; research-bot/1.0)"}
    resp = requests.get(url, headers=headers, timeout=timeout)
    resp.raise_for_status()
    return parse_stooq_csv(resp.text, stock_id, industry)
