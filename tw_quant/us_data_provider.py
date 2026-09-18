"""美股資料介接（yfinance）。

跟 tw_quant/data_provider.py 的 FinMindDataProvider 對應，但抓的是美股
（S&P 500 成分股）而不是台股。刻意沿用跟台股完全一樣的欄位命名
（`stock_id`、`industry`），即使語意上這裡存的其實是美股代號（ticker）
跟 GICS 產業分類（sector）——這樣 `tw_quant/backtest.py`、
`tw_quant/indicators.py`、`tw_quant/factor_backtest.py` 這些泛用回測引擎
未來要重複使用在美股上時，不需要為了欄位名稱不同而另外改寫，只要注意
台股特有的規則（交易稅率、lot_size=1000張的整數股限制、tick size 跳動
表）在真的要對美股回測前需要換一套美股版的成本模型（現在還沒做，這裡
純粹是資料層，還不是回測層）。

yfinance 不需要 API token，免費、沒有 FinMind 那種「當日額度用完就
402」的限制，但仍然要對 Yahoo Finance 客氣一點（呼叫之間加 sleep），
避免被暫時性地限速/封鎖。

跟 FinMindDataProvider 一樣，這裡沒辦法在這個開發沙盒裡連網驗證，串接前
務必先以小範圍資料驗證欄位對應（Yahoo 偶爾會調整回傳格式）。
"""

from __future__ import annotations

import pandas as pd

US_PRICE_COLUMNS = ["date", "stock_id", "industry", "open", "high", "low", "close", "volume", "turnover_value"]


def normalize_yfinance_ticker(ticker: str) -> str:
    """S&P 500 成分股清單裡的代號用句點分隔股份等級（例如 BRK.B），
    但 yfinance/Yahoo Finance 的代號格式是用連字號（BRK-B），呼叫 API
    前要轉換，不然查不到資料。
    """
    return ticker.replace(".", "-")


def parse_sp500_wikipedia_table(raw_table: pd.DataFrame) -> pd.DataFrame:
    """把從維基百科「List of S&P 500 companies」頁面第一張表格抓下來的
    原始 DataFrame，轉成 (stock_id, name, industry) 三欄，跟這個 provider
    其他方法的欄位命名慣例一致。獨立成一個不需要網路的純函式，方便測試
    「欄位改名」「代號格式轉換」這些邏輯本身對不對，不用真的連網。
    """
    df = raw_table.rename(columns={"Symbol": "stock_id", "Security": "name", "GICS Sector": "industry"})
    df["stock_id"] = df["stock_id"].astype(str).map(normalize_yfinance_ticker)
    return df[["stock_id", "name", "industry"]].drop_duplicates(subset=["stock_id"])


def parse_yfinance_history(history: pd.DataFrame, stock_id: str, industry: str) -> pd.DataFrame:
    """把 yf.Ticker(...).history() 回傳的原始 DataFrame（DatetimeIndex +
    Open/High/Low/Close/Volume 欄位）轉成跟 tw_quant.storage.PRICE_COLS
    一樣的長格式欄位。同樣獨立成純函式方便測試，不用真的連網。
    """
    if history.empty:
        return pd.DataFrame(columns=US_PRICE_COLUMNS)
    df = history.reset_index().rename(
        columns={"Date": "date", "Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"}
    )
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    df["stock_id"] = stock_id
    df["industry"] = industry
    # 美股資料沒有現成的「成交金額」欄位，用收盤價*成交量做近似值（TW 的
    # turnover_value 是交易所直接提供的真實成交金額，這裡只是估計）
    df["turnover_value"] = df["close"] * df["volume"]
    return df[US_PRICE_COLUMNS]


class YFinanceUSDataProvider:
    """薄薄一層包住 yfinance，介面盡量比照 FinMindDataProvider（同樣是
    「一次一檔股票、一個日期區間」的呼叫方式），讓 ingest 腳本可以重用
    同一套逐股回填/增量同步邏輯。
    """

    def fetch_sp500_constituents(self) -> pd.DataFrame:
        """從維基百科抓 S&P 500 成分股清單（ticker/公司名/GICS產業分類）。
        這不是官方 API，維基百科頁面格式偶爾會變動，串接後要驗證欄位對應。
        """
        import requests

        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        # 直接用 requests 抓再交給 pandas.read_html，而不是讓 read_html 自己發
        # 請求：維基百科會擋掉沒有 User-Agent 的請求（回 403），這裡自己帶一個
        headers = {"User-Agent": "Mozilla/5.0 (compatible; research-bot/1.0)"}
        resp = requests.get(url, headers=headers, timeout=30)
        resp.raise_for_status()
        tables = pd.read_html(resp.text)
        return parse_sp500_wikipedia_table(tables[0])

    def fetch_price(self, stock_id: str, start_date: str, end_date: str, industry: str = "") -> pd.DataFrame:
        import yfinance as yf

        ticker = yf.Ticker(stock_id)
        history = ticker.history(start=start_date, end=end_date, auto_adjust=True)
        return parse_yfinance_history(history, stock_id, industry)
