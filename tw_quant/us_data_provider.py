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

import io

import pandas as pd

US_PRICE_COLUMNS = ["date", "stock_id", "industry", "open", "high", "low", "close", "volume", "turnover_value"]
US_EARNINGS_COLUMNS = ["date", "stock_id", "eps_estimate", "eps_actual", "surprise_pct"]  # date = 財報公布日


def normalize_yfinance_ticker(ticker: str) -> str:
    """S&P 500 成分股清單裡的代號用句點分隔股份等級（例如 BRK.B），
    但 yfinance/Yahoo Finance 的代號格式是用連字號（BRK-B），呼叫 API
    前要轉換，不然查不到資料。
    """
    return ticker.replace(".", "-")


def parse_sp500_wikipedia_table(raw_table: pd.DataFrame) -> pd.DataFrame:
    """把從維基百科「List of S&P 500 companies」頁面第一張表格抓下來的
    原始 DataFrame，轉成 (stock_id, name, industry, date_added) 四欄，跟
    這個 provider 其他方法的欄位命名慣例一致。獨立成一個不需要網路的純
    函式，方便測試「欄位改名」「代號格式轉換」這些邏輯本身對不對，不用
    真的連網。

    date_added（維基百科原始欄位 "Date added"）是這檔股票被納入 S&P 500
    指數的日期——存起來是為了讓回測能排除「當時還沒加入指數」的股票，
    修正「用今天的成分股名單回填過去歷史」天生帶有的存活者偏差（往未來
    看到還沒加入指數的贏家，見 tw_quant/us_universe.py）。這只解決偏差
    的一半：被踢出指數、已經不在「今天的成分股名單」裡的股票依然完全
    抓不到，這裡沒辦法無中生有。
    """
    df = raw_table.rename(
        columns={"Symbol": "stock_id", "Security": "name", "GICS Sector": "industry", "Date added": "date_added"}
    )
    df["stock_id"] = df["stock_id"].astype(str).map(normalize_yfinance_ticker)
    if "date_added" in df.columns:
        df["date_added"] = pd.to_datetime(df["date_added"], errors="coerce")
    else:
        df["date_added"] = pd.NaT
    return df[["stock_id", "name", "industry", "date_added"]].drop_duplicates(subset=["stock_id"])


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


def _find_col_label(columns, keywords: tuple[str, ...]) -> str | None:
    """在 earnings_dates 的欄位名稱裡，用不分大小寫/空格的關鍵字模糊比對找
    第一個符合的欄位——2026-09-19 用 scripts/probe_us_fundamentals_data.py
    探路時，就是靠這種模糊比對才驗證出 yfinance 1.7.0 實際回傳的欄位可以
    正確對到「預期EPS/實際EPS」，不要求完全相符是刻意的：不同版本/不同
    公司回傳的確切欄位名稱可能不完全一致，模糊比對比較不容易因為 Yahoo
    調整格式就整批解析失敗。
    """
    for label in columns:
        text = str(label).lower().replace(" ", "")
        if all(k in text for k in keywords):
            return label
    return None


def parse_yfinance_earnings_dates(earnings_dates: pd.DataFrame, stock_id: str) -> pd.DataFrame:
    """把 yf.Ticker(...).get_earnings_dates(...) 回傳的原始 DataFrame（列是
    財報公布日、欄位包含類似「EPS Estimate」「Reported EPS」「Surprise(%)」
    的財報公布日期資料）轉成跟 tw_quant.storage.US_EARNINGS_COLS 一樣的
    長格式欄位。

    這裡把 yfinance 自帶的 Surprise(%) 也原樣存起來（surprise_pct 欄位），
    但正式策略訊號（見 scripts/test_us_pead_earnings_drift_from_db.py）
    不直接用它，而是自己用 eps_estimate/eps_actual 重新計算——避免依賴一個
    沒辦法從外部稽核公式定義的第三方欄位。

    2026-09-20 用真實資料跑第一次全量回填（577 檔）時，實測發現約 13%
    的股票（75/577）在同一個 stock_id 底下會出現重複的 date（yfinance
    對少數股票的財報公布日資料本身有重複列，原因不明，可能是估計值
    修正後的重複紀錄），寫進 Postgres 時整批 upsert 用 ON CONFLICT DO
    UPDATE，同一批次裡出現重複主鍵會直接報錯（"ON CONFLICT DO UPDATE
    command cannot affect row a second time"），SQLite 因為是逐列
    INSERT OR REPLACE 沒踩到這個問題，掩蓋了這裡的資料品質瑕疵。這裡
    主動去重：同一天出現多筆時，優先保留欄位比較完整（非空值較多）的
    那一筆，避免整批寫入失敗。
    """
    if earnings_dates is None or earnings_dates.empty:
        return pd.DataFrame(columns=US_EARNINGS_COLUMNS)

    idx = earnings_dates.index
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)

    estimate_col = _find_col_label(earnings_dates.columns, ("estimate", "eps"))
    actual_col = _find_col_label(earnings_dates.columns, ("reported", "eps"))
    surprise_col = _find_col_label(earnings_dates.columns, ("surprise",))

    out = pd.DataFrame({"date": pd.DatetimeIndex(idx), "stock_id": stock_id})
    out["eps_estimate"] = earnings_dates[estimate_col].to_numpy() if estimate_col else float("nan")
    out["eps_actual"] = earnings_dates[actual_col].to_numpy() if actual_col else float("nan")
    out["surprise_pct"] = earnings_dates[surprise_col].to_numpy() if surprise_col else float("nan")

    out["_completeness"] = out[["eps_estimate", "eps_actual", "surprise_pct"]].notna().sum(axis=1)
    out = out.sort_values(["date", "_completeness"], ascending=[True, False])
    out = out.drop_duplicates(subset="date", keep="first").drop(columns="_completeness")
    return out[US_EARNINGS_COLUMNS].reset_index(drop=True)


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
        # pandas >=2.1 把「直接傳字串給 read_html」判定為棄用用法，字串太長
        # 時甚至會被誤判成檔案路徑去 open()，丟出 FileNotFoundError（訊息裡
        # 塞了一截 HTML 內容）。用 io.StringIO 包起來明確告訴 pandas 這是
        # HTML 內容不是路徑。
        tables = pd.read_html(io.StringIO(resp.text))
        return parse_sp500_wikipedia_table(tables[0])

    def fetch_price(self, stock_id: str, start_date: str, end_date: str, industry: str = "") -> pd.DataFrame:
        import yfinance as yf

        ticker = yf.Ticker(stock_id)
        history = ticker.history(start=start_date, end=end_date, auto_adjust=True)
        return parse_yfinance_history(history, stock_id, industry)

    def fetch_price_full_history(self, stock_id: str, industry: str = "") -> pd.DataFrame:
        """跟 fetch_price 一樣，但用 period="max" 而不是指定 start/end——
        2026-09-21 補齊 2008 年代被剔除 S&P 500 成分股的歷史股價時發現，
        部分股票用 history(start=clipped_start, end=clipped_end) 查詢會
        回報「possibly delisted; no timezone found」或「Data doesn't
        exist for startDate=X, endDate=Y」，但 yfinance 對這檔股票本身
        不是完全沒有資料——只是指定的查詢窗口跟 yfinance 內部記錄的實際
        掛牌區間對不上（誤差可能來自 fja05680/sp500 快照的加入/剔除日期
        跟 yfinance 認定的實際交易區間有落差）。用 period="max" 直接要
        「這檔股票 yfinance 有的全部歷史」，呼叫端自己再篩選需要的日期
        範圍，用來跟 fetch_price 的結果交叉比對，篩出「真的查無資料」跟
        「窗口設錯」這兩種不同失敗原因。見
        scripts/retry_missing_sp500_stocks_from_db.py。
        """
        import yfinance as yf

        ticker = yf.Ticker(stock_id)
        history = ticker.history(period="max", auto_adjust=True)
        return parse_yfinance_history(history, stock_id, industry)

    def fetch_earnings_history(self, stock_id: str, limit: int = 80) -> pd.DataFrame:
        """抓某檔股票的財報公布日歷史（含 EPS 預期/實際/驚喜幅度），轉成
        US_EARNINGS_COLUMNS 長格式，供 scripts/ingest_us_earnings_data.py
        寫進資料庫用。跟 fetch_quarterly_fundamentals 不同：這裡回傳的是
        已經清理過、可以直接 upsert 的乾淨格式，fetch_quarterly_fundamentals
        回傳原始 yfinance DataFrame 是給探路腳本看格式用的，不是同一個
        用途。

        limit 預設 80（希望能拿到約 20 年份），實際能拿到多少受 Yahoo
        Finance 保留的歷史深度限制——2026-09-19 用
        scripts/probe_us_fundamentals_data.py 探路 40 檔的實測結果：
        平均約 48.7 筆、大部分回溯到 2013~2014 年（約 10~12 年），少數
        較晚上市/分拆的公司歷史較短。
        """
        import yfinance as yf

        ticker = yf.Ticker(normalize_yfinance_ticker(stock_id))
        earnings_dates = ticker.get_earnings_dates(limit=limit)
        return parse_yfinance_earnings_dates(earnings_dates, stock_id)

    def fetch_quarterly_fundamentals(self, stock_id: str, earnings_limit: int = 40) -> dict:
        """一次性抓某檔股票的季報損益表 + 財報公布日期，供
        scripts/probe_us_fundamentals_data.py 探路用——回傳 yfinance 的原始
        DataFrame，不做欄位對應/清理，因為這一步的目的正是要先看清楚
        yfinance 實際回傳的欄位長怎樣、涵蓋度/歷史深度夠不夠，再決定要不要
        投入正式資料管線（見探路腳本開頭的完整背景說明）。

        earnings_limit：get_earnings_dates 預設只回傳 12 筆（約 3 年），
        探路想知道「最多能拿到多深的歷史」，所以預設調高到 40 筆（約 10 年，
        如果 Yahoo Finance 真的有留這麼久的話）。
        """
        import yfinance as yf

        ticker = yf.Ticker(normalize_yfinance_ticker(stock_id))
        return {
            "quarterly_income_stmt": ticker.quarterly_income_stmt,
            "earnings_dates": ticker.get_earnings_dates(limit=earnings_limit),
        }
