"""抽樣測試 yfinance 能不能抓到「已經被踢出 S&P 500 指數、資料庫裡完全沒有
資料」的股票的歷史股價——這是 scripts/list_sp500_removed_stocks_from_db.py
列出 177 檔缺漏名單之後的下一步，先老實測試可行性，再決定要不要真的動手
把這些股票補進資料庫。

背景：yfinance 底層是 Yahoo Finance 的公開資料，對「已下市」股票的支援
不一致——單純被踢出指數但公司還在正常上市交易的股票（例如指數調整、
市值排名掉出去)理論上完全沒問題；但公司被收購下市、破產下市、或改名換
代號的情況，Yahoo Finance 可能保留歷史資料到下市那天為止、也可能完全查
不到、也可能把舊代號的頁面導向新公司（代號回收），這些都要實測才知道，
不能假設。

刻意挑 8 檔涵蓋不同情境的股票（見 SAMPLE_TICKERS 的註解），日期區間抓
「resurr resurrected 名單裡記錄的加入~剔除區間，前後各留 30 天緩衝」，這樣
如果 yfinance 真的有資料，應該看得到區間開頭到接近區間尾端都有數據；如果
只到區間中途就斷掉、或整段查無資料、或資料一路延續到「今天」（代號被
回收給別家公司的訊號），都能從輸出直接判讀出來。

用法：
    python scripts/test_yfinance_delisted_tickers.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tw_quant.us_data_provider import YFinanceUSDataProvider

# (ticker, 區間起, 區間迄, 情境說明)——區間取自 list_sp500_removed_stocks_from_db.py
# 實際跑出來的真實剔除區間，前後各留 30 天緩衝
SAMPLE_TICKERS = [
    ("AAL", "2018-08-21", "2024-10-23", "單純被踢出指數，公司仍在那斯達克正常交易"),
    ("AAP", "2018-08-21", "2023-09-24", "單純被踢出指數，公司仍在 NYSE 正常交易"),
    ("HES", "2018-08-21", "2025-08-22", "2025 年被雪佛龍收購下市"),
    ("SIVB", "2018-08-21", "2023-04-14", "2023 年矽谷銀行倒閉，股票直接下市"),
    ("FRC", "2018-12-03", "2023-06-03", "2023 年第一共和銀行倒閉"),
    ("TWTR", "2018-08-21", "2022-12-01", "2022 年被馬斯克收購下市（私有化）"),
    ("FB", "2018-08-21", "2022-07-09", "2022 年改名為 META，公司還在、代號換了"),
    ("CELG", "2018-08-21", "2019-12-21", "2019 年被必治妥施貴寶收購（年代較久）"),
]


def summarize_fetch(ticker: str, df: pd.DataFrame) -> dict:
    """把 fetch_price 回傳的長格式 DataFrame 摘要成一列判讀用的統計。
    獨立成純函式方便測試格式化邏輯本身，不用真的連網。
    """
    if df.empty:
        return {"ticker": ticker, "rows": 0, "first": None, "last": None}
    return {
        "ticker": ticker,
        "rows": len(df),
        "first": df["date"].min().date(),
        "last": df["date"].max().date(),
    }


def _fmt_row(ticker: str, requested_end: str, summary: dict, note: str) -> str:
    if summary["rows"] == 0:
        body = "查無資料"
    else:
        body = f"{summary['rows']:>5} 列，{summary['first']} ~ {summary['last']}"
        if str(summary["last"]) >= requested_end:
            body += "（資料一路延續到請求區間尾端附近——留意代號是否被回收給別家公司）"
    return f"  {ticker:<6} 請求區間迄 {requested_end}：{body}\n         情境：{note}"


def main() -> None:
    provider = YFinanceUSDataProvider()
    print(f"抽樣測試 {len(SAMPLE_TICKERS)} 檔已剔除股票，yfinance 能不能抓到歷史股價：\n")

    for ticker, start, end, note in SAMPLE_TICKERS:
        try:
            df = provider.fetch_price(ticker, start, end)
        except Exception as exc:  # noqa: BLE001 -- 診斷腳本，任何一檔失敗都要看得到原因、不中斷其他檔
            print(f"  {ticker:<6} 請求區間 {start}~{end}：拋出例外 {type(exc).__name__}: {exc}\n         情境：{note}")
            time.sleep(1)
            continue

        summary = summarize_fetch(ticker, df)
        print(_fmt_row(ticker, end, summary, note))
        time.sleep(1)  # 對 Yahoo Finance 客氣一點，避免連續呼叫被暫時限速

    print(
        "\n（誠實揭露：這只是 8 檔的抽樣，不是 177 檔缺漏名單的全面測試；"
        "「查無資料」也可能是 yfinance/Yahoo Finance 當下暫時性問題，不一定代表"
        "永久抓不到，正式決定要不要批次補資料前，這 8 檔的結果只能當初步參考。）"
    )


if __name__ == "__main__":
    main()
