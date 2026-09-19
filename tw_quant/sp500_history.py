"""解析 fja05680/sp500 這個社群維護的 S&P 500 歷史成分股快照，重建「時間點
成分股」區間，補上 tw_quant/us_universe.py 目前解決不了的另一半存活者偏差
（被踢出指數、已經不在「今天的成分股名單」裡的股票）。

背景：維基百科「List of S&P 500 companies」頁面經過完整掃描確認只有兩個
表格——目前成分股名單、GICS 產業分類側邊欄——沒有任何歷史異動紀錄表，這
條路已經走不通（見對話紀錄 2026-09-19 的 debug_sp500_full_table_scan.py
一次性診斷結果）。

fja05680/sp500 這個 GitHub repo（社群維護，資料源自 Andreas Clenow
《Trading Evolved》一書附帶資料，之後靠維護者人工核對維基百科異動 + Google
搜尋確認日期，非官方權威來源，大約每兩個月更新一次）的
`S&P 500 Historical Components & Changes (Updated).csv` 提供的正是這個：
每一列是「某天生效的完整成分股清單」（date, tickers 兩欄，tickers 是逗號
分隔的完整名單字串），下一列出現代表清單有變動（不是每個交易日都有新列，
是「異動時才新增一列」的稀疏格式），一路從 1996-01-02 涵蓋到接近最新（見
實際下載驗證，最新一列是 2026-08-18，離抓取當下只差約一個月）。

這裡只做「解析成結構化資料」這個不連網、可單元測試的部分，不連網、不碰
資料庫——那是 scripts/list_sp500_removed_stocks_from_db.py 的事。
"""

from __future__ import annotations

import io

import pandas as pd

from tw_quant.us_data_provider import normalize_yfinance_ticker

SNAPSHOT_CSV_URL = (
    "https://raw.githubusercontent.com/fja05680/sp500/master/"
    "S%26P%20500%20Historical%20Components%20%26%20Changes%20(Updated).csv"
)


def fetch_snapshot_csv_text() -> str:
    """抓 fja05680/sp500 這個 repo 目前 master 分支上的歷史快照 CSV 原始
    文字內容。跟 us_data_provider.py 的 fetch_sp500_constituents 一樣，這
    不是官方 API，是社群維護的 GitHub repo，檔名、格式都可能在未來變動，
    串接後要留意驗證。獨立成一個函式，方便測試時用假資料替換掉，不用真的
    連網。
    """
    import requests

    resp = requests.get(SNAPSHOT_CSV_URL, timeout=60)
    resp.raise_for_status()
    return resp.text


def parse_snapshot_table(csv_text: str) -> pd.DataFrame:
    """把 fja05680/sp500 那個 CSV 的原始文字內容解析成
    (date, tickers) 兩欄的 DataFrame，tickers 欄是 list[str]（已用
    normalize_yfinance_ticker 轉換成 yfinance 慣用的連字號格式，例如
    BRK.B -> BRK-B），依日期由舊到新排序、日期去重。
    """
    df = pd.read_csv(io.StringIO(csv_text))
    df["date"] = pd.to_datetime(df["date"])
    df["tickers"] = df["tickers"].apply(
        lambda s: sorted({normalize_yfinance_ticker(t.strip()) for t in s.split(",") if t.strip()})
    )
    return df.sort_values("date").drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)


def build_membership_intervals(snapshot: pd.DataFrame) -> pd.DataFrame:
    """把 parse_snapshot_table 的逐列快照，轉成「每檔股票在指數裡的
    連續區間」：(stock_id, start_date, end_date) 三欄，一檔股票如果中途被
    剔除又重新加入，會拆成多列分開的區間。

    區間語意：某檔股票出現在第 i 列快照裡，代表它從 date_i 起、到下一次
    快照生效日（date_{i+1}）之前都在指數裡；如果它一路留到最後一列快照
    （目前資料能看到的最新日期）都還在，該區間的 end_date 是 NaT，代表
    「仍是目前成分股，還沒有已知的剔除日期」——不是真的沒有終點，只是
    這份資料截至抓取當下還沒觀察到剔除。
    """
    dates = snapshot["date"].tolist()
    n = len(dates)

    presence: dict[str, list[int]] = {}
    for i, tickers in enumerate(snapshot["tickers"]):
        for t in tickers:
            presence.setdefault(t, []).append(i)

    rows = []
    for stock_id, idxs in presence.items():
        idxs.sort()
        run_start = idxs[0]
        prev = idxs[0]
        for idx in idxs[1:]:
            if idx == prev + 1:
                prev = idx
                continue
            end_date = dates[prev + 1] if prev + 1 < n else pd.NaT
            rows.append((stock_id, dates[run_start], end_date))
            run_start = idx
            prev = idx
        end_date = dates[prev + 1] if prev + 1 < n else pd.NaT
        rows.append((stock_id, dates[run_start], end_date))

    return pd.DataFrame(rows, columns=["stock_id", "start_date", "end_date"]).sort_values(
        ["stock_id", "start_date"]
    ).reset_index(drop=True)
