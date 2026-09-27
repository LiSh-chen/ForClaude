"""美股跨市場訊號：S&P 500指數／VIX每日收盤價，用來測試「美股隔夜漲跌/
恐慌指數能不能預測台指期次一交易日表現」這個假說。

**資料來源與查證**：這個開發沙盒連不到 Yahoo Finance / Stooq / FRED
（見這次會話跟使用者的討論，網路政策擋掉這幾個網域），但
`raw.githubusercontent.com`可以連——資料取自GitHub上一份公開、非官方維護
的CSV快照：

    https://raw.githubusercontent.com/vivek-v-rao/Conditional-Return-Stats/master/gspc_vix.csv

欄位是 Date,GSPC(S&P500收盤),VIX(VIX收盤)，涵蓋1990-01-02到2026-04-02，
共9,130個交易日、沒有重複日期。抓取後逐筆核對過幾個歷史上有公開記錄可以
對照的日期，數值都對得上（不是亂數/損毀資料）：
    2001-01-02  GSPC=1283.27  VIX=29.99
    2008-09-15  GSPC=1192.70  VIX=31.70（雷曼兄弟倒閉當天）
    2020-03-23  GSPC=2237.40  VIX=61.59（COVID崩盤最低點當天）
    2023-12-29  GSPC=4769.83  VIX=12.45（跟這次會話TXF資料的最後一天同一天）

**已知限制**：這是社群非官方維護的快照，不是官方資料源，理論上有被中途
修改/停止更新的風險（跟這次會話其餘資料源一致的警示，見
tw_quant/stooq_provider.py 對免費資料源的說明）；只有收盤價，沒有OHLC，
所以只能算「前一個美股交易日的收盤到收盤報酬」當訊號，沒辦法算美股當天
盤中的走勢細節。

已轉存成 data/us_gspc_vix_daily.parquet（欄位改名 date/sp500_close/
vix_close，日期由早到晚排序），這個模組負責讀取+跟TXF交易日對齊，不重新
連網。
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
US_DATA_PATH = REPO_ROOT / "data" / "us_gspc_vix_daily.parquet"


def load_us_daily() -> pd.DataFrame:
    """回傳 date, sp500_close, vix_close, sp500_ret(當天收盤對前一個美股
    交易日收盤的報酬率), vix_chg(當天VIX對前一天的變化，絕對值不是百分比，
    VIX本身已經是波動率的百分比單位，用差值比用比率更符合一般解讀習慣)。"""
    d = pd.read_parquet(US_DATA_PATH).sort_values("date").reset_index(drop=True)
    d["sp500_ret"] = d["sp500_close"].pct_change()
    d["vix_chg"] = d["vix_close"].diff()
    return d


def align_to_txf_dates(txf_dates: pd.Series) -> pd.DataFrame:
    """對每一個TXF交易日期，找出「該交易日日盤開盤(08:45台北時間)之前，
    最近一個已經收盤的美股交易日」的訊號。做法：把TXF日期往前推1個日曆天
    再跟美股資料做backward asof合併——美股收盤(美東時間下午4點)換算成台北
    時間是隔天凌晨4-5點，TXF日盤開盤在當天08:45，所以「TXF日期當天的美股
    收盤」根本還沒發生，一定要往前推至少1個日曆天，避免用到未來資訊。
    calendar偶爾TW/US交易日沒有完全對齊（假日不同步），backward asof會自
    動找到「最近的較早日期」，不會因為當天剛好沒有美股資料就出錯或誤用
    未來資料。"""
    txf = pd.DataFrame({"txf_date": pd.to_datetime(txf_dates).sort_values().unique()})
    txf["asof_key"] = txf["txf_date"] - pd.Timedelta(days=1)
    us = load_us_daily().rename(columns={"date": "us_date"})
    merged = pd.merge_asof(
        txf.sort_values("asof_key"), us.sort_values("us_date"),
        left_on="asof_key", right_on="us_date", direction="backward",
    )
    return merged[["txf_date", "us_date", "sp500_close", "vix_close", "sp500_ret", "vix_chg"]]
