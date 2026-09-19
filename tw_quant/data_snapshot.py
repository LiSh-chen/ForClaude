"""把資料庫的美股資料匯出成 commit 進 repo 的 Parquet 快照檔，讓策略回測
腳本改成讀這份快照、不用每次都連 Postgres 整表撈一次。

背景：2026-09-19 這次重建 Neon 新帳號時發現免費方案有「每月 5GB 網路傳出
流量」的額度限制（見對話紀錄），11 個策略腳本每個都呼叫
`store.load_us_prices()` 整表讀出來（503+74 檔股票 × 約 2000 個交易日），
一天內重跑個幾輪就有機會把額度用完，稍早舊帳號就是這樣被打滿的。

這些策略回測腳本本來就已經奉行「一律傳完整 us_prices 給引擎、只用
start_date/end_date 限制交易日期」的反未來函數原則，代表它們每次都要讀
「全部」歷史，天生就是整表讀取的重度使用者——但價量資料本身不會在兩次
ingest/backfill 之間改變，同一份資料被同一天的好幾個策略腳本各自重新
向資料庫要一次，是完全可以避免的重複傳輸。

用法：
  - 寫入端（ingest_us_daily_data.py、backfill_removed_sp500_stocks.py）
    完成資料庫寫入後呼叫 export_us_snapshot(store)，把最新資料匯出成
    Parquet，workflow 再把這兩個檔案 commit 回 repo。
  - 讀取端（11 個策略回測腳本）改成呼叫 load_us_prices_snapshot() /
    load_us_index_membership_snapshot()，直接讀 checkout 下來的本機檔案，
    完全不連資料庫——只有真的要「更新」資料（跑 ingest 或 backfill）時
    才會碰到 Postgres 的網路傳輸額度。
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

SNAPSHOT_DIR = Path(__file__).resolve().parents[1] / "data"
US_PRICES_SNAPSHOT_PATH = SNAPSHOT_DIR / "us_prices_snapshot.parquet"
US_INDEX_MEMBERSHIP_SNAPSHOT_PATH = SNAPSHOT_DIR / "us_index_membership_snapshot.parquet"


def export_us_snapshot(store, prices_path: Path = US_PRICES_SNAPSHOT_PATH, membership_path: Path = US_INDEX_MEMBERSHIP_SNAPSHOT_PATH) -> tuple[int, int]:
    """從傳入的 DataStore 讀出目前的 us_prices、us_index_membership 全部
    內容，寫成 Parquet 快照檔。回傳 (us_prices 列數, membership 列數)
    方便呼叫端印出摘要。呼叫端負責在寫入資料庫之後才呼叫這個函式，這裡
    不管寫入邏輯、只管匯出。
    """
    us_prices = store.load_us_prices()
    membership = store.load_us_index_membership()

    prices_path.parent.mkdir(parents=True, exist_ok=True)
    membership_path.parent.mkdir(parents=True, exist_ok=True)
    us_prices.to_parquet(prices_path, index=False)
    membership.to_parquet(membership_path, index=False)

    return len(us_prices), len(membership)


def load_us_prices_snapshot(path: Path = US_PRICES_SNAPSHOT_PATH) -> pd.DataFrame:
    """讀本機的美股價量快照檔，不連資料庫。快照檔不存在時丟出清楚的
    錯誤訊息，而不是靜默回傳空表——空表會讓後面的回測腳本誤判成
    「資料庫是空的」，比直接報錯更容易誤導人。
    """
    if not path.exists():
        raise FileNotFoundError(
            f"找不到美股價量快照檔 {path}——需要先跑過 scripts/export_us_data_snapshot.py"
            "（或讓 us_daily_data_ingest.yml / backfill_removed_sp500_stocks.yml 的"
            "匯出步驟先執行過一次並 commit 回 repo），才會有這個檔案可以讀。"
        )
    return pd.read_parquet(path)


def load_us_index_membership_snapshot(path: Path = US_INDEX_MEMBERSHIP_SNAPSHOT_PATH) -> pd.DataFrame:
    """讀本機的美股指數成分股區間快照檔，語意同上。"""
    if not path.exists():
        raise FileNotFoundError(
            f"找不到美股指數成分股快照檔 {path}——需要先跑過 scripts/export_us_data_snapshot.py"
            "（或讓 us_daily_data_ingest.yml / backfill_removed_sp500_stocks.yml 的"
            "匯出步驟先執行過一次並 commit 回 repo），才會有這個檔案可以讀。"
        )
    return pd.read_parquet(path)
