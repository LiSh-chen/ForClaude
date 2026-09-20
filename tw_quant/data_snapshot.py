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

us_prices 分檔（2026-09-20 新增）：2026-09-20 為了測 2008 金融風暴，把
backfill_years 從 8 拉到 20，匯出的單一 us_prices_snapshot.parquet 檔案
膨脹到 101.2MB，超過 GitHub 單檔 100MB push 上限，被 pre-receive hook
擋下（見對話紀錄）。使用者要求不要繞過快照直連資料庫，改成把 us_prices
拆成多個檔案，維持「所有策略腳本一律讀本機快照、不連資料庫」這個既有
架構。做法：export_us_snapshot 依列數把 us_prices 切成多個檔案，第一份
沿用原本的檔名 us_prices_snapshot.parquet（向後相容，資料量沒大到需要
分檔時，行為跟以前完全一樣、只會有這一個檔案），第二份起用
us_prices_snapshot.part2.parquet、part3.parquet……依序命名；
load_us_prices_snapshot 讀完第一份後，額外 glob 抓所有 part 檔案一併
concat 回單一 DataFrame，呼叫端完全不用感知有沒有分檔。
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

SNAPSHOT_DIR = Path(__file__).resolve().parents[1] / "data"
US_PRICES_SNAPSHOT_PATH = SNAPSHOT_DIR / "us_prices_snapshot.parquet"
US_INDEX_MEMBERSHIP_SNAPSHOT_PATH = SNAPSHOT_DIR / "us_index_membership_snapshot.parquet"
US_EARNINGS_SNAPSHOT_PATH = SNAPSHOT_DIR / "us_earnings_snapshot.parquet"

# us_prices 每個檔案最多幾列才切下一檔。實測 8 年版（2018-2026、577 檔股票）
# 45,640,871 bytes / 1,058,028 列 ≈ 43 bytes/列，這裡抓 1,300,000 列/檔
# （約 56MB），留了接近一倍的安全邊際——即使欄位/型別未來有變動、實際
# 壓縮率沒那麼好，也不太可能真的踩到 GitHub 的 100MB 單檔上限。
MAX_ROWS_PER_PRICE_CHUNK = 1_300_000

_PART_FILE_RE = re.compile(r"\.part(\d+)$")


def _price_chunk_path(base_path: Path, chunk_index: int) -> Path:
    """chunk_index=0 沿用原本檔名（向後相容）；chunk_index>=1 依序命名
    成 .part2、.part3……（part 編號從 2 開始，呼應「這是第 2/3/… 份」的
    直覺，不是程式內部的 0-based 索引）。
    """
    if chunk_index == 0:
        return base_path
    return base_path.with_name(f"{base_path.stem}.part{chunk_index + 1}{base_path.suffix}")


def _existing_part_paths(base_path: Path) -> list[Path]:
    """找出跟 base_path 同目錄、屬於同一份快照的既有 part 檔案（用來在
    重新匯出時清掉舊檔——否則資料量變少、part 檔案數變少時，舊的多餘
    part 檔案會一直留著，讀取端 glob 進來變成重複資料）。
    """
    pattern = f"{base_path.stem}.part*{base_path.suffix}"
    return sorted(base_path.parent.glob(pattern))


def export_us_snapshot(
    store,
    prices_path: Path = US_PRICES_SNAPSHOT_PATH,
    membership_path: Path = US_INDEX_MEMBERSHIP_SNAPSHOT_PATH,
    max_rows_per_chunk: int = MAX_ROWS_PER_PRICE_CHUNK,
) -> tuple[int, int]:
    """從傳入的 DataStore 讀出目前的 us_prices、us_index_membership 全部
    內容，寫成 Parquet 快照檔。回傳 (us_prices 列數, membership 列數)
    方便呼叫端印出摘要。呼叫端負責在寫入資料庫之後才呼叫這個函式，這裡
    不管寫入邏輯、只管匯出。

    us_prices 依 max_rows_per_chunk 切成多個檔案（見檔頭說明），寫入前
    先刪掉這個 base_path 底下所有既有的 part 檔案，避免資料量變少時舊的
    多餘 part 檔案殘留、被讀取端誤當成額外資料 concat 進去。
    max_rows_per_chunk 開放覆寫只是方便測試（不用真的生出百萬列資料才能
    測到「切成多檔」這條路徑），正式呼叫端一律用預設的 MAX_ROWS_PER_PRICE_CHUNK。
    """
    us_prices = store.load_us_prices()
    membership = store.load_us_index_membership()

    prices_path.parent.mkdir(parents=True, exist_ok=True)
    membership_path.parent.mkdir(parents=True, exist_ok=True)

    for stale_part in _existing_part_paths(prices_path):
        stale_part.unlink()

    n_prices = len(us_prices)
    n_chunks = max(1, -(-n_prices // max_rows_per_chunk))  # ceil div，至少寫一個檔（即使是空表）
    for i in range(n_chunks):
        start, end = i * max_rows_per_chunk, (i + 1) * max_rows_per_chunk
        chunk = us_prices.iloc[start:end]
        chunk.to_parquet(_price_chunk_path(prices_path, i), index=False)

    membership.to_parquet(membership_path, index=False)

    return n_prices, len(membership)


def load_us_prices_snapshot(path: Path = US_PRICES_SNAPSHOT_PATH) -> pd.DataFrame:
    """讀本機的美股價量快照檔，不連資料庫。快照檔不存在時丟出清楚的
    錯誤訊息，而不是靜默回傳空表——空表會讓後面的回測腳本誤判成
    「資料庫是空的」，比直接報錯更容易誤導人。

    資料量大到超過單檔上限時，export_us_snapshot 會把 us_prices 拆成
    base 檔 + 多個 .partN 檔（見檔頭說明），這裡讀完 base 檔後自動 glob
    抓所有 part 檔案一併 concat，呼叫端完全不用感知有沒有分檔。
    """
    if not path.exists():
        raise FileNotFoundError(
            f"找不到美股價量快照檔 {path}——需要先跑過 scripts/export_us_data_snapshot.py"
            "（或讓 us_daily_data_ingest.yml / backfill_removed_sp500_stocks.yml 的"
            "匯出步驟先執行過一次並 commit 回 repo），才會有這個檔案可以讀。"
        )
    frames = [pd.read_parquet(path)]
    part_paths = _existing_part_paths(path)
    part_paths.sort(key=lambda p: int(_PART_FILE_RE.search(p.stem).group(1)))
    for part_path in part_paths:
        frames.append(pd.read_parquet(part_path))
    return pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]


def load_us_index_membership_snapshot(path: Path = US_INDEX_MEMBERSHIP_SNAPSHOT_PATH) -> pd.DataFrame:
    """讀本機的美股指數成分股區間快照檔，語意同上。"""
    if not path.exists():
        raise FileNotFoundError(
            f"找不到美股指數成分股快照檔 {path}——需要先跑過 scripts/export_us_data_snapshot.py"
            "（或讓 us_daily_data_ingest.yml / backfill_removed_sp500_stocks.yml 的"
            "匯出步驟先執行過一次並 commit 回 repo），才會有這個檔案可以讀。"
        )
    return pd.read_parquet(path)


def export_us_earnings_snapshot(store, path: Path = US_EARNINGS_SNAPSHOT_PATH) -> int:
    """從傳入的 DataStore 讀出目前的 us_earnings 全部內容，寫成 Parquet
    快照檔。回傳列數方便呼叫端印出摘要。

    獨立成自己的函式（不是塞進 export_us_snapshot），因為財報公布資料的
    更新排程（一次性/低頻，見 scripts/ingest_us_earnings_data.py）跟每日
    價量排程（us_daily_data_ingest.yml）完全不同——如果塞進同一個匯出
    函式，每天排程都會跑的價量匯出步驟會在 earnings ingestion 真的執行過
    一次之前，就用空的 us_earnings 表覆寫掉這份快照。
    """
    us_earnings = store.load_us_earnings()
    path.parent.mkdir(parents=True, exist_ok=True)
    us_earnings.to_parquet(path, index=False)
    return len(us_earnings)


def load_us_earnings_snapshot(path: Path = US_EARNINGS_SNAPSHOT_PATH) -> pd.DataFrame:
    """讀本機的美股財報公布快照檔，語意同 load_us_prices_snapshot。"""
    if not path.exists():
        raise FileNotFoundError(
            f"找不到美股財報公布快照檔 {path}——需要先跑過 scripts/ingest_us_earnings_data.py"
            "（或讓 ingest_us_earnings_data.yml 執行過一次並 commit 回 repo），"
            "才會有這個檔案可以讀。"
        )
    return pd.read_parquet(path)
