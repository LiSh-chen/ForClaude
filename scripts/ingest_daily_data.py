"""每日資料抓取排程腳本：從 FinMind 抓最新的價量與融資券資料，寫進資料庫。

設計給 GitHub Actions 每日排程呼叫（見 .github/workflows/daily_data_ingest.yml）。
之所以不能在 Claude Code 這個開發沙盒裡直接跑：沙盒對外網路走白名單代理，
擋掉了 api.finmindtrade.com；GitHub Actions runner 有正常的網際網路存取權，
才是實際抓資料該執行的地方。

環境變數：
  FINMIND_TOKEN          FinMind API token（選填；沒有 token 也能用，但速率限制更嚴）
  DATABASE_URL           雲端 Postgres 連線字串（選填；沒設就退回本地 SQLite，
                         見 tw_quant/storage.py 的 get_data_store()）
  SQLITE_DB_PATH         SQLite 檔案路徑（預設 data/tw_market.db）
  STOCK_UNIVERSE         逗號分隔的股票代號清單（選填）
  STOCK_UNIVERSE_FILE    每行一個股票代號的文字檔路徑（選填）
  LOOKBACK_DAYS          增量同步時往回抓幾天，涵蓋補資料與假日（預設 10）
  REQUEST_SLEEP_SECONDS  每個 API 請求間隔秒數，避免打到免費額度速率限制（預設 0.5）
  FORCE_BACKFILL         設為 "true" 時，忽略每檔股票現有的資料，強制全部重新
                         回填 3 年（upsert 是冪等的，重跑很安全）。用來修復
                         「某次執行中途被打斷，導致部分股票只寫進一小段資料，
                         之後又被誤判成已經同步過」這種情況——這正是第一次接
                         Neon 時實際發生過的事，見 2026-09-17 的除錯紀錄。

沒有設定 STOCK_UNIVERSE / STOCK_UNIVERSE_FILE 時，只會抓一份示範用的
十檔權值股清單（見 DEFAULT_UNIVERSE）——這只是為了讓系統「今天就能動」，
真正要做「全上市櫃市場」的寬度/量能判定與選股，你需要把完整股票清單
（可以用 fetch_stock_info() 抓，或直接用證交所公開資訊）放進
STOCK_UNIVERSE_FILE，並注意 FinMind 免費額度的速率限制會讓全市場同步
耗時拉長，必要時應付費升級 token。
"""

from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path
from typing import Callable, TypeVar

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from tw_quant.data_provider import FinMindDataProvider
from tw_quant.storage import get_data_store

DEFAULT_UNIVERSE = ["2330", "2317", "2454", "2412", "2308", "1301", "2882", "2881", "3008", "2303"]

T = TypeVar("T")


def _call_with_timeout(func: Callable[[], T], timeout_s: float) -> T:
    """真正的 wall-clock 逾時保護，包住任何一次可能卡住的呼叫。

    requests 的 timeout 參數只在「單次讀取之間沒有新資料」時才會觸發，如果
    伺服器持續慢速吐資料（trickle），連線可能永遠不會逾時——實測
    FORCE_BACKFILL 那次，抓 TaiwanStockInfo 卡了整整 15 分鐘，直到
    workflow job 本身的逾時上限把整個程序砍掉，requests 自己設定的
    timeout=30 完全沒生效。改用背景執行緒 + join(timeout=...) 才是真正的
    保護：不管卡在哪裡、卡多久，主執行緒最多只等 timeout_s 秒就放棄並繼續
    往下跑；背景執行緒設成 daemon，就算它真的卡死也不會擋住程式正常結束
    （Python 直譯器只會等非 daemon 執行緒，daemon 執行緒會直接被拋棄）。
    """
    result: list[T] = []
    error: list[BaseException] = []

    def _worker() -> None:
        try:
            result.append(func())
        except BaseException as exc:  # noqa: BLE001 - 任何例外都要轉交回主執行緒判斷
            error.append(exc)

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()
    thread.join(timeout=timeout_s)

    if thread.is_alive():
        raise TimeoutError(f"呼叫逾時（{timeout_s} 秒），背景執行緒仍在等待回應")
    if error:
        raise error[0]
    return result[0]


def _load_universe() -> list[str]:
    env_list = os.environ.get("STOCK_UNIVERSE")
    if env_list:
        return [s.strip() for s in env_list.split(",") if s.strip()]
    env_file = os.environ.get("STOCK_UNIVERSE_FILE")
    if env_file and Path(env_file).exists():
        return [line.strip() for line in Path(env_file).read_text().splitlines() if line.strip()]
    print(f"[info] 未設定 STOCK_UNIVERSE，使用示範清單（{len(DEFAULT_UNIVERSE)} 檔）", file=sys.stderr)
    return DEFAULT_UNIVERSE


def _industry_lookup(provider: FinMindDataProvider) -> dict[str, str]:
    try:
        info = _call_with_timeout(provider.fetch_stock_info, timeout_s=20.0)
        if info.empty:
            return {}
        return dict(zip(info["stock_id"], info["industry"]))
    except Exception as exc:  # noqa: BLE001 - 網路/逾時/格式錯誤都降級為空對照表，不讓整個任務失敗
        print(f"[warn] 無法取得產業分類對照表: {exc}", file=sys.stderr)
        return {}


def main() -> None:
    token = os.environ.get("FINMIND_TOKEN")
    lookback_days = int(os.environ.get("LOOKBACK_DAYS", "10"))
    sleep_s = float(os.environ.get("REQUEST_SLEEP_SECONDS", "0.5"))
    force_backfill = os.environ.get("FORCE_BACKFILL", "").lower() == "true"

    provider = FinMindDataProvider(token=token)
    store = get_data_store()
    universe = _load_universe()
    industry_map = _industry_lookup(provider)

    end_date = pd.Timestamp.today().strftime("%Y-%m-%d")
    backfill_start = (pd.Timestamp.today() - pd.Timedelta(days=365 * 3)).strftime("%Y-%m-%d")

    print(f"同步結束日: {end_date}，共 {len(universe)} 檔股票（每檔各自判斷要回填還是增量同步）")

    total_price_rows = 0
    total_margin_rows = 0
    failures: list[str] = []

    for stock_id in universe:
        try:
            # 逐檔判斷同步起點：這檔股票在資料庫裡完全沒有資料 -> 回填 3 年；
            # 已經有資料 -> 只抓最近 lookback_days 天補齊缺口即可。
            # 刻意不用「全市場最新日期」當全域基準：一批股票裡只要有任何一檔
            # 曾經同步成功過，其他從來沒同步過的股票會被誤判成「已經有資料」，
            # 只抓到 lookback_days 天就再也補不回完整的 3 年歷史了（這是實測
            # 第一次接 Neon 時，某次執行中途被打斷、部分股票寫入成功、部分
            # 完全沒寫入，之後再也沒有回填的根本原因）。
            stock_latest = None if force_backfill else store.latest_date("prices", stock_id=stock_id)
            if stock_latest is not None:
                start_date = (stock_latest - pd.Timedelta(days=lookback_days)).strftime("%Y-%m-%d")
            else:
                start_date = backfill_start

            price_df = _call_with_timeout(
                lambda sid=stock_id, s=start_date, e=end_date: provider.fetch_price(sid, s, e), timeout_s=30.0
            )
            if not price_df.empty:
                price_df["industry"] = industry_map.get(stock_id, "UNKNOWN")
                store.upsert_prices(price_df)
                total_price_rows += len(price_df)
            time.sleep(sleep_s)

            margin_df = _call_with_timeout(
                lambda sid=stock_id, s=start_date, e=end_date: provider.fetch_margin_short(sid, s, e), timeout_s=30.0
            )
            if not margin_df.empty:
                store.upsert_margin_short(margin_df)
                total_margin_rows += len(margin_df)
            time.sleep(sleep_s)
        except Exception as exc:  # noqa: BLE001 - 單一檔失敗不該中斷整個排程
            print(f"[warn] {stock_id} 抓取失敗: {exc}", file=sys.stderr)
            failures.append(stock_id)

    print(f"完成。寫入價量 {total_price_rows} 筆，融資券 {total_margin_rows} 筆。")
    if failures:
        print(f"[warn] {len(failures)} 檔抓取失敗: {failures}", file=sys.stderr)

    store.close()


if __name__ == "__main__":
    main()
