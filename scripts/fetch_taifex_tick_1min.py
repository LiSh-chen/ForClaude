"""從TAIFEX官方「前30個交易日期貨每筆成交資料」下載當天台指期(TX)逐筆
成交，聚合成1分K，補上舊1分K靜態檔（只到2023-12-29，來源是網路分享的
歷史檔案，沒有更新管道）沒有的「今天」資料，讓三腳組合（開盤上衝/午盤
放空/盤中翻多）也能接上每日模擬倉追蹤。

資料源：https://www.taifex.com.tw/cht/3/dlFutPrevious30DaysSalesData
（頁面本身，下載連結是 https://www.taifex.com.tw/file/taifex/Dailydownload/
DailydownloadCSV/Daily_YYYY_MM_DD.zip，只保留最近30個交易日，逐筆成交
含全部商品，這裡只篩TX、只留近月合約的標準月）。

**交易日定義，不用自己推算**：這份下載本來就是「以交易日為單位」打包的
——Daily_2026_09_29.zip 內含的「成交日期」欄位橫跨09/24晚上15:00開始的
夜盤到09/29的日盤（中間09/25~09/28休市），但整個檔案就是「2026/09/29
這個交易日」的完整資料，檔名的日期本身就是trading_date，不需要另外用
時間+日期反推屬於哪個交易日（這正是1分K原始資料處理時最容易犯錯的地方）。

用法：
    python scripts/fetch_taifex_tick_1min.py 2026-09-29
    python scripts/fetch_taifex_tick_1min.py            # 不帶參數=抓最新交易日
"""

from __future__ import annotations

import io
import subprocess
import sys
import zipfile
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "data" / "raw" / "taifex_tick_1min"
ENDPOINT = "https://www.taifex.com.tw/file/taifex/Dailydownload/DailydownloadCSV/Daily_{y}_{m}_{d}.zip"


def fetch_daily_zip(trading_date: str) -> bytes | None:
    """trading_date格式 YYYY-MM-DD。回傳zip檔bytes，抓不到(例如那天不是
    交易日、或超過30天窗口)回傳None。"""
    y, m, d = trading_date.split("-")
    url = ENDPOINT.format(y=y, m=m, d=d)
    result = subprocess.run(["curl", "-sSL", "-m", "60", url], capture_output=True, timeout=65)
    if result.returncode != 0 or len(result.stdout) < 1000:
        return None
    # 官方查無資料時可能回傳一個小的錯誤頁而非zip，用zip檔頭簽章("PK")確認
    if result.stdout[:2] != b"PK":
        return None
    return result.stdout


def parse_tx_1min(zip_bytes: bytes, trading_date: str) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
        names = z.namelist()
        if len(names) != 1:
            raise ValueError(f"預期zip內只有一個檔案，實際: {names}")
        raw = z.read(names[0])
    text = raw.decode("big5", errors="replace")
    df = pd.read_csv(io.StringIO(text), dtype=str, index_col=False)
    df.columns = [c.strip() for c in df.columns]

    d = df[df["商品代號"].str.strip() == "TX"].copy()
    if d.empty:
        return pd.DataFrame(columns=["datetime", "open", "high", "low", "close", "volume"])

    d["到期月份(週別)"] = d["到期月份(週別)"].str.strip()
    d = d[d["到期月份(週別)"].str.match(r"^\d{6}$", na=False)]  # 只留標準月合約，排除週選
    nearest_expiry = d["到期月份(週別)"].min()
    d = d[d["到期月份(週別)"] == nearest_expiry].copy()

    d["price"] = pd.to_numeric(d["成交價格"], errors="coerce")
    d["qty"] = pd.to_numeric(d["成交數量(B+S)"], errors="coerce")
    d["date_str"] = d["成交日期"].str.strip()
    d["time_str"] = d["成交時間"].str.strip().str.zfill(6)
    d["datetime"] = pd.to_datetime(d["date_str"] + d["time_str"], format="%Y%m%d%H%M%S")
    d = d[d["price"].notna() & (d["price"] > 0)]
    # sort_values預設用quicksort，不保證穩定——datetime只精確到秒，同一秒內
    # 常有數十筆成交，quicksort會打亂它們在原始CSV裡的真實先後順序，導致
    # groupby().first()/.last()取到錯的開盤/收盤價（例如開盤那筆284口的大單
    # 排到中間去，第一筆變成隨機一筆，被抓到過：實測開盤價因此偏差29點）。
    # 原始CSV本身就是依成交序號正序排列，這裡用穩定排序，同一秒內保留
    # CSV原始順序，才能正確還原「這一分鐘內第一筆/最後一筆成交」。
    d = d.sort_values("datetime", kind="stable")

    d["minute"] = d["datetime"].dt.floor("min")
    agg = d.groupby("minute").agg(
        open=("price", "first"), high=("price", "max"), low=("price", "min"),
        close=("price", "last"), volume=("qty", "sum"),
    ).reset_index().rename(columns={"minute": "datetime"})
    return agg.sort_values("datetime").reset_index(drop=True)


def recent_trading_date_guess() -> str:
    """不帶參數執行時的預設猜測：今天，若今天抓不到（非交易日/資料還沒
    公告），呼叫端會自動往前一天重試，見main()。"""
    return date.today().strftime("%Y-%m-%d")


def fetch_and_save(trading_date: str) -> pd.DataFrame | None:
    out_path = OUT_DIR / f"{trading_date}.parquet"
    zip_bytes = fetch_daily_zip(trading_date)
    if zip_bytes is None:
        return None
    agg = parse_tx_1min(zip_bytes, trading_date)
    if agg.empty:
        return None
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    agg.to_parquet(out_path, index=False)
    return agg


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if len(sys.argv) > 1:
        trading_date = sys.argv[1]
        agg = fetch_and_save(trading_date)
        if agg is None:
            print(f"{trading_date}: 抓不到資料（非交易日、超過30天窗口、或TAIFEX尚未公告）")
            sys.exit(1)
        print(f"{trading_date}: {len(agg)} 根1分K，{agg['datetime'].min()} ~ {agg['datetime'].max()}")
        return

    # 不帶參數：從今天往前找，最多試5天（涵蓋連假），抓到第一個有資料的就停
    d = date.today()
    for _ in range(5):
        ds = d.strftime("%Y-%m-%d")
        agg = fetch_and_save(ds)
        if agg is not None:
            print(f"{ds}: {len(agg)} 根1分K，{agg['datetime'].min()} ~ {agg['datetime'].max()}")
            return
        print(f"{ds}: 抓不到，往前一天重試")
        d -= timedelta(days=1)
    print("連續5天都抓不到資料，中止")
    sys.exit(1)


if __name__ == "__main__":
    main()
