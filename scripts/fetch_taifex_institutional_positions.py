"""抓取三大法人（自營商/投信/外資及陸資）臺股期貨(TXF)每日買賣與未平倉
資料——這是「籌碼面」訊號裡最多台指期玩家關注的資料，這次會話之前完全
沒測過（先前只測過put/call未平倉比率，屬於選擇權籌碼，不是期貨籌碼）。

端點：https://www.taifex.com.tw/cht/3/futContractsDateDown (POST, commodityId=TXF)

**重要限制（先探測後才知道）**：這個公開下載端點目前只提供
**2023-10-02之後**的資料，2023-09-01以前的查詢一律回傳「日期時間錯誤」
的錯誤頁面(不是查無資料，是端點本身的服務窗口限制，原因不明，可能是
TAIFEX對這個下載端點做過系統更版)。這代表三大法人期貨籌碼資料目前
只有約2年多的樣本可用，遠不足以比照這次會話一貫的IS(2001-2020)/
OOS(2021-2023)驗證紀律——這裡誠實記錄這個限制，不假裝能做到完整驗證。

用法：
    python scripts/fetch_taifex_institutional_positions.py
"""

from __future__ import annotations

import subprocess
import time
from io import StringIO
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = REPO_ROOT / "data" / "taifex_txf_institutional_daily.parquet"
ENDPOINT = "https://www.taifex.com.tw/cht/3/futContractsDateDown"

# 探測得知的可用範圍起點；結束用今天
EARLIEST_AVAILABLE = "2023/10/02"


def fetch_range(start: str, end: str) -> pd.DataFrame | None:
    result = subprocess.run(
        ["curl", "-sS", "-m", "40", "-X", "POST", ENDPOINT,
         "-d", f"queryStartDate={start}", "-d", f"queryEndDate={end}", "-d", "commodityId=TXF"],
        capture_output=True, timeout=45,
    )
    if result.returncode != 0 or len(result.stdout) < 100:
        return None
    text = result.stdout.decode("big5", errors="replace")
    if "日期" not in text or "身份別" not in text:
        return None
    df = pd.read_csv(StringIO(text), dtype=str, index_col=False)
    df.columns = [c.strip() for c in df.columns]
    return df


def main() -> None:
    print(f"探測到的可用範圍起點：{EARLIEST_AVAILABLE}（更早的查詢會回傳DateTime error錯誤頁）")
    today = pd.Timestamp.today().strftime("%Y/%m/%d")

    # 分季抓，避免單次查詢範圍過大被拒絕
    start_dt = pd.Timestamp(EARLIEST_AVAILABLE.replace("/", "-"))
    end_dt = pd.Timestamp.today()
    chunks = []
    cur = start_dt
    while cur <= end_dt:
        chunk_end = min(cur + pd.Timedelta(days=90), end_dt)
        s, e = cur.strftime("%Y/%m/%d"), chunk_end.strftime("%Y/%m/%d")
        print(f"抓取 {s} ~ {e} ...")
        df = fetch_range(s, e)
        if df is not None:
            chunks.append(df)
            print(f"  取得 {len(df)} 列")
        else:
            print("  失敗或無資料")
        time.sleep(2)
        cur = chunk_end + pd.Timedelta(days=1)

    if not chunks:
        print("完全沒有抓到資料，中止")
        return

    all_df = pd.concat(chunks, ignore_index=True)
    all_df["date"] = pd.to_datetime(all_df["日期"], format="%Y/%m/%d")

    def to_num(col):
        return pd.to_numeric(all_df[col].str.replace(",", "", regex=False), errors="coerce")

    out = pd.DataFrame({
        "date": all_df["date"],
        "identity": all_df["身份別"].str.strip(),
        "long_oi": to_num("多方未平倉口數"),
        "short_oi": to_num("空方未平倉口數"),
        "net_oi": to_num("多空未平倉口數淨額"),
        "net_oi_value_k": to_num("多空未平倉契約金額淨額(千元)"),
    })
    out = out.drop_duplicates(subset=["date", "identity"]).sort_values(["date", "identity"])
    out.to_parquet(OUT_PATH, index=False)
    print(f"\n共 {out['date'].nunique()} 個交易日，範圍 {out['date'].min()} ~ {out['date'].max()}")
    print(f"saved: {OUT_PATH}")
    print("\n身份別分布:")
    print(out["identity"].value_counts())


if __name__ == "__main__":
    main()
