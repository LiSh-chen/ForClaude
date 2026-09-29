"""逐月下載 TAIFEX 台指選擇權(TXO)原始逐履約價CSV，當場聚合成「當日
put/call成交量、未平倉量、成交值」等統計後就丟棄原始檔（不像期貨原始檔
那樣整份commit進repo）——選擇權資料是「每天x每個履約價x買權/賣權x
到期月份」的組合，一個月就有上萬列，23年完整逐履約價資料太大，不適合
整份存進git repo，只保留聚合後的每日統計。

聚合欄位（只用「一般」時段、只用最近到期月份，排除週選擇權，避免同一天
多個到期月份的部位混在一起稀釋訊號）：
- call_volume / put_volume：買權/賣權當日成交量加總
- call_oi / put_oi：買權/賣權當日未沖銷契約數加總
- put_call_volume_ratio = put_volume / call_volume
- put_call_oi_ratio = put_oi / call_oi
- atm_straddle_price：離現貨最近履約價的買權+賣權結算價加總（粗略的
  隱含波動度代理指標，不是真正用Black-Scholes反推的IV，但方向性夠用）

用法：
    python scripts/fetch_and_aggregate_txo.py 2001-12 2023-12
"""

from __future__ import annotations

import io
import sys
import time
from datetime import date
from pathlib import Path

import pandas as pd
import subprocess

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = REPO_ROOT / "data" / "taifex_txo_daily_putcall.parquet"
PROGRESS_DIR = REPO_ROOT / "data" / "raw" / "taifex_txo_progress"
ENDPOINT = "https://www.taifex.com.tw/cht/3/optDataDown"
SLEEP_SECONDS = 2.0
MAX_RETRIES = 3


def month_range(start_ym: str, end_ym: str) -> list[str]:
    sy, sm = map(int, start_ym.split("-"))
    ey, em = map(int, end_ym.split("-"))
    months = []
    y, m = sy, sm
    while (y, m) <= (ey, em):
        months.append(f"{y:04d}-{m:02d}")
        m += 1
        if m > 12:
            m = 1
            y += 1
    return months


def month_bounds(ym: str) -> tuple[str, str]:
    y, m = map(int, ym.split("-"))
    start = date(y, m, 1)
    end = date(y + 1, 1, 1) if m == 12 else date(y, m + 1, 1)
    end = date.fromordinal(end.toordinal() - 1)
    return start.strftime("%Y/%m/%d"), end.strftime("%Y/%m/%d")


def fetch_month_raw(ym: str) -> bytes | None:
    start_s, end_s = month_bounds(ym)
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            result = subprocess.run(
                ["curl", "-sSL", "-m", "60", "-X", "POST", ENDPOINT,
                 "-d", "down_type=1", "-d", "commodity_id=TXO", "-d", "commodity_id2=",
                 "-d", f"queryStartDate={start_s}", "-d", f"queryEndDate={end_s}"],
                capture_output=True, timeout=65,
            )
            if result.returncode == 0 and len(result.stdout) > 100:
                return result.stdout
            print(f"  [{ym}] attempt {attempt}: returncode={result.returncode} len={len(result.stdout)}")
        except Exception as e:
            print(f"  [{ym}] attempt {attempt} failed: {e}")
        time.sleep(3)
    return None


def aggregate_month(raw: bytes) -> pd.DataFrame:
    text = raw.decode("big5", errors="replace")
    if "交易日期" not in text:
        return pd.DataFrame()
    df = pd.read_csv(io.StringIO(text), dtype=str, index_col=False)
    df.columns = [c.strip() for c in df.columns]
    df["到期月份(週別)"] = df["到期月份(週別)"].str.strip()
    df["交易時段"] = df["交易時段"].str.strip()
    df["買賣權"] = df["買賣權"].str.strip()

    d = df[df["交易時段"] == "一般"].copy()
    d["date"] = pd.to_datetime(d["交易日期"], format="%Y/%m/%d")
    d["volume"] = pd.to_numeric(d["成交量"].str.replace(",", "", regex=False), errors="coerce")
    d["oi"] = pd.to_numeric(d["未沖銷契約數"].str.replace(",", "", regex=False), errors="coerce")
    d["settlement"] = pd.to_numeric(d["結算價"].str.replace(",", "", regex=False), errors="coerce")
    d["strike"] = pd.to_numeric(d["履約價"], errors="coerce")

    # 只用標準月合約(6碼到期月份，不含週別W1/W2等)。put/call量能比用「當天
    # 全部到期月份加總」(不限近月)——這是CBOE式put/call ratio的標準做法，
    # 也避開近月單獨算會在每月到期換月時出現的人為斷點(跟期貨OI踩過的
    # 陷阱一樣)。ATM跨式價才需要限定單一到期月份(近月)，混合到期日的
    # call+put價格加總沒有意義。
    d = d[d["到期月份(週別)"].str.match(r"^\d{6}$", na=False)]

    rows = []
    for dt, day_df in d.groupby("date"):
        calls_all = day_df[day_df["買賣權"] == "買權"]
        puts_all = day_df[day_df["買賣權"] == "賣權"]

        call_vol = calls_all["volume"].sum()
        put_vol = puts_all["volume"].sum()
        call_oi = calls_all["oi"].sum()
        put_oi = puts_all["oi"].sum()

        nearest_expiry = day_df["到期月份(週別)"].min()
        near_df = day_df[day_df["到期月份(週別)"] == nearest_expiry]
        calls = near_df[near_df["買賣權"] == "買權"]
        puts = near_df[near_df["買賣權"] == "賣權"]

        # ATM: 近月合約裡，call+put成交量加總最大的履約價，當「最活躍」
        # 履約價的代理（不需要現貨價格就能抓到市場關注的價位）
        merged = calls.merge(puts, on="strike", suffixes=("_call", "_put"))
        merged["total_vol"] = merged["volume_call"].fillna(0) + merged["volume_put"].fillna(0)
        atm_straddle = None
        if not merged.empty and merged["total_vol"].sum() > 0:
            atm_row = merged.loc[merged["total_vol"].idxmax()]
            if pd.notna(atm_row["settlement_call"]) and pd.notna(atm_row["settlement_put"]):
                atm_straddle = atm_row["settlement_call"] + atm_row["settlement_put"]

        rows.append(dict(
            date=dt, nearest_expiry=nearest_expiry,
            call_volume=call_vol, put_volume=put_vol,
            call_oi=call_oi, put_oi=put_oi,
            atm_straddle_price=atm_straddle,
        ))
    return pd.DataFrame(rows)


def main() -> None:
    if len(sys.argv) != 3:
        print("用法: python scripts/fetch_and_aggregate_txo.py START_YYYY-MM END_YYYY-MM")
        sys.exit(1)
    start_ym, end_ym = sys.argv[1], sys.argv[2]
    PROGRESS_DIR.mkdir(parents=True, exist_ok=True)

    months = month_range(start_ym, end_ym)
    print(f"共 {len(months)} 個月，從 {start_ym} 到 {end_ym}")

    all_frames = []
    # 讀取已經存在的進度檔（可續跑）
    existing_months = set()
    for fp in PROGRESS_DIR.glob("txo_daily_*.parquet"):
        ym = fp.stem.replace("txo_daily_", "")
        existing_months.add(ym)
        all_frames.append(pd.read_parquet(fp))

    ok, failed = 0, []
    for i, ym in enumerate(months, 1):
        if ym in existing_months:
            continue
        raw = fetch_month_raw(ym)
        if raw is None:
            failed.append(ym)
            print(f"[{i}/{len(months)}] {ym}: 失敗")
            continue
        agg = aggregate_month(raw)
        if agg.empty:
            print(f"[{i}/{len(months)}] {ym}: 無資料（可能是TXO開始交易前，2001-12-24才開始）")
        else:
            out_fp = PROGRESS_DIR / f"txo_daily_{ym}.parquet"
            agg.to_parquet(out_fp, index=False)
            all_frames.append(agg)
            ok += 1
        if i % 10 == 0 or i == len(months):
            print(f"[{i}/{len(months)}] 進度：成功={ok} 失敗={len(failed)}")
        time.sleep(SLEEP_SECONDS)

    if all_frames:
        combined = pd.concat(all_frames, ignore_index=True).sort_values("date").reset_index(drop=True)
        combined.to_parquet(OUT_PATH, index=False)
        print(f"\n合併後總交易日數: {len(combined)}")
        print(f"saved: {OUT_PATH}")

    print(f"\n完成。成功={ok} 失敗={len(failed)}")
    if failed:
        print("失敗月份:", failed)


if __name__ == "__main__":
    main()
