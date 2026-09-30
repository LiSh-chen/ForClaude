"""逐月下載 TAIFEX 台指選擇權(TXO)原始逐履約價CSV，這次額外保留「最大OI
履約價」相關聚合（fetch_and_aggregate_txo.py 只留put/call總量，把履約價
資訊丟掉了，這裡要用履約價資訊測「最大OI履約價=支撐/壓力」的假說，所以
要重新下載一次，當場算好就丟棄原始檔，邏輯跟原本那支腳本一樣）。

新增聚合欄位（只用「一般」時段、只用最近到期月份的標準月合約，排除週選
擇權——原因跟原本那支腳本一樣：避免同一天多個到期月份的部位混在一起）：
- max_call_oi_strike / max_call_oi：買權未沖銷契約數最大的履約價跟其OI
  （使用者說的「壓力」，莊家怕結算要賠付，會在這個價位上方放空指數成分股
  壓回去）
- max_put_oi_strike / max_put_oi：賣權未沖銷契約數最大的履約價跟其OI
  （使用者說的「支撐」）
- max_pain_strike：正式的max pain計算——對每個候選結算價S，加總所有
  買權賣方在S結算時要賠付的內含價值(OI_call(K)*max(S-K,0)加總所有K)，
  加上所有賣權賣方要賠付的(OI_put(K)*max(K-S,0)加總)，找出讓「賣方總
  賠付」最小的S。這是比「單一最大OI履約價」更嚴謹的版本，兩者通常接近
  但不一定相同，順便算出來供比較。
- call_oi_total / put_oi_total：跟原本聚合腳本一樣的總量（拿來跟舊檔案
  交叉驗證下載/解析邏輯有沒有出錯）

用法：
    python scripts/fetch_txo_max_oi_strike.py 2001-12 2026-09
"""

from __future__ import annotations

import io
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import subprocess

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = REPO_ROOT / "data" / "taifex_txo_max_oi_strike.parquet"
PROGRESS_DIR = REPO_ROOT / "data" / "raw" / "taifex_txo_maxoi_progress"
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


def max_pain_strike(call_oi_by_strike: pd.Series, put_oi_by_strike: pd.Series) -> float | None:
    strikes = sorted(set(call_oi_by_strike.index) | set(put_oi_by_strike.index))
    if not strikes:
        return None
    strikes_arr = np.array(strikes, dtype=float)
    call_k = call_oi_by_strike.index.to_numpy(dtype=float)
    call_v = call_oi_by_strike.to_numpy(dtype=float)
    put_k = put_oi_by_strike.index.to_numpy(dtype=float)
    put_v = put_oi_by_strike.to_numpy(dtype=float)
    best_s, best_payout = None, None
    for s in strikes_arr:
        payout = float(np.sum(call_v * np.maximum(s - call_k, 0.0)) +
                        np.sum(put_v * np.maximum(put_k - s, 0.0)))
        if best_payout is None or payout < best_payout:
            best_payout, best_s = payout, s
    return best_s


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
    d["oi"] = pd.to_numeric(d["未沖銷契約數"].str.replace(",", "", regex=False), errors="coerce")
    d["strike"] = pd.to_numeric(d["履約價"], errors="coerce")

    # 只用標準月合約(6碼到期月份，不含週別W1/W2等)，理由跟
    # fetch_and_aggregate_txo.py一樣：避免多個到期月份的OI混在一起。
    d = d[d["到期月份(週別)"].str.match(r"^\d{6}$", na=False)]

    rows = []
    for dt, day_df in d.groupby("date"):
        nearest_expiry = day_df["到期月份(週別)"].min()
        near_df = day_df[day_df["到期月份(週別)"] == nearest_expiry]
        calls = near_df[near_df["買賣權"] == "買權"]
        puts = near_df[near_df["買賣權"] == "賣權"]

        call_oi_by_strike = calls.groupby("strike")["oi"].sum().dropna()
        put_oi_by_strike = puts.groupby("strike")["oi"].sum().dropna()

        if call_oi_by_strike.empty or put_oi_by_strike.empty:
            continue

        max_call_strike = call_oi_by_strike.idxmax()
        max_put_strike = put_oi_by_strike.idxmax()
        mp = max_pain_strike(call_oi_by_strike, put_oi_by_strike)

        rows.append(dict(
            date=dt, nearest_expiry=nearest_expiry,
            max_call_oi_strike=float(max_call_strike), max_call_oi=float(call_oi_by_strike.max()),
            max_put_oi_strike=float(max_put_strike), max_put_oi=float(put_oi_by_strike.max()),
            max_pain_strike=mp,
            call_oi_total=float(call_oi_by_strike.sum()), put_oi_total=float(put_oi_by_strike.sum()),
        ))
    return pd.DataFrame(rows)


def main() -> None:
    if len(sys.argv) != 3:
        print("用法: python scripts/fetch_txo_max_oi_strike.py START_YYYY-MM END_YYYY-MM")
        sys.exit(1)
    start_ym, end_ym = sys.argv[1], sys.argv[2]
    PROGRESS_DIR.mkdir(parents=True, exist_ok=True)

    months = month_range(start_ym, end_ym)
    print(f"共 {len(months)} 個月，從 {start_ym} 到 {end_ym}")

    all_frames = []
    existing_months = set()
    for fp in PROGRESS_DIR.glob("txo_maxoi_*.parquet"):
        ym = fp.stem.replace("txo_maxoi_", "")
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
            out_fp = PROGRESS_DIR / f"txo_maxoi_{ym}.parquet"
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
