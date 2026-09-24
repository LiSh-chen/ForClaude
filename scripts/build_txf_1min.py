"""合併 TXF（台指期）CrazyIndicator 1 分鐘 K 棒原始 CSV（存在 7z 內），輸出成
單一乾淨、依時間排序、無重複時間戳的 parquet 檔（data/txf_1min.parquet）。

★★★ 重要：TXF20010101_20210731_Fix_Gap 這個檔案已知價格嚴重失真，不使用 ★★★
這個檔案原本標榜是「補齊缺漏分鐘」的修正版，一開始被當成 2001~2021/07 這段
的權威來源。事後用其餘三個「未修正」原始檔案逐年交叉比對才發現：它的價格
從 2011 年起跟原始資料出現持續擴大的正向偏差（2011 年平均偏差約 +46%，
2020 年已經到約 +70%，2021/07 最嚴重處單日偏差達 +93%；2021/07/30 →
2021/08/02 之間甚至出現一個完全不存在的「單日跳空 -35%」，是這份資料
唯一一次觸發全資料集的異常跳空檢查）。研判是 Fix_Gap 檔案的轉倉調整
（近月合約銜接）算法有 bug，逐月累積放大成長期正向偏誤，不是單次可忽略
的雜訊。這個檔案保留在 `data/raw/txf/` 只當作已知有問題的歷史紀錄，
不再參與資料合併。

改用經過交叉驗證、彼此首尾完全銜接、沒有這個問題的三個原始檔案：
- TXF20010101~20101231：2001-01-02 ~ 2010-12-31
- TXF20110101_20201231：2011-01-03 ~ 2020-12-31（銜接處 8899→9006，正常）
- TXF20210101_20231231：2020-12-31（夜盤起）~ 2023-12-29（銜接處
  14679→14646，正常）

三者時間範圍完全不重疊、首尾銜接處價格連續（見 `_check_boundary_continuity`），
直接串接即可，不需要任何「取代/優先」邏輯。

資料本身已經是近月連續（未依合約別拆分、未做逆向調整），沒有 Date/Time 以外
的欄位可以還原原始合約與到期日，使用前請留意近月轉倉時可能存在的價格跳空。
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import py7zr

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = REPO_ROOT / "data" / "raw" / "txf"
OUT_PATH = REPO_ROOT / "data" / "txf_1min.parquet"

SOURCE_7Z_FILES = [
    RAW_DIR / "TXF20010101~20101231(CrazyIndicator.pixnet.net).7z",
    RAW_DIR / "TXF20110101_20201231(CrazyIndicator.pixnet.net).7z",
    RAW_DIR / "TXF20210101_20231231(CrazyIndicator.pixnet.net).7z",
]
MAX_BOUNDARY_JUMP_PCT = 5.0  # 檔案銜接處收盤價變動超過這個百分比就視為可疑，中止合併


def _read_csv_from_7z(path: Path) -> pd.DataFrame:
    with tempfile.TemporaryDirectory() as tmp:
        with py7zr.SevenZipFile(path, mode="r") as z:
            names = z.getnames()
            if len(names) != 1:
                raise ValueError(f"{path.name} 預期只有一個檔案，實際: {names}")
            z.extractall(path=tmp)
        df = pd.read_csv(Path(tmp) / names[0])

    df.columns = [c.lower() for c in df.columns]
    df["datetime"] = pd.to_datetime(df["date"] + " " + df["time"], format="%Y/%m/%d %H:%M:%S")
    return df[["datetime", "open", "high", "low", "close", "volume"]].sort_values("datetime").reset_index(drop=True)


def _check_boundary_continuity(prev: pd.DataFrame, nxt: pd.DataFrame, prev_name: str, nxt_name: str) -> None:
    prev_close = prev["close"].iloc[-1]
    next_open = nxt["open"].iloc[0]
    jump_pct = abs(next_open - prev_close) / prev_close * 100
    if jump_pct > MAX_BOUNDARY_JUMP_PCT:
        raise AssertionError(
            f"{prev_name} 結尾收盤價 {prev_close} 跟 {nxt_name} 開頭開盤價 {next_open} "
            f"相差 {jump_pct:.1f}%，超過 {MAX_BOUNDARY_JUMP_PCT}% 門檻，疑似資料銜接錯誤，中止合併"
        )


def build() -> pd.DataFrame:
    parts = [_read_csv_from_7z(p) for p in SOURCE_7Z_FILES]
    for i in range(len(parts) - 1):
        _check_boundary_continuity(parts[i], parts[i + 1], SOURCE_7Z_FILES[i].name, SOURCE_7Z_FILES[i + 1].name)

    merged = pd.concat(parts, ignore_index=True)
    merged = merged.drop_duplicates(subset="datetime", keep="first")
    merged = merged.sort_values("datetime").reset_index(drop=True)

    if not merged["datetime"].is_unique:
        raise AssertionError("合併後仍有重複時間戳")
    if not (merged["high"] >= merged["low"]).all():
        raise AssertionError("存在 high < low 的異常列")
    if not ((merged["high"] >= merged["open"]) & (merged["high"] >= merged["close"])).all():
        raise AssertionError("存在 high 小於 open/close 的異常列")
    if not ((merged["low"] <= merged["open"]) & (merged["low"] <= merged["close"])).all():
        raise AssertionError("存在 low 大於 open/close 的異常列")
    if not (merged["volume"] >= 0).all():
        raise AssertionError("存在負成交量")

    # 全資料集掃一次日對日跳空，任何超過 8% 的都要人工確認，不能默默通過。
    # 下面兩個日期已人工核對過收盤價，是真實市場重挫（不是資料錯誤），予以排除：
    #   2008-11-20：金融海嘯尾聲，4261->3903（-8.4%），跟當時 TAIEX 真實低點區（~3955）吻合
    #   2020-03-12：COVID「黑色星期四」全球股災，10827->9876（-8.78%）
    KNOWN_LEGITIMATE_CRASH_DATES = {pd.Timestamp("2008-11-20"), pd.Timestamp("2020-03-12")}
    daily_close = merged.set_index("datetime")["close"].resample("1D").last().dropna()
    daily_gap_pct = daily_close.pct_change().abs() * 100
    suspicious = daily_gap_pct[daily_gap_pct > 8]
    suspicious = suspicious[~suspicious.index.isin(KNOWN_LEGITIMATE_CRASH_DATES)]
    if len(suspicious) > 0:
        raise AssertionError(f"發現 {len(suspicious)} 個單日跳空 > 8% 的可疑時間點，需要人工確認:\n{suspicious}")

    return merged


def main() -> None:
    df = build()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PATH, index=False)

    print(f"rows: {len(df):,}")
    print(f"range: {df['datetime'].min()} ~ {df['datetime'].max()}")
    gaps = df["datetime"].diff().dt.total_seconds().div(60)
    big_gaps = (gaps > 60).sum()
    print(f"分鐘間隔 > 60 分鐘的斷點數（含每日收盤/跨日/例假日屬正常）: {big_gaps:,}")
    print(f"saved: {OUT_PATH} ({OUT_PATH.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
