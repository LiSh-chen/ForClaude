"""合併 TXF（台指期）CrazyIndicator 1 分鐘 K 棒原始 CSV（存在 7z 內），輸出成
單一乾淨、依時間排序、無重複時間戳的 parquet 檔（data/txf_1min.parquet）。

資料來源涵蓋四個 7z 檔案，範圍互有重疊，合併規則：
- TXF20010101_20210731_Fix_Gap: 2001-01-02 ~ 2021-07-30，作者標註「補齊
  缺漏分鐘」的修正版，範圍內以此檔為準（主檔）。
- TXF20210101_20231231: 2020-12-31 ~ 2023-12-29，只取主檔結束時間點之後的
  部分接續進來（延伸檔）。
- 另外兩個檔案（2001~2010 / 2011~2020）的時間範圍已被主檔涵蓋，是主檔修正
  前的原始版本，不參與合併，僅留作對照。

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

PRIMARY_7Z = RAW_DIR / "TXF20010101_20210731_Fix_Gap(CrazyIndicator.pixnet.net).7z"
EXTEND_7Z = RAW_DIR / "TXF20210101_20231231(CrazyIndicator.pixnet.net).7z"


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
    return df[["datetime", "open", "high", "low", "close", "volume"]]


def build() -> pd.DataFrame:
    primary = _read_csv_from_7z(PRIMARY_7Z)
    extend = _read_csv_from_7z(EXTEND_7Z)

    cutover = primary["datetime"].max()
    extend_new = extend[extend["datetime"] > cutover]

    merged = pd.concat([primary, extend_new], ignore_index=True)
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
