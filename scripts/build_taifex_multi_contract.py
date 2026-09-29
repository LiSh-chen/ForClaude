"""把 data/raw/taifex_tx/ 逐月原始CSV合併，依「到期月份」排序，對每個交易日
建立近月(nearmonth)/次近月(secondmonth，即遠月，用於算calendar spread)兩條
連續序列，輸出 data/taifex_tx_multi_contract.parquet。

規則（跟 GitHub 上另一份已驗證過的 hermes-skills taifex-futures-history skill
文件描述的規則一致，見這次會話的討論）：
- 只用「一般」(regular)交易時段的資料，不用「盤後」(after-hours/夜盤，2017年
  之後才出現)——避免同一天日盤/夜盤重複計入「當天」的近月/遠月判斷。
- 排除價差對(spread)合約，即到期月份欄位包含'/'的列（例如"200506/200507"）。
- 排除週選擇權類的契約代碼變體（到期月份欄位含非6碼數字的情況，此欄位正常
  應該剛好是YYYYMM 6碼）。
- 每個交易日，依到期月份由小到大排序剩下的合約：最小=近月(nearmonth)，
  次小=次近月(secondmonth)。
- 價格用「結算價」（收盤後由交易所公告的正式結算價，缺值時退回收盤價）。

用法：
    python scripts/build_taifex_multi_contract.py
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = REPO_ROOT / "data" / "raw" / "taifex_tx"
OUT_PATH = REPO_ROOT / "data" / "taifex_tx_multi_contract.parquet"

EXPIRY_RE = re.compile(r"^\d{6}$")


def load_all_raw() -> pd.DataFrame:
    files = sorted(RAW_DIR.glob("TX_*.csv"))
    if not files:
        raise FileNotFoundError(f"找不到原始檔案，先跑 fetch_taifex_tx_raw.py：{RAW_DIR}")
    frames = []
    for fp in files:
        df = pd.read_csv(fp, dtype=str)
        frames.append(df)
    all_df = pd.concat(frames, ignore_index=True)
    all_df.columns = [c.strip() for c in all_df.columns]
    return all_df


def clean(all_df: pd.DataFrame) -> pd.DataFrame:
    d = all_df.copy()
    d["到期月份(週別)"] = d["到期月份(週別)"].str.strip()
    d["交易時段"] = d["交易時段"].str.strip()

    d = d[d["交易時段"] == "一般"]
    d = d[d["到期月份(週別)"].str.match(EXPIRY_RE, na=False)]

    d["date"] = pd.to_datetime(d["交易日期"], format="%Y/%m/%d")
    d["expiry_ym"] = d["到期月份(週別)"]

    def to_num(col: str) -> pd.Series:
        return pd.to_numeric(d[col].str.replace(",", "", regex=False), errors="coerce")

    d["open"] = to_num("開盤價")
    d["high"] = to_num("最高價")
    d["low"] = to_num("最低價")
    d["close"] = to_num("收盤價")
    d["settlement"] = to_num("結算價")
    d["volume"] = to_num("成交量")
    d["open_interest"] = to_num("未沖銷契約數")
    d["price"] = d["settlement"].fillna(d["close"])

    d = d[d["price"].notna() & (d["volume"].fillna(0) > 0)]
    return d[["date", "expiry_ym", "open", "high", "low", "close", "settlement",
              "volume", "open_interest", "price"]]


def build_nearfar(d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dt, sub in d.groupby("date", sort=True):
        sub = sub.sort_values("expiry_ym")
        if len(sub) == 0:
            continue
        near = sub.iloc[0]
        far = sub.iloc[1] if len(sub) > 1 else None
        rows.append(dict(
            date=dt,
            near_expiry=near["expiry_ym"], near_price=near["price"],
            near_volume=near["volume"], near_oi=near["open_interest"],
            far_expiry=far["expiry_ym"] if far is not None else None,
            far_price=far["price"] if far is not None else None,
            far_volume=far["volume"] if far is not None else None,
            far_oi=far["open_interest"] if far is not None else None,
            n_contracts=len(sub),
        ))
    out = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    out["calendar_spread"] = out["near_price"] - out["far_price"]
    return out


def main() -> None:
    print("讀取原始月檔...")
    raw = load_all_raw()
    print(f"原始總列數: {len(raw)}")

    cleaned = clean(raw)
    print(f"清理後列數(一般時段+單式合約+有成交量): {len(cleaned)}")
    print(f"交易日數: {cleaned['date'].nunique()}")
    print(f"日期範圍: {cleaned['date'].min()} ~ {cleaned['date'].max()}")

    nearfar = build_nearfar(cleaned)
    print(f"\n近月/次近月序列交易日數: {len(nearfar)}")
    print(f"有次近月資料的天數: {nearfar['far_price'].notna().sum()} "
          f"({nearfar['far_price'].notna().mean()*100:.1f}%)")
    print(f"\n樣本(前5列):")
    print(nearfar.head().to_string(index=False))
    print(f"\n樣本(後5列):")
    print(nearfar.tail().to_string(index=False))

    print(f"\ncalendar_spread 統計:")
    print(nearfar["calendar_spread"].describe().to_string())

    nearfar.to_parquet(OUT_PATH, index=False)
    print(f"\nsaved: {OUT_PATH}")


if __name__ == "__main__":
    main()
