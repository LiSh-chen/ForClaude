"""平滑化版體制分類器：對 check_regime_segmentation_feasibility.py 發現的
「簡單均線交叉雜訊很多」問題，加遲滯帶(hysteresis band)解決——多空體制
要收盤價突破均線一定幅度才切換，不是均線一穿越就馬上翻轉，避免在均線
附近來回打結的雜訊週期。

遲滯帶設計：
- 進入多頭：收盤價 > 200日均線 x (1+band)
- 進入空頭：收盤價 < 200日均線 x (1-band)
- 在兩個門檻之間：維持目前體制不變（這是遲滯帶的核心，不是「無體制」）
- 波動度體制同樣邏輯，套用在 ATR14 vs 252日ATR均值上

band參數本身用敏感度掃描（0%/1%/2%/3%/5%），不是憑感覺挑一個就definition
——體制分類器的設計選擇也要透明，不能暗中調到「看起來最好看」的那個。

用法：
    python scripts/build_smoothed_regime_classifier.py
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = REPO_ROOT / "data" / "raw" / "taifex_tx"
OUT_PATH = REPO_ROOT / "data" / "regime_classification_daily.parquet"
EXPIRY_RE = re.compile(r"^\d{6}$")


def build_near_month_ohlc() -> pd.DataFrame:
    files = sorted(RAW_DIR.glob("TX_*.csv"))
    frames = [pd.read_csv(fp, dtype=str, index_col=False) for fp in files]
    all_df = pd.concat(frames, ignore_index=True)
    all_df.columns = [c.strip() for c in all_df.columns]
    d = all_df.copy()
    d["到期月份(週別)"] = d["到期月份(週別)"].str.strip()
    d["交易時段"] = d["交易時段"].str.strip()
    d = d[d["交易時段"] == "一般"]
    d = d[d["到期月份(週別)"].str.match(EXPIRY_RE, na=False)]
    d["date"] = pd.to_datetime(d["交易日期"], format="%Y/%m/%d")

    def to_num(col):
        return pd.to_numeric(d[col].str.replace(",", "", regex=False), errors="coerce")

    d["open"] = to_num("開盤價")
    d["high"] = to_num("最高價")
    d["low"] = to_num("最低價")
    d["close"] = to_num("收盤價")
    d["volume"] = to_num("成交量")
    d = d[d["volume"].fillna(0) > 0]

    rows = []
    for dt, sub in d.groupby("date", sort=True):
        sub = sub.sort_values("到期月份(週別)")
        near = sub.iloc[0]
        rows.append(dict(date=dt, open=near["open"], high=near["high"],
                          low=near["low"], close=near["close"]))
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def hysteresis_regime(value: pd.Series, reference: pd.Series, band: float) -> pd.Series:
    """遲滯帶體制分類：value突破reference*(1+band)才切到True，跌破
    reference*(1-band)才切到False，中間維持前一天的狀態。回傳float
    (1.0/0.0/nan)，nan代表reference還沒有值(暖機期)。"""
    upper = reference * (1 + band)
    lower = reference * (1 - band)
    regime = np.full(len(value), np.nan)
    current = np.nan
    for i in range(len(value)):
        if pd.isna(reference.iloc[i]):
            regime[i] = np.nan
            continue
        v = value.iloc[i]
        if pd.isna(current):
            current = 1.0 if v > reference.iloc[i] else 0.0
        elif v > upper.iloc[i]:
            current = 1.0
        elif v < lower.iloc[i]:
            current = 0.0
        regime[i] = current
    return pd.Series(regime, index=value.index)


def find_episodes(regime_series: pd.Series, dates: pd.Series) -> pd.DataFrame:
    valid = regime_series.notna()
    r = regime_series[valid].astype(bool)
    d = dates[valid]
    group_id = r.ne(r.shift()).cumsum()
    tmp = pd.DataFrame({"date": d.values, "regime": r.values, "group": group_id.values})
    episodes = []
    for gid, sub in tmp.groupby("group"):
        episodes.append(dict(regime=sub["regime"].iloc[0], start=sub["date"].min(),
                              end=sub["date"].max(), n_days=len(sub)))
    return pd.DataFrame(episodes)


def main() -> None:
    print("重建近月OHLC...")
    ohlc = build_near_month_ohlc()
    print(f"交易日數: {len(ohlc)} ({ohlc['date'].min()} ~ {ohlc['date'].max()})\n")

    ohlc["ma200"] = ohlc["close"].rolling(200).mean()
    prev_close = ohlc["close"].shift(1)
    tr = pd.concat([
        ohlc["high"] - ohlc["low"],
        (ohlc["high"] - prev_close).abs(),
        (ohlc["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    ohlc["atr14"] = tr.ewm(alpha=1 / 14, adjust=False).mean()
    ohlc["atr_ma252"] = ohlc["atr14"].rolling(252).mean()

    print("=" * 70)
    print("遲滯帶寬度敏感度掃描：多空體制（收盤 vs 200日均線）")
    print("=" * 70)
    for band in [0.0, 0.01, 0.02, 0.03, 0.05]:
        regime = hysteresis_regime(ohlc["close"], ohlc["ma200"], band)
        episodes = find_episodes(regime, ohlc["date"])
        n_total = len(episodes)
        n_long = (episodes["n_days"] >= 60).sum()
        median_len = episodes["n_days"].median()
        print(f"  band={band*100:.0f}%: 總週期數={n_total}  中位數長度={median_len:.0f}天  "
              f"長週期(>=60天)數={n_long}")

    print("\n" + "=" * 70)
    print("遲滯帶寬度敏感度掃描：波動度體制（ATR14 vs 252日ATR均值）")
    print("=" * 70)
    for band in [0.0, 0.05, 0.10, 0.15, 0.20]:
        regime = hysteresis_regime(ohlc["atr14"], ohlc["atr_ma252"], band)
        episodes = find_episodes(regime, ohlc["date"])
        n_total = len(episodes)
        n_long = (episodes["n_days"] >= 60).sum()
        median_len = episodes["n_days"].median()
        print(f"  band={band*100:.0f}%: 總週期數={n_total}  中位數長度={median_len:.0f}天  "
              f"長週期(>=60天)數={n_long}")

    # 選定版本：多空用2%遲滯帶、波動度用10%遲滯帶（掃描結果裡明顯改善雜訊、
    # 又還沒過度平滑到只剩幾個週期的中間值，不是挑最好看的那個）
    CHOSEN_BULL_BAND = 0.02
    CHOSEN_VOL_BAND = 0.10

    ohlc["bull_regime"] = hysteresis_regime(ohlc["close"], ohlc["ma200"], CHOSEN_BULL_BAND)
    ohlc["high_vol_regime"] = hysteresis_regime(ohlc["atr14"], ohlc["atr_ma252"], CHOSEN_VOL_BAND)

    print(f"\n" + "=" * 70)
    print(f"最終選定版本：多空體制band={CHOSEN_BULL_BAND*100:.0f}%，波動度體制band={CHOSEN_VOL_BAND*100:.0f}%")
    print("=" * 70)
    bull_ep = find_episodes(ohlc["bull_regime"], ohlc["date"])
    print("\n多空體制週期列表:")
    print(bull_ep.sort_values("start").to_string(index=False))

    vol_ep = find_episodes(ohlc["high_vol_regime"], ohlc["date"])
    print("\n波動度體制週期列表:")
    print(vol_ep.sort_values("start").to_string(index=False))

    out = ohlc[["date", "close", "ma200", "bull_regime", "atr14", "atr_ma252", "high_vol_regime"]]
    out.to_parquet(OUT_PATH, index=False)
    print(f"\nsaved: {OUT_PATH}")


if __name__ == "__main__":
    main()
