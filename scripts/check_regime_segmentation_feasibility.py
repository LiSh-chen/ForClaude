"""體制篩選機制可行性檢查：在花力氣建「體制分類器+策略選擇器」兩層系統
之前，先確認用完全跟策略績效無關的獨立指標，能不能把23年歷史切出
足夠多、足夠長的獨立體制週期，撐得起後續的驗證。

用兩個經典、跟任何這次會話測過的策略都無關的獨立體制定義：
1. 多空體制：近月期貨收盤價 vs 200日均線（站上=多頭，跌破=空頭）
2. 波動度體制：ATR(14) 相對 252日(約1年)滾動均值（高於=高波動，低於=低波動）

只看切出來的體制週期本身「夠不夠格」拿去驗證策略，不涉及任何策略績效
——這是體制分類器合法性的必要條件：體制邊界完全由價格/波動度自己的
走勢決定，不是回頭看哪段策略表現好才畫的線。

用法：
    python scripts/check_regime_segmentation_feasibility.py
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = REPO_ROOT / "data" / "raw" / "taifex_tx"
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


def find_episodes(regime_series: pd.Series, dates: pd.Series) -> pd.DataFrame:
    """把regime_series(True/False/NaN)切成連續同值的episode清單。"""
    valid = regime_series.notna()
    r = regime_series[valid].astype(bool)
    d = dates[valid]
    change = r.ne(r.shift())
    group_id = change.cumsum()
    episodes = []
    for gid, idx in pd.Series(group_id.values, index=d.values).groupby(group_id.values):
        pass
    tmp = pd.DataFrame({"date": d.values, "regime": r.values, "group": group_id.values})
    for gid, sub in tmp.groupby("group"):
        episodes.append(dict(
            regime=sub["regime"].iloc[0], start=sub["date"].min(), end=sub["date"].max(),
            n_days=len(sub),
        ))
    return pd.DataFrame(episodes)


def summarize_episodes(episodes: pd.DataFrame, label_true: str, label_false: str) -> None:
    for val, label in [(True, label_true), (False, label_false)]:
        sub = episodes[episodes["regime"] == val]
        print(f"\n  【{label}】週期數: {len(sub)}")
        if len(sub) == 0:
            continue
        print(f"    總交易日數: {sub['n_days'].sum()}")
        print(f"    週期長度(交易日): min={sub['n_days'].min()} median={sub['n_days'].median():.0f} "
              f"max={sub['n_days'].max()}")
        n_long = (sub["n_days"] >= 60).sum()
        print(f"    長度>=60個交易日(約3個月以上)的週期數: {n_long} / {len(sub)}")
        print(f"    週期列表(前10個，依開始日期):")
        print(sub.sort_values("start").head(10)[["start", "end", "n_days"]].to_string(index=False))


def main() -> None:
    print("重建近月OHLC（含高低價，才能算ATR）...")
    ohlc = build_near_month_ohlc()
    print(f"交易日數: {len(ohlc)} ({ohlc['date'].min()} ~ {ohlc['date'].max()})")

    ohlc["ma200"] = ohlc["close"].rolling(200).mean()
    ohlc["bull_regime"] = np.where(ohlc["ma200"].isna(), np.nan, ohlc["close"] > ohlc["ma200"])

    prev_close = ohlc["close"].shift(1)
    tr = pd.concat([
        ohlc["high"] - ohlc["low"],
        (ohlc["high"] - prev_close).abs(),
        (ohlc["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    ohlc["atr14"] = tr.ewm(alpha=1 / 14, adjust=False).mean()
    ohlc["atr_ma252"] = ohlc["atr14"].rolling(252).mean()
    ohlc["high_vol_regime"] = np.where(ohlc["atr_ma252"].isna(), np.nan, ohlc["atr14"] > ohlc["atr_ma252"])

    print("\n" + "=" * 70)
    print("1) 多空體制（收盤價 vs 200日均線）")
    print("=" * 70)
    bull_episodes = find_episodes(ohlc["bull_regime"], ohlc["date"])
    summarize_episodes(bull_episodes, "多頭(收盤>200MA)", "空頭(收盤<200MA)")

    print("\n" + "=" * 70)
    print("2) 波動度體制（ATR14 vs 252日ATR均值）")
    print("=" * 70)
    vol_episodes = find_episodes(ohlc["high_vol_regime"], ohlc["date"])
    summarize_episodes(vol_episodes, "高波動", "低波動")

    print("\n" + "=" * 70)
    print("可行性評估")
    print("=" * 70)
    bull_long = (bull_episodes[bull_episodes["regime"]]["n_days"] >= 60).sum()
    bear_long = (bull_episodes[~bull_episodes["regime"]]["n_days"] >= 60).sum()
    vol_high_long = (vol_episodes[vol_episodes["regime"]]["n_days"] >= 60).sum()
    vol_low_long = (vol_episodes[~vol_episodes["regime"]]["n_days"] >= 60).sum()
    print(f"多頭週期(>=60交易日)數: {bull_long}，空頭週期(>=60交易日)數: {bear_long}")
    print(f"高波動週期(>=60交易日)數: {vol_high_long}，低波動週期(>=60交易日)數: {vol_low_long}")

    if min(bull_long, bear_long) >= 4 and min(vol_high_long, vol_low_long) >= 4:
        print("\n結論：兩種體制定義切出來的長週期數都>=4，有基本的可行性可以往下走。")
    else:
        print("\n結論：至少一種體制定義切出來的長週期數<4，獨立樣本太少，"
              "貿然往下建完整系統風險很高。")


if __name__ == "__main__":
    main()
