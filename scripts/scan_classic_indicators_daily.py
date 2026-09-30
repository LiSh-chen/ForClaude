"""補「其他技術指標組合」的缺口：這次會話之前只測過ATR(體制濾網)跟VOL
(體制濾網替代方案)，從沒把RSI/KD/MACD/BIAS/CCI/威廉指標(WM%R)當成方向性
訊號本身測試過(使用者列出的APP指標清單：VOL/KD/MACD/RSI/DMI/KDJ/BIAS/
PSY/威廉/OBV/MTM/BBI/AR/BR/VR/CCI/ROC，這裡選六個最常見的獨立測)。

用近月期貨日線(2001-2020重建的OHLC，跟build_smoothed_regime_classifier.py
同一份)，每個指標都用「極端分位(z-score或標準門檻)->次日收盤對收盤報酬」
的方法論，跟這次會話測put/call比率、跳空回補用的是同一套(先IS篩選、
只對IS顯著的才花OOS)。

測試清單：
1. RSI14 超賣(<30)/超買(>70)
2. KD(9,3,3) K值超賣(<20)/超買(>80)
3. MACD(12,26,9) 柱狀圖翻正/翻負(零軸交叉)
4. BIAS20 (收盤價乖離20日均線) 正負2個標準差
5. CCI20 超賣(<-100)/超買(>100)
6. 威廉指標WM%R(14) 超賣(<-80)/超買(>-20)

不開參數網格搜尋(每個指標都用市場慣例的標準參數，不調參數去配合結果)，
只在IS(2001-2020)篩選，任何指標若|t|>=3且方向合理才進一步花OOS額度。

用法：
    python scripts/scan_classic_indicators_daily.py
"""

from __future__ import annotations

import sys
import re
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = REPO_ROOT / "data" / "raw" / "taifex_tx"
IS_END = "2021-01-01"
OOS_END = "2024-01-01"
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


def tstat(x: pd.Series) -> tuple[float, int, float]:
    x = x.dropna()
    n = len(x)
    if n < 2 or x.std(ddof=1) == 0:
        return np.nan, n, np.nan
    return x.mean() / (x.std(ddof=1) / np.sqrt(n)), n, x.mean()


def add_indicators(d: pd.DataFrame) -> pd.DataFrame:
    close, high, low = d["close"], d["high"], d["low"]

    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    rs = gain / loss
    d["rsi14"] = 100 - 100 / (1 + rs)

    low9 = low.rolling(9).min()
    high9 = high.rolling(9).max()
    rsv = (close - low9) / (high9 - low9) * 100
    d["k"] = rsv.ewm(alpha=1 / 3, adjust=False).mean()
    d["d"] = d["k"].ewm(alpha=1 / 3, adjust=False).mean()

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    d["macd_hist"] = macd - signal

    ma20 = close.rolling(20).mean()
    d["bias20"] = (close - ma20) / ma20 * 100

    tp = (high + low + close) / 3
    tp_ma = tp.rolling(20).mean()
    tp_md = tp.rolling(20).apply(lambda x: np.abs(x - x.mean()).mean(), raw=False)
    d["cci20"] = (tp - tp_ma) / (0.015 * tp_md)

    high14 = high.rolling(14).max()
    low14 = low.rolling(14).min()
    d["wmr14"] = (high14 - close) / (high14 - low14) * -100

    d["ret_next"] = (close.shift(-1) / close - 1) * 100
    return d


def report(label: str, is_sub: pd.Series, direction_note: str) -> float:
    t, n, m = tstat(is_sub)
    print(f"  {label}: n={n} mean_next_ret={m:.4f}% t={t:.2f}  ({direction_note})")
    return t


def main() -> None:
    print("重建近月OHLC...")
    ohlc = build_near_month_ohlc()
    d = add_indicators(ohlc)
    is_d = d[d["date"] < IS_END]
    print(f"IS樣本數: {len(is_d)} ({is_d['date'].min().date()} ~ {is_d['date'].max().date()})\n")

    print("=" * 78)
    print("IS(2001-2020) 六個經典技術指標極端值 vs 次日報酬")
    print("=" * 78)

    print("\n[1] RSI14")
    report("超賣(<30)", is_d.loc[is_d["rsi14"] < 30, "ret_next"], "測試反轉:超賣是否反彈")
    report("超買(>70)", is_d.loc[is_d["rsi14"] > 70, "ret_next"], "測試反轉:超買是否回落")

    print("\n[2] KD(9,3,3) K值")
    report("超賣(K<20)", is_d.loc[is_d["k"] < 20, "ret_next"], "測試反轉:超賣是否反彈")
    report("超買(K>80)", is_d.loc[is_d["k"] > 80, "ret_next"], "測試反轉:超買是否回落")

    print("\n[3] MACD柱狀圖零軸交叉(前一天負、今天翻正=金叉；反之=死叉)")
    macd_prev = is_d["macd_hist"].shift(1)
    golden = is_d.loc[(macd_prev < 0) & (is_d["macd_hist"] >= 0), "ret_next"]
    dead = is_d.loc[(macd_prev > 0) & (is_d["macd_hist"] <= 0), "ret_next"]
    report("翻正(金叉)", golden, "測試延續:翻正是否續漲")
    report("翻負(死叉)", dead, "測試延續:翻負是否續跌")

    print("\n[4] BIAS20 (收盤乖離20日均線)")
    bias_std = is_d["bias20"].std()
    report(f"正乖離>2std({2*bias_std:.2f}%)", is_d.loc[is_d["bias20"] > 2 * bias_std, "ret_next"], "測試反轉:乖離過大是否修正")
    report(f"負乖離<-2std", is_d.loc[is_d["bias20"] < -2 * bias_std, "ret_next"], "測試反轉:乖離過大是否修正")

    print("\n[5] CCI20")
    report("超賣(<-100)", is_d.loc[is_d["cci20"] < -100, "ret_next"], "測試反轉:超賣是否反彈")
    report("超買(>100)", is_d.loc[is_d["cci20"] > 100, "ret_next"], "測試反轉:超買是否回落")

    print("\n[6] 威廉指標WM%R(14)")
    report("超賣(<-80)", is_d.loc[is_d["wmr14"] < -80, "ret_next"], "測試反轉:超賣是否反彈")
    report("超買(>-20)", is_d.loc[is_d["wmr14"] > -20, "ret_next"], "測試反轉:超買是否回落")

    print("\n" + "=" * 78)
    print("對照：全樣本無條件次日報酬")
    print("=" * 78)
    report("全樣本", is_d["ret_next"], "基準對照")


if __name__ == "__main__":
    main()
