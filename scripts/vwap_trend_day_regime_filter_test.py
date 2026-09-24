"""VWAP趨勢日候選(frac=0.90)衰退急救嘗試：200日均線趨勢對齊濾網。

背景：risk_analysis已經發現這個策略的統計顯著性幾乎全部來自2001-2010，
2011-2023(含OOS)合計是淨虧損。一個自然的假設是「策略本身沒壞，是市場
regime變了——策略需要有明確趨勢的市場才有效，2011年後TXF可能變得更
盤整/效率化，只是剛好蓋掉了策略在真正趨勢期間依然有效的部分」。

測試方法：用最標準、最不需要調參的趨勢regime定義（200日均線，市場上
最通用的長期趨勢判斷慣例，刻意不做網格搜尋去挑「最好」的均線天數，
避免又引入一次多重比較風險）——前一天收盤在200日均線之上視為多頭
regime，之下視為空頭regime。「濾網對齊」＝多單只在多頭regime進場、
空單只在空頭regime進場，理論上如果decay是regime造成的，對齊regime的
子集應該表現明顯優於不對齊的子集，且能讓2011-2023、OOS的t值回升。

這是對已經算好、已經鎖定的全歷史交易序列（frac=0.90，見
vwap_trend_day_risk_analysis.py）做post-hoc篩選分析，不是重新調整
策略本身的參數或重新做OOS驗證。

用法：
    python scripts/vwap_trend_day_regime_filter_test.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tw_quant.technical_indicators import build_daily_bars  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"


def tstat(x: pd.Series) -> float:
    n = len(x)
    if n < 2 or x.std(ddof=1) == 0:
        return np.nan
    return x.mean() / (x.std(ddof=1) / np.sqrt(n))


def report(label: str, sub: pd.DataFrame) -> None:
    if sub.empty:
        print(f"{label}: n=0")
        return
    print(f"{label}: n={len(sub)}, t={tstat(sub['pnl_points']):.3f}, "
          f"net_sum={sub['net_twd'].sum():,.0f}, win_rate={(sub['net_twd']>0).mean():.3f}")


def main() -> None:
    df = pd.read_parquet(DATA_DIR / "txf_1min.parquet")
    daily = build_daily_bars(df)
    daily["ma200"] = daily["close"].shift(1).rolling(200).mean()
    daily["regime_long"] = daily["close"].shift(1) > daily["ma200"]
    regime_map = dict(zip(daily["date"].dt.date, daily["regime_long"]))

    trades = pd.read_parquet(DATA_DIR / "trend_day_relaxed_full_history_trades.parquet")
    trades["trading_date_only"] = pd.to_datetime(trades["trading_date"]).dt.date
    trades["regime_long"] = trades["trading_date_only"].map(regime_map)
    trades = trades.dropna(subset=["regime_long"])
    trades["aligned"] = ((trades["direction"] == "long") & trades["regime_long"]) | \
                         ((trades["direction"] == "short") & ~trades["regime_long"])

    print("=" * 70)
    print("1) 全歷史：regime對齊 vs 不對齊 子集比較")
    print("=" * 70)
    report("全部(有regime資料)", trades)
    report("順勢regime對齊", trades[trades["aligned"]])
    report("逆regime", trades[~trades["aligned"]])

    print("\n" + "=" * 70)
    print("2) 2011-2023（含OOS），加regime濾網前後對比")
    print("=" * 70)
    recent = trades[pd.to_datetime(trades["trading_date"]) >= "2011-01-01"]
    report("2011-2023全部", recent)
    report("2011-2023+regime對齊", recent[recent["aligned"]])

    print("\n" + "=" * 70)
    print("3) OOS（2021-2023），加regime濾網前後對比")
    print("=" * 70)
    oos = trades[pd.to_datetime(trades["trading_date"]) >= "2021-01-01"]
    report("OOS全部", oos)
    report("OOS+regime對齊", oos[oos["aligned"]])


if __name__ == "__main__":
    main()
