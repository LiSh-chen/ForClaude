"""三腿策略（og開盤上衝 + lu午盤放空 + re盤中翻多+量能濾網）walk-forward
穩健性補充檢查。

背景：這個策略目前只用過一次固定的IS(2001-2020)/OOS(2021-2023)切分做驗證
（見 scripts/revalidate_three_legs_after_data_fix.py，OOS三腿合併 t=3.59,
n=1644, 中檔成本淨損益+59,147元，通過）。這代表：
1. 「樣本外驗證」目前只驗證過「用2001-2020找到的量能門檻，套用在2021-2023
   這一段特定期間」，沒有看過策略在其他年代的樣本外表現。
2. re腿的量能三分位門檻是從IS資料估計出來的唯一一個「參數」（og、lu兩腿
   進出場時間都是固定的，沒有可調參數）；walk-forward在意的是這個門檻在
   不同的估計窗格下穩不穩定，以及套用到各自的樣本外窗格表現是否一致。

做法：expanding window（訓練窗格從2001年開始逐步往後擴大，每次估計出新的
量能門檻），對應的測試窗格是訓練窗格結束後緊接著的3年，往前滾動：

    訓練 2001-2005 -> 測試 2006-2008
    訓練 2001-2008 -> 測試 2009-2011
    訓練 2001-2011 -> 測試 2012-2014
    訓練 2001-2014 -> 測試 2015-2017
    訓練 2001-2017 -> 測試 2018-2020
    訓練 2001-2020 -> 測試 2021-2023  （=原本那次OOS驗證，這裡只是重新列出來
                                        比對，不是新的資訊）

**這不是在找新的顯著性**：目的是看「量能門檻是否穩定」以及「策略是否在
每一段樣本外窗格都表現持平/沒有崩潰」，而不是把每一段的t值拿來跟顯著性
門檻比、挑最好看的段落報告——這樣做只是把同一份資料切更多刀，變成新的
多重比較問題。所以這裡報告全部窗格的結果，不做任何篩選。

用法：
    python scripts/three_legs_walk_forward_validation.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tw_quant.hammer_signal_backtest import COST_SCENARIOS, apply_costs  # noqa: E402
from tw_quant.lunch_reversal_strategy import backtest as backtest_lunch  # noqa: E402
from tw_quant.opening_rally_strategy import backtest as backtest_opening  # noqa: E402
from tw_quant.technical_indicators import daily_indicators, filter_trades_by_volume  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = REPO_ROOT / "data" / "txf_1min.parquet"
OUT_DIR = REPO_ROOT / "data"

FOLDS = [
    ("2001-01-01", "2006-01-01", "2006-01-01", "2009-01-01"),
    ("2001-01-01", "2009-01-01", "2009-01-01", "2012-01-01"),
    ("2001-01-01", "2012-01-01", "2012-01-01", "2015-01-01"),
    ("2001-01-01", "2015-01-01", "2015-01-01", "2018-01-01"),
    ("2001-01-01", "2018-01-01", "2018-01-01", "2021-01-01"),
    ("2001-01-01", "2021-01-01", "2021-01-01", "2024-01-01"),
]


def tstat(points: pd.Series) -> float:
    n = len(points)
    if n < 2 or points.std(ddof=1) == 0:
        return np.nan
    return points.mean() / (points.std(ddof=1) / np.sqrt(n))


def legs_for(data_df: pd.DataFrame, vol_threshold: float) -> dict[str, pd.DataFrame]:
    open_trades = backtest_opening(data_df)
    lunch_trades = backtest_lunch(data_df)
    short_leg = lunch_trades[lunch_trades["leg"] == "short_lunch_dip"]
    long_leg_raw = lunch_trades[lunch_trades["leg"] == "long_afternoon_rebound"]
    long_leg = filter_trades_by_volume(long_leg_raw, data_df, vol_threshold)
    return dict(opening=open_trades, short_lunch=short_leg, long_rebound=long_leg)


def summarize(trades: pd.DataFrame) -> dict:
    n = len(trades)
    if n == 0:
        return dict(n=0, t_stat=np.nan, mid_net_twd=0.0)
    t = tstat(trades["pnl_points"])
    mid_net = apply_costs(trades, COST_SCENARIOS[1])["net_twd"].sum()
    return dict(n=n, t_stat=t, mid_net_twd=mid_net)


def main() -> None:
    df = pd.read_parquet(DATA_PATH)
    rows = []

    print(f"{'訓練窗格':<22}{'量能門檻':>10}  {'測試窗格':<22}{'n':>6}{'t值':>8}{'中檔淨損益':>14}")
    print("-" * 90)

    for train_start, train_end, test_start, test_end in FOLDS:
        train_df = df[(df["datetime"] >= train_start) & (df["datetime"] < train_end)]
        test_df = df[(df["datetime"] >= test_start) & (df["datetime"] < test_end)]

        train_indicators = daily_indicators(train_df)
        vol_threshold = train_indicators["vol_ratio_lag1"].quantile(2 / 3)

        test_legs = legs_for(test_df, vol_threshold)
        combined = pd.concat(test_legs.values(), ignore_index=True)
        summary = summarize(combined)

        train_label = f"{train_start[:4]}-{int(train_end[:4])-1}"
        test_label = f"{test_start[:4]}-{int(test_end[:4])-1}"
        print(f"{train_label:<22}{vol_threshold:>10.4f}  {test_label:<22}"
              f"{summary['n']:>6}{summary['t_stat']:>8.2f}{summary['mid_net_twd']:>14,.0f}")

        rows.append(dict(
            train_period=train_label, test_period=test_label,
            vol_threshold=vol_threshold, **summary,
        ))

    result = pd.DataFrame(rows)
    result.to_parquet(OUT_DIR / "three_legs_walk_forward_results.parquet", index=False)

    print("\n" + "=" * 70)
    print("量能門檻穩定性（六個訓練窗格重新估計出來的三分位門檻值）：")
    print(result[["train_period", "vol_threshold"]].to_string(index=False))

    print("\n各測試窗格中檔成本淨損益是否都是正的：")
    print(f"  正的窗格數: {(result['mid_net_twd'] > 0).sum()} / {len(result)}")
    print(f"  t值介於: {result['t_stat'].min():.2f} ~ {result['t_stat'].max():.2f}")

    print("\nsaved: three_legs_walk_forward_results.parquet")


if __name__ == "__main__":
    main()
