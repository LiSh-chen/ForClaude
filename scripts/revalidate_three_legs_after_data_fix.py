"""資料修正（見 build_txf_1min.py，排除損壞的 Fix_Gap 來源檔）後，重新驗證
之前三個已通過 OOS 檢驗的時段策略：

1. 開盤上衝（08:45 進場 -> 09:00 出場，多）
2. 午盤放空（12:00 進場 -> 12:30 出場，空）
3. 盤中翻多 + 成交量濾網（12:30 進場 -> 13:00 出場，多，昨量比率前 1/3 分位）

損壞資料主要影響 2011-2020（偏差最大），OOS（2021+，主要落在乾淨的延伸檔）
基本沒受影響，但 IS 的子區間穩健性數字、樣本內算出的成交量門檻，都需要
重新確認。跑法：
- 子區間穩健性：四個 5 年窗格個別算 t 值，跟之前的數字（開盤上衝 IS 全體
  t=6.79、四個子區間 t>=2.12）比對，確認結論沒有因為資料修正而改變。
- 成交量門檻：用修正後的 IS 資料重新算三分位切點（舊門檻是用壞資料算出來
  的，數字本身可能已經跟著資料偏差跑掉，需要重新算一次再套用到 OOS）。
- OOS：三腿分別、以及合併後，只跑一次，不因為結果不理想回頭調整。

用法：
    python scripts/revalidate_three_legs_after_data_fix.py
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

IS_CUTOFF = "2021-01-01"
SUB_PERIODS = [
    ("2001-2005", "2001-01-01", "2006-01-01"),
    ("2006-2010", "2006-01-01", "2011-01-01"),
    ("2011-2015", "2011-01-01", "2016-01-01"),
    ("2016-2020", "2016-01-01", "2021-01-01"),
]


def tstat(points: pd.Series) -> float:
    n = len(points)
    if n < 2 or points.std(ddof=1) == 0:
        return np.nan
    return points.mean() / (points.std(ddof=1) / np.sqrt(n))


def report(trades: pd.DataFrame, label: str) -> dict:
    n = len(trades)
    if n == 0:
        print(f"  {label}: n=0")
        return dict(label=label, n=0)
    t = tstat(trades["pnl_points"])
    mid_net = apply_costs(trades, COST_SCENARIOS[1])["net_twd"].sum()
    print(f"  {label}: n={n} t={t:.2f} 中檔成本淨損益={mid_net:,.0f}")
    return dict(label=label, n=n, t_stat=t, mid_net_twd=mid_net)


def legs_for(data_df: pd.DataFrame, vol_threshold: float) -> dict[str, pd.DataFrame]:
    open_trades = backtest_opening(data_df)
    lunch_trades = backtest_lunch(data_df)
    short_leg = lunch_trades[lunch_trades["leg"] == "short_lunch_dip"]
    long_leg_raw = lunch_trades[lunch_trades["leg"] == "long_afternoon_rebound"]
    long_leg = filter_trades_by_volume(long_leg_raw, data_df, vol_threshold)
    return dict(opening=open_trades, short_lunch=short_leg, long_rebound=long_leg)


def main() -> None:
    df = pd.read_parquet(DATA_PATH)
    is_df = df[df["datetime"] < IS_CUTOFF]
    oos_df = df[df["datetime"] >= IS_CUTOFF]

    is_indicators = daily_indicators(is_df)
    vol_threshold = is_indicators["vol_ratio_lag1"].quantile(2 / 3)
    print(f"成交量比率門檻（修正後 IS 資料重新算）: {vol_threshold:.4f}\n")

    print("=" * 70)
    print("1) IS（2001-2020）整體")
    print("=" * 70)
    is_legs = legs_for(is_df, vol_threshold)
    is_rows = [report(t, name) for name, t in is_legs.items()]
    is_combined = pd.concat(is_legs.values(), ignore_index=True)
    is_rows.append(report(is_combined, "三腿合併"))

    print("\n" + "=" * 70)
    print("2) IS 子區間穩健性（四個 5 年窗格，各腿分開）")
    print("=" * 70)
    sub_rows = []
    for name, start, end in SUB_PERIODS:
        sub_df = df[(df["datetime"] >= start) & (df["datetime"] < end)]
        print(f"\n--- {name} ---")
        sub_legs = legs_for(sub_df, vol_threshold)
        for leg_name, trades in sub_legs.items():
            row = report(trades, leg_name)
            row["period"] = name
            row["leg"] = leg_name
            sub_rows.append(row)
        combined = pd.concat(sub_legs.values(), ignore_index=True)
        row = report(combined, "三腿合併")
        row["period"] = name
        row["leg"] = "combined"
        sub_rows.append(row)

    print("\n" + "=" * 70)
    print("3) OOS（2021-2023）唯一一次驗證")
    print("=" * 70)
    oos_legs = legs_for(oos_df, vol_threshold)
    oos_rows = [report(t, name) for name, t in oos_legs.items()]
    oos_combined = pd.concat(oos_legs.values(), ignore_index=True)
    oos_rows.append(report(oos_combined, "三腿合併"))

    print("\n三種成本情境下 OOS 三腿合併淨損益：")
    for cost in COST_SCENARIOS:
        net = apply_costs(oos_combined, cost)["net_twd"].sum()
        print(f"  {cost.label}: {net:,.0f}")

    pd.DataFrame(is_rows).to_parquet(OUT_DIR / "three_legs_revalidation_is.parquet", index=False)
    pd.DataFrame(sub_rows).to_parquet(OUT_DIR / "three_legs_revalidation_subperiods.parquet", index=False)
    pd.DataFrame(oos_rows).to_parquet(OUT_DIR / "three_legs_revalidation_oos.parquet", index=False)
    print("\nsaved: three_legs_revalidation_{is,subperiods,oos}.parquet")


if __name__ == "__main__":
    main()
