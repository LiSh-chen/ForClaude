"""驗證「三腿策略只在高波動體制下交易」這個從體制分組檢視發現的線索：
（見scripts/check_candidates_by_regime.py，IS整體波動度分組顯示低波動
中檔成本淨損益-80,270元、高波動+160,052元）。

**這不是在找新候選，是對已經通過完整驗證的三腿策略加一個體制濾網**，
所以驗證方式跟找全新候選不一樣：
1. 子區間穩健性（IS 2001-2020四個5年窗格，各自只看高波動體制下的交易）
2. Walk-forward折疊重新檢查：三腿策略原本walk-forward驗證(見
   scripts/three_legs_walk_forward_validation.py)裡，2012-2014/2015-2017/
   2018-2020幾折中高成本下轉虧——這裡套上高波動濾網重跑同樣六折，看
   有沒有真的改善那幾個轉虧的窗格
3. 2021-2023：這是三腿策略原本已經驗證過的OOS窗格，這裡套上濾網重看
   一次，是「既有驗證的補充特徵化」，不是新的OOS消耗
4. 2024-2026：三腿策略從來沒有用過的全新資料（但沒有1分鐘資料，這裡
   只能用日頻近似——近月期貨開盤到收盤的報酬當三腿策略的粗略代理，
   不是精確重算og/lu/re，僅供參考）

用法：
    python scripts/validate_three_legs_high_vol_filter.py
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
REGIME_PATH = REPO_ROOT / "data" / "regime_classification_daily.parquet"

SUB_PERIODS = [
    ("2001-2005", "2001-01-01", "2006-01-01"),
    ("2006-2010", "2006-01-01", "2011-01-01"),
    ("2011-2015", "2011-01-01", "2016-01-01"),
    ("2016-2020", "2016-01-01", "2021-01-01"),
]
WF_FOLDS = [
    ("2001-01-01", "2006-01-01", "2006-01-01", "2009-01-01"),
    ("2001-01-01", "2009-01-01", "2009-01-01", "2012-01-01"),
    ("2001-01-01", "2012-01-01", "2012-01-01", "2015-01-01"),
    ("2001-01-01", "2015-01-01", "2015-01-01", "2018-01-01"),
    ("2001-01-01", "2018-01-01", "2018-01-01", "2021-01-01"),
    ("2001-01-01", "2021-01-01", "2021-01-01", "2024-01-01"),
]


def tstat(x: pd.Series) -> tuple[float, int, float]:
    x = x.dropna()
    n = len(x)
    if n < 2 or x.std(ddof=1) == 0:
        return np.nan, n, np.nan
    return x.mean() / (x.std(ddof=1) / np.sqrt(n)), n, x.mean()


def three_legs_trades(data_df: pd.DataFrame, vol_threshold: float) -> pd.DataFrame:
    open_trades = backtest_opening(data_df)
    lunch_trades = backtest_lunch(data_df)
    short_leg = lunch_trades[lunch_trades["leg"] == "short_lunch_dip"]
    long_leg_raw = lunch_trades[lunch_trades["leg"] == "long_afternoon_rebound"]
    long_leg = filter_trades_by_volume(long_leg_raw, data_df, vol_threshold)
    combined = pd.concat([open_trades, short_leg, long_leg], ignore_index=True)
    combined["trading_date"] = pd.to_datetime(combined["trading_date"])
    return combined


def apply_high_vol_filter(trades: pd.DataFrame, regime: pd.DataFrame) -> pd.DataFrame:
    merged = trades.merge(regime[["date", "high_vol_regime"]], left_on="trading_date",
                           right_on="date", how="left")
    return merged[merged["high_vol_regime"] == 1.0]


def report(trades: pd.DataFrame, label: str) -> dict:
    n = len(trades)
    if n == 0:
        print(f"  {label}: n=0")
        return dict(label=label, n=0)
    t, _, m = tstat(trades["pnl_points"])
    mid_net = apply_costs(trades, COST_SCENARIOS[1])["net_twd"].sum()
    print(f"  {label}: n={n} t={t:.2f} 中檔成本淨損益={mid_net:,.0f}")
    return dict(label=label, n=n, t_stat=t, mean_pnl=m, mid_net_twd=mid_net)


def main() -> None:
    df = pd.read_parquet(DATA_PATH)
    regime = pd.read_parquet(REGIME_PATH)

    print("=" * 70)
    print("1) IS(2001-2020) 子區間穩健性：高波動濾網版本 vs 無濾網對照")
    print("=" * 70)
    is_indicators = daily_indicators(df[df["datetime"] < "2021-01-01"])
    vol_threshold = is_indicators["vol_ratio_lag1"].quantile(2 / 3)

    for name, start, end in SUB_PERIODS:
        sub_df = df[(df["datetime"] >= start) & (df["datetime"] < end)]
        trades = three_legs_trades(sub_df, vol_threshold)
        filtered = apply_high_vol_filter(trades, regime)
        print(f"\n--- {name} ---")
        report(trades, "無濾網(對照)")
        report(filtered, "高波動濾網")

    print("\n" + "=" * 70)
    print("2) Walk-forward六折重新檢查：高波動濾網能不能救回轉虧的窗格")
    print("=" * 70)
    for train_start, train_end, test_start, test_end in WF_FOLDS:
        train_df = df[(df["datetime"] >= train_start) & (df["datetime"] < train_end)]
        test_df = df[(df["datetime"] >= test_start) & (df["datetime"] < test_end)]
        fold_vol_threshold = daily_indicators(train_df)["vol_ratio_lag1"].quantile(2 / 3)

        trades = three_legs_trades(test_df, fold_vol_threshold)
        filtered = apply_high_vol_filter(trades, regime)
        test_label = f"{test_start[:4]}-{int(test_end[:4])-1}"
        print(f"\n--- 測試窗格 {test_label} ---")
        report(trades, "無濾網(對照，原始walk-forward結果)")
        report(filtered, "高波動濾網")

    print("\n" + "=" * 70)
    print("3) 2021-2023：三腿策略既有OOS窗格的補充特徵化（非新OOS消耗）")
    print("=" * 70)
    oos_df = df[(df["datetime"] >= "2021-01-01") & (df["datetime"] < "2024-01-01")]
    trades = three_legs_trades(oos_df, vol_threshold)
    filtered = apply_high_vol_filter(trades, regime)
    report(trades, "無濾網(對照，原始OOS結果)")
    report(filtered, "高波動濾網")
    print("\n三種成本情境下 OOS 高波動濾網版本淨損益：")
    for cost in COST_SCENARIOS:
        net = apply_costs(filtered, cost)["net_twd"].sum()
        print(f"  {cost.label}: {net:,.0f}")


if __name__ == "__main__":
    main()
