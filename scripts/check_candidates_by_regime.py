"""把這次會話找到的候選，套上平滑化版體制分類器(bull_regime/high_vol_regime,
見 build_smoothed_regime_classifier.py)分開檢視，看表現有沒有系統性的
體制差異——這是驗證「體制篩選」這個想法本身有沒有用的關鍵一步。

**紀律**：體制邊界已經在上一步用跟策略績效無關的獨立指標(200日均線、
ATR)固定好了，這裡只是把候選的交易/訊號套進去分組看，不會回頭調整
體制定義去配合某個候選的績效——那樣就違反了體制分類器要獨立於策略
績效的原則。

檢視兩個候選：
1. 三腿策略（唯一通過完整IS/子區間/OOS驗證的候選，但walk-forward顯示
   對成本敏感）——分bull/bear、high/low vol，用IS(2001-2020)。
2. put/call比率候選（OOS(2021-2023)失敗、2024-2026補充驗證極強的那個）
   ——分bull/bear、high/low vol，看2021-2023失敗是不是剛好落在特定體制。

用法：
    python scripts/check_candidates_by_regime.py
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
TXO_PATH = REPO_ROOT / "data" / "taifex_txo_daily_putcall.parquet"
TX_PATH = REPO_ROOT / "data" / "taifex_tx_multi_contract.parquet"

IS_CUTOFF = "2021-01-01"


def tstat(x: pd.Series) -> tuple[float, int, float]:
    x = x.dropna()
    n = len(x)
    if n < 2 or x.std(ddof=1) == 0:
        return np.nan, n, np.nan
    return x.mean() / (x.std(ddof=1) / np.sqrt(n)), n, x.mean()


def three_legs_by_regime() -> None:
    print("=" * 70)
    print("1) 三腿策略 x 體制（IS 2001-2020，逐筆交易pnl_points的t值）")
    print("=" * 70)
    df = pd.read_parquet(DATA_PATH)
    is_df = df[df["datetime"] < IS_CUTOFF]

    is_indicators = daily_indicators(is_df)
    vol_threshold = is_indicators["vol_ratio_lag1"].quantile(2 / 3)

    open_trades = backtest_opening(is_df)
    lunch_trades = backtest_lunch(is_df)
    short_leg = lunch_trades[lunch_trades["leg"] == "short_lunch_dip"]
    long_leg_raw = lunch_trades[lunch_trades["leg"] == "long_afternoon_rebound"]
    long_leg = filter_trades_by_volume(long_leg_raw, is_df, vol_threshold)
    combined = pd.concat([open_trades, short_leg, long_leg], ignore_index=True)
    combined["trading_date"] = pd.to_datetime(combined["trading_date"])

    regime = pd.read_parquet(REGIME_PATH)[["date", "bull_regime", "high_vol_regime"]]
    merged = combined.merge(regime, left_on="trading_date", right_on="date", how="left")

    for col, labels in [("bull_regime", ("空頭", "多頭")), ("high_vol_regime", ("低波動", "高波動"))]:
        print(f"\n--- 依 {col} 分組 ---")
        for val, label in [(0.0, labels[0]), (1.0, labels[1])]:
            sub = merged[merged[col] == val]
            t, n, m = tstat(sub["pnl_points"])
            mid_net = apply_costs(sub, COST_SCENARIOS[1])["net_twd"].sum() if n > 0 else 0
            print(f"  {label}: n={n} t={t:.2f} mean_pnl={m:.2f}點 中檔成本淨損益={mid_net:,.0f}")


def putcall_by_regime() -> None:
    print("\n" + "=" * 70)
    print("2) put/call比率候選 x 體制（pc_oi_ratio_z高分位, N=10天前瞻報酬）")
    print("=" * 70)
    txo = pd.read_parquet(TXO_PATH)
    tx = pd.read_parquet(TX_PATH)[["date", "near_price"]]
    df = txo.merge(tx, on="date", how="inner").sort_values("date").reset_index(drop=True)

    df["pc_oi_ratio"] = df["put_oi"] / df["call_oi"]
    W = 60
    roll_mean = df["pc_oi_ratio"].rolling(W).mean().shift(1)
    roll_std = df["pc_oi_ratio"].rolling(W).std(ddof=1).shift(1)
    df["z"] = (df["pc_oi_ratio"].shift(1) - roll_mean) / roll_std
    N = 10
    df["fwd_ret"] = (df["near_price"].shift(-N) / df["near_price"] - 1) * 100

    regime = pd.read_parquet(REGIME_PATH)[["date", "bull_regime", "high_vol_regime"]]
    merged = df.merge(regime, on="date", how="left")
    high = merged[merged["z"] >= 1.0]

    for col, labels in [("bull_regime", ("空頭", "多頭")), ("high_vol_regime", ("低波動", "高波動"))]:
        print(f"\n--- 依 {col} 分組（全樣本2001-2026，高分位z>=1組）---")
        for val, label in [(0.0, labels[0]), (1.0, labels[1])]:
            sub = high[high[col] == val]
            t, n, m = tstat(sub["fwd_ret"])
            print(f"  {label}: n={n} t={t:.2f} mean_fwd_ret={m:.3f}%")

    # 特別檢查：2021-2023(OOS失敗窗格)是在哪個體制
    print("\n--- 特別檢查：2021-2023(OOS失敗窗格) 的體制分布 ---")
    oos = merged[(merged["date"] >= "2021-01-01") & (merged["date"] < "2024-01-01")]
    print(oos["bull_regime"].value_counts(dropna=False))
    print(oos["high_vol_regime"].value_counts(dropna=False))

    print("\n--- 特別檢查：2024-2026(補充驗證極強窗格) 的體制分布 ---")
    fresh = merged[merged["date"] >= "2024-01-01"]
    print(fresh["bull_regime"].value_counts(dropna=False))
    print(fresh["high_vol_regime"].value_counts(dropna=False))


if __name__ == "__main__":
    three_legs_by_regime()
    putcall_by_regime()
