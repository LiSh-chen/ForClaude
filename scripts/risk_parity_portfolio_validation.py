"""風險平價組合驗證：VWAP趨勢日(frac=0.90) + 壓力支撐順勢突破。

上一步（candidate_correlation_check.py）發現兩策略OOS月度相關係數僅0.025
（近乎獨立），但等權重（各1口）組合後t值沒有改善，因為SR突破的月度損益
波動是VWAP的6倍多，組合風險被SR突破主導。

這一版用「風險平價」權重重新組合：口數比例 = 兩策略月度損益標準差的反比，
讓兩者對組合波動的貢獻相等。關鍵方法論要求——維持這次會話一貫的IS/OOS
紀律：**權重必須只用IS(2001-2020)資料算出來，不能用OOS(2021-2023)的
波動數字去反推權重**（用OOS資料選權重等於用答案去調參數，會讓後面的OOS
t值檢定失去意義）。所以流程是：

1. 在IS資料上分別跑兩策略，算IS月度損益標準差比例 -> 固定住口數權重
2. 把這個「用IS資料鎖定、對OOS未知」的權重，套用在OOS月度損益上
3. 算組合後的OOS t值，跟兩個策略個別的OOS t值比較

這是對組合可行性的single-shot驗證，不回頭調整權重。

用法：
    python scripts/risk_parity_portfolio_validation.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tw_quant.hammer_signal_backtest import COST_SCENARIOS, apply_costs  # noqa: E402
from tw_quant.support_resistance_fade_strategy import SupportResistanceFadeConfig, backtest as backtest_sr  # noqa: E402
from tw_quant.trend_day_strategy import TrendDayConfig, backtest as backtest_trend_day  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
MID_COST = COST_SCENARIOS[1]
IS_CUTOFF = "2021-01-01"

VWAP_CFG = TrendDayConfig(min_dominant_side_fraction=0.90)
SR_CFG = SupportResistanceFadeConfig(
    direction_mode="breakout", channel_window=20, stop_atr_mult=1.0,
    min_range_pct=2.0, trend_slope_threshold_pct=5.0,
)


def tstat(x: pd.Series) -> float:
    n = len(x)
    if n < 2 or x.std(ddof=1) == 0:
        return np.nan
    return x.mean() / (x.std(ddof=1) / np.sqrt(n))


def monthly_net_series(trades: pd.DataFrame, date_col: str, start: str, end: str) -> pd.Series:
    full_range = pd.date_range(start, end, freq="D")
    daily = pd.Series(0.0, index=full_range)
    if trades.empty:
        return daily.resample("ME").sum()
    t = apply_costs(trades, MID_COST)
    dates = pd.to_datetime(trades[date_col])
    for d, net in zip(dates, t["net_twd"]):
        if d in daily.index:
            daily[d] += net
    return daily.resample("ME").sum()


def main() -> None:
    df = pd.read_parquet(DATA_DIR / "txf_1min.parquet")
    is_df = df[df["datetime"] < IS_CUTOFF]

    print("=" * 70)
    print("1) 在IS資料上重跑兩策略，算IS月度損益標準差 -> 鎖定風險平價權重")
    print("=" * 70)
    vwap_is_trades = backtest_trend_day(is_df, VWAP_CFG)
    sr_is_trades = pd.read_parquet(DATA_DIR / "sr_breakout_candidate_is_trades.parquet")

    vwap_is_m = monthly_net_series(vwap_is_trades, "trading_date", "2001-01-01", "2020-12-31")
    sr_is_m = monthly_net_series(sr_is_trades, "exit_date", "2001-01-01", "2020-12-31")

    vwap_is_std = vwap_is_m.std()
    sr_is_std = sr_is_m.std()
    print(f"VWAP IS月度損益標準差: {vwap_is_std:,.0f}")
    print(f"SR突破 IS月度損益標準差: {sr_is_std:,.0f}")

    # 風險平價：口數比例 = 標準差反比，讓兩者對組合波動貢獻相等
    w_vwap = sr_is_std / vwap_is_std  # VWAP波動小，配較多口數
    w_sr = 1.0
    print(f"\n鎖定口數權重（僅用IS資料，套用到OOS不再調整）：VWAP={w_vwap:.2f}口 : SR突破={w_sr:.2f}口")

    print("\n" + "=" * 70)
    print("2) 把IS鎖定的權重套用到OOS，計算組合後t值")
    print("=" * 70)
    vwap_oos_trades = pd.read_parquet(DATA_DIR / "trend_day_relaxed_oos_trades.parquet")
    sr_oos_trades = pd.read_parquet(DATA_DIR / "sr_breakout_candidate_oos_trades.parquet")

    vwap_oos_m = monthly_net_series(vwap_oos_trades, "trading_date", "2021-01-01", "2023-12-31")
    sr_oos_m = monthly_net_series(sr_oos_trades, "exit_date", "2021-01-01", "2023-12-31")

    combined_oos_m = vwap_oos_m * w_vwap + sr_oos_m * w_sr

    print(f"VWAP單獨 OOS：月度t={tstat(vwap_oos_m):.3f}, 月度sum={vwap_oos_m.sum():,.0f}")
    print(f"SR突破單獨 OOS：月度t={tstat(sr_oos_m):.3f}, 月度sum={sr_oos_m.sum():,.0f}")
    print(f"風險平價組合({w_vwap:.2f}口VWAP + 1口SR突破) OOS：月度t={tstat(combined_oos_m):.3f}, "
          f"月度sum={combined_oos_m.sum():,.0f}")

    combined_std = combined_oos_m.std()
    naive_sum_std = vwap_oos_m.std() * w_vwap + sr_oos_m.std() * w_sr
    print(f"\n組合月度標準差={combined_std:,.0f} vs 兩者標準差直接相加={naive_sum_std:,.0f} "
          f"(縮減 {1 - combined_std/naive_sum_std:.1%})")

    print("\n" + "=" * 70)
    print("3) 逐月明細")
    print("=" * 70)
    detail = pd.DataFrame({
        "vwap_oos": vwap_oos_m, "sr_oos": sr_oos_m,
        f"combined({w_vwap:.2f}x+1x)": combined_oos_m,
    })
    print(detail)


if __name__ == "__main__":
    main()
