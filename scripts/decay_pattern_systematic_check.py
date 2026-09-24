"""系統性檢驗「策略邊際逐年衰退」這個型態，是不是這次會話裡反覆看到的
同一件事，還是各自獨立的巧合。

統一切成三段（不是子區間慣用的5年一格，是刻意配合「衰退」這個問題本身
訂的三段）：
- 2001-2010：早期
- 2011-2020：近十年（IS的後半段）
- 2021-2023：OOS
每個策略在三段都用同一套中檔成本（稅+手續費60元）計算 t 值/PF/淨損益，
直接放在同一張表比較斜率方向是否一致。

涵蓋這次會話（資料修正後）測過、有留下逐筆交易記錄的策略：
1. 開盤上衝（day-session 08:45->09:00）
2. 午盤放空（day-session 12:00->12:30）
3. 盤中翻多+成交量濾網（day-session 12:30->13:00）
4. 唐奇安突破（日線趨勢跟隨，baseline無濾網）
5. RSI(2)均值回歸（日線）
6. 壓力支撐逆勢操作（日線，已知整體顯著負）
7. 開盤區間突破 ORB（1分鐘精確執行）

用法：
    python scripts/decay_pattern_systematic_check.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tw_quant.donchian_breakout_strategy import DonchianConfig, backtest as backtest_donchian  # noqa: E402
from tw_quant.hammer_signal_backtest import COST_SCENARIOS, apply_costs  # noqa: E402
from tw_quant.lunch_reversal_strategy import backtest as backtest_lunch  # noqa: E402
from tw_quant.opening_range_breakout_strategy import backtest as backtest_orb  # noqa: E402
from tw_quant.opening_rally_strategy import backtest as backtest_opening  # noqa: E402
from tw_quant.rsi2_mean_reversion_strategy import Rsi2Config, backtest as backtest_rsi2  # noqa: E402
from tw_quant.support_resistance_fade_strategy import SupportResistanceFadeConfig, backtest as backtest_sr_fade  # noqa: E402
from tw_quant.technical_indicators import daily_indicators, filter_trades_by_volume  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = REPO_ROOT / "data" / "txf_1min.parquet"
OUT_PATH = REPO_ROOT / "data" / "decay_pattern_systematic_check.parquet"
MID_COST = COST_SCENARIOS[1]

ERAS = [
    ("2001-2010", "2001-01-01", "2011-01-01"),
    ("2011-2020", "2011-01-01", "2021-01-01"),
    ("2021-2023(OOS)", "2021-01-01", "2024-01-01"),
]


def tstat(points: pd.Series) -> float:
    n = len(points)
    if n < 2 or points.std(ddof=1) == 0:
        return np.nan
    return points.mean() / (points.std(ddof=1) / np.sqrt(n))


def era_stats(trades: pd.DataFrame, date_col: str, strategy: str) -> list[dict]:
    rows = []
    dates = pd.to_datetime(trades[date_col])
    for era_name, start, end in ERAS:
        mask = (dates >= start) & (dates < end)
        sub = trades.loc[mask]
        if sub.empty:
            rows.append(dict(strategy=strategy, era=era_name, n=0, t_stat=np.nan, profit_factor=np.nan, mean_net_twd=np.nan, sum_net_twd=np.nan))
            continue
        costed = apply_costs(sub, MID_COST)
        net = costed["net_twd"]
        gross_win = net[net > 0].sum()
        gross_loss = -net[net <= 0].sum()
        pf = gross_win / gross_loss if gross_loss > 0 else np.inf
        rows.append(dict(strategy=strategy, era=era_name, n=len(sub), t_stat=tstat(sub["pnl_points"]),
                          profit_factor=pf, mean_net_twd=net.mean(), sum_net_twd=net.sum()))
    return rows


def main() -> None:
    df = pd.read_parquet(DATA_PATH)
    is_df = df[df["datetime"] < "2021-01-01"]  # 用來算「樣本內算出、寫死套用」的門檻，不能用全期資料算
    vol_threshold = daily_indicators(is_df)["vol_ratio_lag1"].quantile(2 / 3)

    all_rows = []

    print("計算開盤上衝...")
    opening_trades = backtest_opening(df)
    all_rows += era_stats(opening_trades, "trading_date", "開盤上衝")

    print("計算午盤放空 / 盤中翻多+量能濾網...")
    lunch_trades = backtest_lunch(df)
    short_leg = lunch_trades[lunch_trades["leg"] == "short_lunch_dip"]
    long_leg = filter_trades_by_volume(lunch_trades[lunch_trades["leg"] == "long_afternoon_rebound"], df, vol_threshold)
    all_rows += era_stats(short_leg, "trading_date", "午盤放空")
    all_rows += era_stats(long_leg, "trading_date", "盤中翻多+量能濾網")

    print("計算唐奇安突破（baseline）...")
    donchian_trades = backtest_donchian(df, DonchianConfig())
    all_rows += era_stats(donchian_trades, "entry_date", "唐奇安突破")

    print("計算RSI(2)均值回歸...")
    rsi2_trades = backtest_rsi2(df, Rsi2Config())
    all_rows += era_stats(rsi2_trades, "entry_date", "RSI(2)均值回歸")

    print("計算壓力支撐逆勢操作...")
    sr_trades = backtest_sr_fade(df, SupportResistanceFadeConfig())
    all_rows += era_stats(sr_trades, "entry_date", "壓力支撐逆勢")

    print("計算開盤區間突破ORB...")
    orb_trades = backtest_orb(df)
    all_rows += era_stats(orb_trades, "trading_date", "開盤區間突破ORB")

    result = pd.DataFrame(all_rows)
    pivot_t = result.pivot(index="strategy", columns="era", values="t_stat")
    pivot_t = pivot_t[[e[0] for e in ERAS]]
    pivot_pf = result.pivot(index="strategy", columns="era", values="profit_factor")
    pivot_pf = pivot_pf[[e[0] for e in ERAS]]

    pd.set_option("display.width", 160)
    pd.set_option("display.float_format", lambda x: f"{x:.2f}")
    print("\n=== t 值（三段） ===")
    print(pivot_t.to_string())
    print("\n=== Profit Factor（三段） ===")
    print(pivot_pf.to_string())

    # 檢驗「單調衰退」：早期 t 值 > 近十年 t 值 > OOS t 值 是否成立
    monotonic = ((pivot_t.iloc[:, 0] > pivot_t.iloc[:, 1]) & (pivot_t.iloc[:, 1] >= pivot_t.iloc[:, 2] - 1.0)).sum()
    print(f"\n{monotonic}/{len(pivot_t)} 個策略呈現「早期最強、之後同等或轉弱」的型態（早期t > 近十年t，且近十年t不明顯低於OOS）")

    result.to_parquet(OUT_PATH, index=False)
    print(f"\nsaved: {OUT_PATH}")


if __name__ == "__main__":
    main()
