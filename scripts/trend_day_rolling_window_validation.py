"""趨勢日策略（放寬版 min_dominant_side_fraction=0.90）滾動窗口驗證。

資料集只到 2023-12-29，沒有更新的原始檔案可以真的延長樣本外年份。改用
滾動窗口的方式，把現有 23 年資料切成多個跟原本 OOS 期間（2021-2023，
約3年）差不多長度的窗口，全部套用同一組已經鎖定的參數（decision_time=
11:00, min_move_points=20, min_dominant_side_fraction=0.90, slippage=1pt
——這組參數是用 2001-2020 IS 資料選出來的，這裡完全不重新調參，單純
拿去對每個窗口分別驗證一次），直接看這個訊號在「不同的3年切片」下是否
普遍存在，而不是只看兩個端點（早期 IS vs 唯一一次 2021-2023 OOS）。

這不是新資料，是把現有資料切得更細來提升統計檢定力，跟本來的單一
IS/OOS切分是互補而非取代關係——2021-2023那次OOS驗證依然是最後、最
接近「未來」的檢驗，這裡只是額外多看幾組「如果OOS落在其他年份會怎樣」。

用法：
    python scripts/trend_day_rolling_window_validation.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tw_quant.hammer_signal_backtest import COST_SCENARIOS, apply_costs  # noqa: E402
from tw_quant.trend_day_strategy import TrendDayConfig, backtest  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = REPO_ROOT / "data" / "txf_1min.parquet"
OUT_PATH = REPO_ROOT / "data" / "trend_day_rolling_window_validation.parquet"

# 已經鎖定、不再回頭調整的參數（來自 2001-2020 IS 選出的 min_dominant_side_fraction=0.90）
LOCKED_CFG = TrendDayConfig(min_dominant_side_fraction=0.90)

WINDOWS = [
    ("2001-2003", "2001-01-01", "2004-01-01"),
    ("2004-2006", "2004-01-01", "2007-01-01"),
    ("2007-2009", "2007-01-01", "2010-01-01"),
    ("2010-2012", "2010-01-01", "2013-01-01"),
    ("2013-2015", "2013-01-01", "2016-01-01"),
    ("2016-2018", "2016-01-01", "2019-01-01"),
    ("2019-2021", "2019-01-01", "2022-01-01"),
    ("2021-2023(原OOS)", "2021-01-01", "2024-01-01"),
]


def tstat(points: pd.Series) -> float:
    n = len(points)
    if n < 2 or points.std(ddof=1) == 0:
        return np.nan
    return points.mean() / (points.std(ddof=1) / np.sqrt(n))


def main() -> None:
    df = pd.read_parquet(DATA_PATH)

    rows = []
    for name, start, end in WINDOWS:
        sub_df = df[(df["datetime"] >= start) & (df["datetime"] < end)]
        trades = backtest(sub_df, LOCKED_CFG)
        if trades.empty:
            rows.append(dict(window=name, n=0))
            continue
        costed = apply_costs(trades, COST_SCENARIOS[1])
        net = costed["net_twd"]
        gross_win = net[net > 0].sum()
        gross_loss = -net[net <= 0].sum()
        pf = gross_win / gross_loss if gross_loss > 0 else np.inf
        rows.append(dict(
            window=name, n=len(trades), win_rate=(net > 0).mean(),
            mean_net_twd=net.mean(), sum_net_twd=net.sum(), profit_factor=pf,
            t_stat=tstat(trades["pnl_points"]), mean_pnl_points=trades["pnl_points"].mean(),
        ))

    result = pd.DataFrame(rows)
    pd.set_option("display.width", 160)
    pd.set_option("display.float_format", lambda x: f"{x:.2f}")
    print(result.to_string(index=False))

    positive_windows = (result["sum_net_twd"] > 0).sum()
    significant_windows = (result["t_stat"].abs() >= 2).sum()
    print(f"\n{positive_windows}/{len(result)} 個窗口淨損益為正")
    print(f"{significant_windows}/{len(result)} 個窗口 |t| >= 2（個別窗口樣本數通常偏小，這個門檻僅供參考）")

    # 把所有窗口的交易直接合併，看整體 23 年、同一組固定參數的總表現
    all_trades = pd.concat([backtest(df[(df["datetime"] >= s) & (df["datetime"] < e)], LOCKED_CFG)
                             for _, s, e in WINDOWS], ignore_index=True)
    all_costed = apply_costs(all_trades, COST_SCENARIOS[1])
    overall_t = tstat(all_trades["pnl_points"])
    print(f"\n全部23年合併（同一組固定參數）: n={len(all_trades)}, t={overall_t:.2f}, "
          f"中檔成本淨損益={all_costed['net_twd'].sum():,.0f}")

    result.to_parquet(OUT_PATH, index=False)
    print(f"\nsaved: {OUT_PATH}")


if __name__ == "__main__":
    main()
