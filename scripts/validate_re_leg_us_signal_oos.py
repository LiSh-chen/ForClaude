"""OOS驗證候選：「re腿（12:30進場多單、13:00出場）＋前一美股交易日S&P500下跌」。

背景（見 scripts/scan_us_signal_time_window_grid.py 的IS探索與這次會話跟
使用者的討論）：把三腿策略裡已知的 re 窗口（12:30-13:00 多單）依「前一個美股
交易日 S&P500 收盤對收盤報酬」正負分組後，四個5年子區間（2001-2005 / 2006-
2010 / 2011-2015 / 2016-2020）重算t值：

    美股下跌日：t = 2.67 / 2.48 / 2.99 / 2.12  （全部>2，沒有翻負）
    無條件基準：t = 0.69 / 1.82 / 3.25 / 2.43  （早期不穩定）
    美股上漲日：t = -1.90 / 0.06 / 1.45 / 1.29  （不一致）

「re窗口＋美股下跌日」比原本無條件的re腿本身更跨期穩健，被選為下一個花掉
單次OOS驗證機會的候選。

**重要**：這裡驗證的是「美股訊號濾網」單獨的效果，不疊加既有三腿策略裡的
成交量三分位濾網（那是另一個獨立找到的濾網，疊加兩個濾網會變成沒有IS/
子區間穩健性檢驗過的新組合）。用 tw_quant.lunch_reversal_strategy 產生的
原始 long_afternoon_rebound 交易（不經過volume filter），只套用美股訊號
濾網。

只跑一次OOS，不管結果如何都不回頭調整。

用法：
    python scripts/validate_re_leg_us_signal_oos.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tw_quant.hammer_signal_backtest import COST_SCENARIOS, apply_costs  # noqa: E402
from tw_quant.lunch_reversal_strategy import backtest as backtest_lunch  # noqa: E402
from tw_quant.us_market_signal import align_to_txf_dates  # noqa: E402

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


def us_down_mask(trading_dates: pd.Series) -> pd.Series:
    aligned = align_to_txf_dates(pd.to_datetime(trading_dates))
    aligned = aligned.set_index(aligned["txf_date"].dt.date)
    sp500_ret = aligned["sp500_ret"].reindex(trading_dates).to_numpy()
    return pd.Series(sp500_ret < 0, index=trading_dates.index)


def re_leg_trades(data_df: pd.DataFrame) -> pd.DataFrame:
    lunch_trades = backtest_lunch(data_df)
    return lunch_trades[lunch_trades["leg"] == "long_afternoon_rebound"].reset_index(drop=True)


def report(trades: pd.DataFrame, label: str) -> dict:
    n = len(trades)
    if n == 0:
        print(f"  {label}: n=0")
        return dict(label=label, n=0)
    t = tstat(trades["pnl_points"])
    mid_net = apply_costs(trades, COST_SCENARIOS[1])["net_twd"].sum()
    print(f"  {label}: n={n} t={t:.2f} 中檔成本淨損益={mid_net:,.0f}")
    return dict(label=label, n=n, t_stat=t, mid_net_twd=mid_net)


def main() -> None:
    df = pd.read_parquet(DATA_PATH)
    is_df = df[df["datetime"] < IS_CUTOFF]
    oos_df = df[df["datetime"] >= IS_CUTOFF]

    print("=" * 70)
    print("0) 重新確認：IS（2001-2020）整體，美股下跌日 vs 全部/上漲日")
    print("=" * 70)
    is_trades = re_leg_trades(is_df)
    is_mask = us_down_mask(is_trades["trading_date"])
    is_rows = []
    is_rows.append(report(is_trades, "IS 全部"))
    is_rows.append(report(is_trades[is_mask.to_numpy()], "IS 美股下跌日"))
    is_rows.append(report(is_trades[~is_mask.to_numpy()], "IS 美股上漲日"))

    print("\n" + "=" * 70)
    print("1) IS 子區間穩健性複查（跟scan_us_signal_time_window_grid.py的原始資料一致性）")
    print("=" * 70)
    sub_rows = []
    for name, start, end in SUB_PERIODS:
        sub_df = df[(df["datetime"] >= start) & (df["datetime"] < end)]
        sub_trades = re_leg_trades(sub_df)
        sub_mask = us_down_mask(sub_trades["trading_date"])
        print(f"\n--- {name} ---")
        row = report(sub_trades[sub_mask.to_numpy()], "美股下跌日")
        row["period"] = name
        sub_rows.append(row)

    print("\n" + "=" * 70)
    print("2) OOS（2021-2023）唯一一次驗證：re腿 + 前一美股交易日下跌")
    print("=" * 70)
    oos_trades = re_leg_trades(oos_df)
    oos_mask = us_down_mask(oos_trades["trading_date"])
    oos_down = oos_trades[oos_mask.to_numpy()]
    oos_up = oos_trades[~oos_mask.to_numpy()]

    oos_rows = []
    oos_rows.append(report(oos_trades, "OOS 全部（對照組，未過濾）"))
    oos_rows.append(report(oos_down, "OOS 美股下跌日（候選本身）"))
    oos_rows.append(report(oos_up, "OOS 美股上漲日（對照）"))

    print("\n三種成本情境下 OOS candidate（美股下跌日）淨損益：")
    for cost in COST_SCENARIOS:
        net = apply_costs(oos_down, cost)["net_twd"].sum()
        print(f"  {cost.label}: {net:,.0f}")

    pd.DataFrame(is_rows).to_parquet(OUT_DIR / "re_leg_us_signal_validation_is.parquet", index=False)
    pd.DataFrame(sub_rows).to_parquet(OUT_DIR / "re_leg_us_signal_validation_subperiods.parquet", index=False)
    pd.DataFrame(oos_rows).to_parquet(OUT_DIR / "re_leg_us_signal_validation_oos.parquet", index=False)
    print("\nsaved: re_leg_us_signal_validation_{is,subperiods,oos}.parquet")


if __name__ == "__main__":
    main()
