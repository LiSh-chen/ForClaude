"""把午盤效應策略的進出場時段拉長，看看能不能找到更好的版本或其他新的有效
時段。三個檢查，全部只用樣本內（2001-2020）資料做決定，最後才對「定案設定」
跑一次樣本外驗證：

1. 空腿進場時間往前拉（11:00~12:00）：檢查 12:00 是不是真的是最佳進場點，
   還是只是隨便選的整數時間。
2. 多腿出場時間往後拉到收盤（12:45~13:45）：反正跟空腿共用同一筆交易，
   拉長持有不會多產生一筆交易成本，理論上有機會「免費」多賺一點。
3. 夜盤子夜前後（23:30~23:45 進場、00:15~01:30 出場）：粗略掃一次夜盤有沒有
   類似的流動性失衡窗口——樣本只有 890 個夜盤 session（2017 年才開始），
   遠少於日盤的 4943 天，且太短無法做 5 年子區間穩健性檢查，結果僅供參考。

結論（寫在這裡，不用重新跑就知道）：
- 空腿進場：11:00→12:00 t 值單調從 2.61 升到 5.70，12:00 確實是目前抓得到的
  最佳進場點，往前拉只會稀釋訊號。
- 多腿出場拉到收盤：樣本內看起來更好（毛損益 622,800→774,250，中成本情境
  淨損益 -195,231→-4,032），但樣本外反而變差（毛損益 159,800→141,200，
  2022 年從正轉負）——判定為過度配適到樣本內較早期的衰減效應，**不採用**，
  維持原本 13:00 出場。
- 夜盤子夜窗口：最好的組合（23:45 進場、01:30 出場）t=2.62，低於這次分析
  一路採用的 |t|>=3 門檻，且樣本量/涵蓋年數都遠不如日盤版本，不足以當成
  可信的策略，暫不採用，需要更多年份的夜盤資料累積後再重新檢視。
"""

from __future__ import annotations

import sys
from datetime import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tw_quant.hammer_signal_backtest import COST_SCENARIOS, apply_costs  # noqa: E402
from tw_quant.lunch_reversal_strategy import LunchReversalConfig, backtest  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = REPO_ROOT / "data" / "txf_1min.parquet"
OUT_PATH = REPO_ROOT / "data" / "lunch_reversal_window_length_exploration.parquet"

IS_CUTOFF = "2021-01-01"


def short_entry_sweep(is_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for et in [time(11, 0), time(11, 15), time(11, 30), time(11, 45), time(12, 0)]:
        cfg = LunchReversalConfig(short_entry=et)
        trades = backtest(is_df, cfg)
        t = trades[trades["leg"] == "short_lunch_dip"]
        se = t["pnl_points"].std(ddof=1) / np.sqrt(len(t))
        rows.append(dict(check="short_entry_sweep", param=et.strftime("%H:%M"), n=len(t),
                          mean_pts=t["pnl_points"].mean(), tstat=t["pnl_points"].mean() / se))
    return pd.DataFrame(rows)


def long_exit_sweep(is_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for xt in [time(12, 45), time(13, 0), time(13, 15), time(13, 30), time(13, 45)]:
        cfg = LunchReversalConfig(long_exit=xt)
        trades = backtest(is_df, cfg)
        t = trades[trades["leg"] == "long_afternoon_rebound"]
        se = t["pnl_points"].std(ddof=1) / np.sqrt(len(t))
        mid_net = apply_costs(trades, COST_SCENARIOS[1])["net_twd"].sum()
        rows.append(dict(check="long_exit_sweep", param=xt.strftime("%H:%M"), n=len(t),
                          mean_pts=t["pnl_points"].mean(), tstat=t["pnl_points"].mean() / se,
                          gross_total_twd=trades["pnl_twd"].sum(), net_mid_twd=mid_net))
    return pd.DataFrame(rows)


def oos_comparison(oos_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label, xt in [("原版 13:00", time(13, 0)), ("延長至收盤 13:45", time(13, 45))]:
        cfg = LunchReversalConfig(long_exit=xt)
        trades = backtest(oos_df, cfg)
        for cost in COST_SCENARIOS:
            net = apply_costs(trades, cost)["net_twd"].sum()
            rows.append(dict(check="oos_comparison", param=label, cost_scenario=cost.label,
                              n=len(trades), gross_total_twd=trades["pnl_twd"].sum(), net_twd=net))
    return pd.DataFrame(rows)


def main() -> None:
    df = pd.read_parquet(DATA_PATH)
    is_df = df[df["datetime"] < IS_CUTOFF]
    oos_df = df[df["datetime"] >= IS_CUTOFF]

    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", lambda x: f"{x:,.3f}")

    print("=== 1. 空腿進場時間往前拉（樣本內）===")
    s1 = short_entry_sweep(is_df)
    print(s1.to_string(index=False))

    print("\n=== 2. 多腿出場時間往後拉到收盤（樣本內）===")
    s2 = long_exit_sweep(is_df)
    print(s2.to_string(index=False))

    print("\n=== 3. 原版 vs 延長版，樣本外對照（只跑一次）===")
    s3 = oos_comparison(oos_df)
    print(s3.to_string(index=False))

    pd.concat([s1, s2, s3], ignore_index=True).to_parquet(OUT_PATH, index=False)
    print(f"\nsaved: {OUT_PATH}")
    print("\n結論：兩個延伸方向都沒有產出可採用的改版，維持原本 12:00/12:30/13:00 設定。")


if __name__ == "__main__":
    main()
