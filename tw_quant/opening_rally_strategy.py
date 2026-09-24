"""開盤搶漲策略：日盤開盤 08:45-09:00 做多。

跟 lunch_reversal_strategy.py 一樣，起點是先在樣本內（2001-2020）資料上做
「完整 (start, end) 時段報酬網格」掃描（見
`scripts/scan_full_time_window_grid.py`），而不是先假設型態。這次把時段掃描
從原本固定 30 分鐘的格子，換成任意起訖時間、任意長度的完整網格，才挖出
08:45-09:00 這個全樣本內最強的單一時段（t=6.79），比午盤效應的兩腿都強，
且四個 5 年子區間 t 值都 >= 2（2.12~3.99），比午盤效應更一致地不衰減。

同一次網格掃描也發現「09:00-09:15 立即回吐」「13:15-13:45 收盤前上衝」兩個
全樣本 |t|>=5 的候選，但兩者最近一個子區間（2016-2020）都不顯著（t=-0.63 /
1.36），判定跟之前被否掉的「多腿延長到收盤」屬於同一種「早期歷史撐起來的
衰減效應」，不採用，只保留開盤這一腿。

規則：08:45 那根K棒開盤價做多，09:00 那根K棒開盤價出場。只算毛損益
（點數 × 50）；成本用 `tw_quant.hammer_signal_backtest.TradeCost` 套用。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import time

import pandas as pd

from tw_quant.lunch_reversal_strategy import POINT_VALUE, _day_session_frame, _first_bar_at_or_after

OPEN_ENTRY_TIME = time(8, 45)
OPEN_EXIT_TIME = time(9, 0)


@dataclass
class OpeningRallyConfig:
    entry: time = OPEN_ENTRY_TIME
    exit: time = OPEN_EXIT_TIME


def backtest(df: pd.DataFrame, cfg: OpeningRallyConfig | None = None) -> pd.DataFrame:
    cfg = cfg or OpeningRallyConfig()
    d = _day_session_frame(df)

    trades = []
    for trading_date, day_bars in d.groupby("trading_date", sort=True):
        day_bars = day_bars.reset_index(drop=True)

        entry_bar = _first_bar_at_or_after(day_bars, cfg.entry)
        exit_bar = _first_bar_at_or_after(day_bars, cfg.exit)
        if entry_bar is None or exit_bar is None or entry_bar["datetime"] >= exit_bar["datetime"]:
            continue

        trades.append(dict(
            leg="opening_rally", trading_date=trading_date,
            entry_dt=entry_bar["datetime"], entry_price=entry_bar["open"],
            exit_dt=exit_bar["datetime"], exit_price=exit_bar["open"],
            pnl_points=exit_bar["open"] - entry_bar["open"],
        ))

    trades_df = pd.DataFrame(trades)
    if trades_df.empty:
        return trades_df
    trades_df["pnl_twd"] = trades_df["pnl_points"] * POINT_VALUE
    return trades_df.sort_values("entry_dt").reset_index(drop=True)
