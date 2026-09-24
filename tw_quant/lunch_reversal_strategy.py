"""午盤效應策略：日盤 12:00-12:30「午盤下殺」放空 + 12:30-13:00「盤中反彈」
做多。

這個策略的起點不是一個事先假設的技術型態，是先在 2001-2020 樣本內資料上做
「每日固定 30 分鐘時段」的報酬統計掃描（見 `scripts/scan_intraday_seasonality.py`），
發現 12:00-12:30 這個區塊 20 年一致顯著下跌（每個 5 年子區間 t 值都 <= -2.0），
12:30-13:00 一致顯著上漲（多數子區間 t >= 2），才反過來設計交易規則去捕捉它——
跟本次會話先前幾個策略「先有型態、再測有沒有用」的順序相反。

規則：
- 12:00 這根K棒的開盤價放空 1 口，12:30 那根K棒的開盤價回補（同時間立刻翻多）。
- 12:30 那根K棒的開盤價做多 1 口，13:00 那根K棒的開盤價出場。
- 若當天資料在某個時間點缺漏（找不到 >= 該時刻的K棒），當天這一腿直接跳過。
- 可選的保護性停損 `max_loss_points`：兩腿持有期間，只要虧損點數觸及這個上限
  就提前出場（用觸及當根的極端價位計算，不是等到排定時間），預設 None（不設
  停損，因為這是靠「多次重複的統計期望值」賺錢的日曆效應策略，不是靠單筆
  行情判斷，過緊的停損可能只是把雜訊當訊號提前出場、反而傷害期望值——但仍
  提供這個選項讓使用者自行評估要不要加保護）。
- 只算毛損益（點數 × 50）；成本另外用 `tw_quant.hammer_signal_backtest.TradeCost`
  套用（同一套稅金+手續費模型）。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import time

import numpy as np
import pandas as pd

POINT_VALUE = 50.0

SHORT_ENTRY_TIME = time(12, 0)
FLIP_TIME = time(12, 30)
LONG_EXIT_TIME = time(13, 0)


@dataclass
class LunchReversalConfig:
    short_entry: time = SHORT_ENTRY_TIME
    flip: time = FLIP_TIME
    long_exit: time = LONG_EXIT_TIME
    max_loss_points: float | None = None  # 保護性停損，預設不設


def _day_session_frame(df: pd.DataFrame) -> pd.DataFrame:
    t = df["datetime"].dt.time
    mask = (t >= time(8, 45)) & (t <= time(13, 45))
    d = df.loc[mask].copy()
    d["trading_date"] = d["datetime"].dt.date
    return d.sort_values("datetime").reset_index(drop=True)


def _first_bar_at_or_after(day_bars: pd.DataFrame, t: time) -> pd.Series | None:
    hit = day_bars[day_bars["datetime"].dt.time >= t]
    return hit.iloc[0] if len(hit) else None


def _apply_protective_stop(
    path_bars: pd.DataFrame, entry_price: float, direction: str, max_loss_points: float | None
) -> tuple[float, pd.Timestamp, str]:
    """direction: 'short' 或 'long'。回傳 (exit_price, exit_dt, exit_reason)；
    若沒有觸發停損（或未設停損、或中間沒有K棒），回傳 (None, None, 'scheduled')
    交給呼叫端用排定時間的那根開盤價出場。"""
    if max_loss_points is None or path_bars.empty:
        return None, None, "scheduled"
    stop_price = entry_price + max_loss_points if direction == "short" else entry_price - max_loss_points
    for _, bar in path_bars.iterrows():
        if direction == "short" and bar["high"] >= stop_price:
            return stop_price, bar["datetime"], "stop_loss"
        if direction == "long" and bar["low"] <= stop_price:
            return stop_price, bar["datetime"], "stop_loss"
    return None, None, "scheduled"


def backtest(df: pd.DataFrame, cfg: LunchReversalConfig | None = None) -> pd.DataFrame:
    cfg = cfg or LunchReversalConfig()
    d = _day_session_frame(df)

    trades = []
    for trading_date, day_bars in d.groupby("trading_date", sort=True):
        day_bars = day_bars.reset_index(drop=True)

        short_entry_bar = _first_bar_at_or_after(day_bars, cfg.short_entry)
        flip_bar = _first_bar_at_or_after(day_bars, cfg.flip)
        long_exit_bar = _first_bar_at_or_after(day_bars, cfg.long_exit)
        if short_entry_bar is None or flip_bar is None or long_exit_bar is None:
            continue
        if not (short_entry_bar["datetime"] < flip_bar["datetime"] < long_exit_bar["datetime"]):
            continue

        # --- 空腿：12:00 開盤放空，12:30 開盤回補（除非提早被停損打到）---
        short_entry_price = short_entry_bar["open"]
        path1 = day_bars[(day_bars["datetime"] > short_entry_bar["datetime"]) &
                          (day_bars["datetime"] <= flip_bar["datetime"])]
        sp, sdt, sreason = _apply_protective_stop(path1, short_entry_price, "short", cfg.max_loss_points)
        if sreason == "scheduled":
            sp, sdt = flip_bar["open"], flip_bar["datetime"]
        trades.append(dict(
            leg="short_lunch_dip", trading_date=trading_date,
            entry_dt=short_entry_bar["datetime"], entry_price=short_entry_price,
            exit_dt=sdt, exit_price=sp, exit_reason=sreason,
            pnl_points=short_entry_price - sp,
        ))

        # --- 多腿：12:30 開盤做多，13:00 開盤出場（除非提早被停損打到）---
        long_entry_price = flip_bar["open"]
        path2 = day_bars[(day_bars["datetime"] > flip_bar["datetime"]) &
                          (day_bars["datetime"] <= long_exit_bar["datetime"])]
        lp, ldt, lreason = _apply_protective_stop(path2, long_entry_price, "long", cfg.max_loss_points)
        if lreason == "scheduled":
            lp, ldt = long_exit_bar["open"], long_exit_bar["datetime"]
        trades.append(dict(
            leg="long_afternoon_rebound", trading_date=trading_date,
            entry_dt=flip_bar["datetime"], entry_price=long_entry_price,
            exit_dt=ldt, exit_price=lp, exit_reason=lreason,
            pnl_points=lp - long_entry_price,
        ))

    trades_df = pd.DataFrame(trades)
    if trades_df.empty:
        return trades_df
    trades_df["pnl_twd"] = trades_df["pnl_points"] * POINT_VALUE
    return trades_df.sort_values("entry_dt").reset_index(drop=True)
