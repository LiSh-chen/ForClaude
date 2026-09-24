"""動能檢查點確認 + ATR固定停損停利（日盤當沖），刻意設計成**完全不需要任何
連續指標運算**——跟趨勢日偵測（VWAP版）不同，這裡整個交易日只需要看幾個
固定時間點的價格、跟一個「前一天就已經算好」的數字，全部可以用眼睛讀價位
表操作，不需要即時運算任何東西：

- 進場判斷只看兩個固定時間點（預設09:45、11:00）的收盤價，跟開盤價相減
  就好，不是連續追蹤VWAP或任何移動水準。
- 停損/停利水準用「前一天收盤為止」的日線ATR算一次，整天固定不變，
  可以在進場當下就直接跟券商掛好停損單+停利單（限價單），掛完就不用
  再盯盤，等觸價自動成交，這是最貼近一般散戶/半自動下單流程的設計。

核心邏輯（用兩個檢查點代替VWAP的連續持續性判斷）：
- 檢查點一（09:45）：從開盤到這個時間點，價格要先走出一個小幅度
  （min_move_at_checkpoint1），確立初步方向。
- 檢查點二（11:00，決策點）：價格要沿著檢查點一確立的方向繼續延伸，
  且從開盤累積到這個時間點的總幅度要達到 min_move_at_checkpoint2——
  等於用「兩個時間點都朝同一個方向、且幅度持續擴大」間接確認動能的
  延續性，不需要逐分鐘比對是否曾經反向穿越任何水準。

下單機制：
- 進場：檢查點二那根K棒收盤後，用下一根K棒開盤價成交（市價單，扣滑價）。
- 停損：進場價 -/+ atr_stop_mult × 前一天為止的日線ATR(atr_window)——
  這個水準從進場那一刻就固定，不會再變，可以直接掛真實停損單，逐分鐘
  用 donchian_breakout_strategy._fill_price 同一套「跳空用開盤價、沒跳空
  用水準+滑價」邏輯模擬觸發。
- 停利（可選，target_r_multiple=None時不設）：進場價 +/- target_r_multiple
  × 停損距離，固定風報比，限價單概念（價格到了就成交，不用額外滑價，
  跳空穿越是價格改善）。
- 停損/停利都沒觸發，收盤前（session_end）強制市價出場，扣滑價。

簡化與已知限制：
- 只做日盤（08:45-13:45），不含夜盤。
- 日線ATR用 donchian_breakout_strategy._atr 的同一套算法（EWM平滑
  true range），shift(1) 確保用的是「今天開盤前就已經確定」的數值，
  不含未來函數。
- 停損判斷跟唐奇安突破一樣，只用當天日內K棒的高低點跟水準比較，沒有
  逐檔查真正成交明細。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import time

import pandas as pd

from tw_quant.donchian_breakout_strategy import _atr, _fill_price

POINT_VALUE = 50.0


@dataclass
class MomentumCheckpointConfig:
    checkpoint1_time: time = time(9, 45)
    checkpoint2_time: time = time(11, 0)
    min_move_at_checkpoint1: float = 10.0
    min_move_at_checkpoint2: float = 20.0
    atr_window: int = 14
    atr_stop_mult: float = 1.5
    target_r_multiple: float | None = 2.0
    session_end: time = time(13, 25)
    slippage_points: float = 1.0


def _prior_day_atr(df: pd.DataFrame, cfg: MomentumCheckpointConfig) -> pd.DataFrame:
    from tw_quant.technical_indicators import build_daily_bars

    daily = build_daily_bars(df)
    daily["atr_prior"] = _atr(daily, cfg.atr_window).shift(1)
    return daily[["date", "atr_prior"]]


def _day_session_frame(df: pd.DataFrame) -> pd.DataFrame:
    t = df["datetime"].dt.time
    d = df[(t >= time(8, 45)) & (t <= time(13, 45))].copy()
    d["trading_date"] = d["datetime"].dt.date
    return d


def backtest(df: pd.DataFrame, cfg: MomentumCheckpointConfig | None = None) -> pd.DataFrame:
    cfg = cfg or MomentumCheckpointConfig()
    day_df = _day_session_frame(df)
    atr_table = _prior_day_atr(df, cfg)
    atr_map = dict(zip(atr_table["date"].dt.date, atr_table["atr_prior"]))

    trades = []
    for trading_date, g in day_df.groupby("trading_date"):
        atr_prior = atr_map.get(trading_date)
        if atr_prior is None or pd.isna(atr_prior):
            continue

        g = g.sort_values("datetime").reset_index(drop=True)
        t = g["datetime"].dt.time
        day_open = g["open"].iloc[0]

        cp1_mask = t <= cfg.checkpoint1_time
        cp2_mask = t <= cfg.checkpoint2_time
        if not cp1_mask.any() or not cp2_mask.any():
            continue
        cp1_idx = cp1_mask[cp1_mask].index[-1]
        cp2_idx = cp2_mask[cp2_mask].index[-1]
        if cp2_idx + 1 >= len(g):
            continue

        cp1_close = g["close"].iloc[cp1_idx]
        move1 = cp1_close - day_open
        if abs(move1) < cfg.min_move_at_checkpoint1:
            continue
        direction_sign = 1 if move1 > 0 else -1

        cp2_close = g["close"].iloc[cp2_idx]
        move2 = (cp2_close - day_open) * direction_sign
        if move2 < cfg.min_move_at_checkpoint2:
            continue  # 檢查點二沒有沿著檢查點一的方向繼續延伸足夠幅度

        direction = "long" if direction_sign > 0 else "short"
        entry_bar = g.iloc[cp2_idx + 1]
        entry_price = entry_bar["open"] + (cfg.slippage_points if direction == "long" else -cfg.slippage_points)
        entry_dt = entry_bar["datetime"]

        stop_distance = cfg.atr_stop_mult * atr_prior
        if direction == "long":
            stop = entry_price - stop_distance
            target = entry_price + cfg.target_r_multiple * stop_distance if cfg.target_r_multiple else None
        else:
            stop = entry_price + stop_distance
            target = entry_price - cfg.target_r_multiple * stop_distance if cfg.target_r_multiple else None

        exit_price = exit_dt = exit_reason = None
        for j in range(cp2_idx + 1, len(g)):
            r = g.iloc[j]
            if t.iloc[j] >= cfg.session_end:
                break
            if direction == "long":
                stop_hit = r["low"] <= stop
                target_hit = target is not None and r["high"] >= target
                if stop_hit:
                    exit_price, exit_reason = _fill_price(stop, r["open"], r["high"], r["low"], "sell_stop", cfg.slippage_points)
                    exit_dt = r["datetime"]
                    break
                if target_hit:
                    exit_price = target if r["open"] < target else r["open"]
                    exit_dt, exit_reason = r["datetime"], "target"
                    break
            else:
                stop_hit = r["high"] >= stop
                target_hit = target is not None and r["low"] <= target
                if stop_hit:
                    exit_price, exit_reason = _fill_price(stop, r["open"], r["high"], r["low"], "buy_stop", cfg.slippage_points)
                    exit_dt = r["datetime"]
                    break
                if target_hit:
                    exit_price = target if r["open"] > target else r["open"]
                    exit_dt, exit_reason = r["datetime"], "target"
                    break

        if exit_price is None:
            close_mask = t >= cfg.session_end
            last_bar = g.loc[close_mask].iloc[0] if close_mask.any() else g.iloc[-1]
            exit_price = last_bar["open"] + (-cfg.slippage_points if direction == "long" else cfg.slippage_points)
            exit_dt, exit_reason = last_bar["datetime"], "session_close"

        pnl_points = (exit_price - entry_price) if direction == "long" else (entry_price - exit_price)
        trades.append(dict(
            trading_date=trading_date, direction=direction, atr_prior=atr_prior, stop_distance=stop_distance,
            entry_dt=entry_dt, entry_price=entry_price, exit_dt=exit_dt, exit_price=exit_price,
            exit_reason=exit_reason, pnl_points=pnl_points,
        ))

    trades_df = pd.DataFrame(trades)
    if trades_df.empty:
        return trades_df
    trades_df["pnl_twd"] = trades_df["pnl_points"] * POINT_VALUE
    return trades_df.sort_values("entry_dt").reset_index(drop=True)
