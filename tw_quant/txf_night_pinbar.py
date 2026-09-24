"""小台指 (MTX) 夜盤長下影線策略回測（策略一：震盪低谷防禦 / 策略二：突破回測 10MA 跟隨）。

規格書多處留有解讀空間，本模組採用的明確假設：

- 前波高/低點的回看根數 N：規格未給數字，預設 20（可調，見 `StrategyConfig.swing_lookback`）。
  `roll_high`/`roll_low` 一律用「當根之前」的 N 根（shift(1) 後再 rolling），
  不含當根本身，避免用到未來資訊。
- 「初始停損點 = 進場K線最低點 - 5」中的「進場K線」，採用訊號K棒（成立長下影線
  那一根）的低點，而不是次根（實際成交）K棒的低點。
- 10MA、前波高低點 N 根視窗，都在**每個夜盤 session 內重新計算**（15:00 開盤即
  重置滾動視窗，不跨日銜接前一個 session 的資料）。
- 同一策略同一時間只允許一筆未平倉部位；訊號出現時若已在場內，忽略該訊號。
- 若同一根K棒同時滿足停損與停利（或強制平倉）條件，保守假設停損優先觸發。
- 策略二「突破後拉回觸及」：只要價格觸及仍未回測過的最近一次突破水準且尚未形成
  長下影線，就持續等待（不失效）；一旦更高的新突破出現，watch 的水準會更新為
  最新這次突破前的高點；一旦觸及且形成長下影線（不論是否因時間窗/風控被跳過），
  該次突破水準即視為已使用，重新等待下一次突破。
- 只算毛損益（點數 × 50），未計入期交稅、手續費、滑價。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import time

import numpy as np
import pandas as pd

POINT_VALUE = 50  # 小台指 1 點 = 50 元
FORCE_CLOSE_TIME = time(23, 30)


@dataclass
class StrategyConfig:
    swing_lookback: int = 20  # N：前波高/低點回看根數
    ma_window: int = 10
    sl_buffer: float = 5.0
    max_risk_points: float = 140.0
    entry_start: time = time(21, 30)
    entry_end: time = time(23, 30)  # exclusive
    force_close: time = FORCE_CLOSE_TIME
    point_value: float = POINT_VALUE


def _night_session_frame(df: pd.DataFrame) -> pd.DataFrame:
    """只留夜盤時段（15:00~23:30，策略只在此區間內交易，23:30 後的資料用不到），
    並標上 session_date：15:00~23:59 屬當天，00:00~04:59 屬前一天的 session。"""
    t = df["datetime"].dt.time
    mask = (t >= time(15, 0)) & (t <= time(23, 30))
    d = df.loc[mask].copy()
    d["session_date"] = d["datetime"].dt.date
    d = d.sort_values("datetime").reset_index(drop=True)
    return d


def compute_indicators(df: pd.DataFrame, cfg: StrategyConfig) -> pd.DataFrame:
    d = _night_session_frame(df)
    g = d.groupby("session_date", group_keys=False)

    d["body"] = (d["open"] - d["close"]).abs()
    d["lower_shadow"] = d[["open", "close"]].min(axis=1) - d["low"]
    d["pin_bar"] = d["lower_shadow"] >= 2 * d["body"]

    d["roll_high"] = g["high"].transform(lambda s: s.shift(1).rolling(cfg.swing_lookback).max())
    d["roll_low"] = g["low"].transform(lambda s: s.shift(1).rolling(cfg.swing_lookback).min())
    d["ma10"] = g["close"].transform(lambda s: s.rolling(cfg.ma_window).mean())

    d["time"] = d["datetime"].dt.time
    return d


def _simulate_range_strategy(bars: pd.DataFrame, cfg: StrategyConfig) -> list[dict]:
    n = len(bars)
    b = bars.to_dict("records")
    trades: list[dict] = []
    i = 0
    while i < n - 1:
        bar = b[i]
        is_range = not (pd.notna(bar["roll_high"]) and bar["high"] > bar["roll_high"])
        touched_low = pd.notna(bar["roll_low"]) and bar["low"] <= bar["roll_low"]
        if not (is_range and touched_low and bar["pin_bar"]):
            i += 1
            continue

        nxt = b[i + 1]
        if not (cfg.entry_start <= nxt["time"] < cfg.entry_end):
            i += 1
            continue

        entry_price = nxt["open"]
        sl = bar["low"] - cfg.sl_buffer
        risk = entry_price - sl
        if not (0 < risk <= cfg.max_risk_points):
            i += 1
            continue

        tp = entry_price + 2 * risk
        exit_price = exit_reason = exit_dt = None
        exit_idx = None
        for j in range(i + 1, n):
            bj = b[j]
            if bj["low"] <= sl:
                exit_price, exit_reason, exit_idx = sl, "stop_loss", j
                break
            if bj["high"] >= tp:
                exit_price, exit_reason, exit_idx = tp, "take_profit", j
                break
            if bj["time"] == cfg.force_close:
                exit_price, exit_reason, exit_idx = bj["open"], "forced_close", j
                break
        if exit_idx is None:
            exit_price, exit_reason, exit_idx = b[-1]["close"], "session_end_fallback", n - 1
        exit_dt = b[exit_idx]["datetime"]

        trades.append(
            dict(
                strategy="range_hammer",
                session_date=bar["session_date"],
                signal_dt=bar["datetime"],
                entry_dt=nxt["datetime"],
                entry_price=entry_price,
                sl=sl,
                tp=tp,
                exit_dt=exit_dt,
                exit_price=exit_price,
                exit_reason=exit_reason,
                risk_points=risk,
            )
        )
        i = exit_idx + 1
    return trades


def _simulate_breakout_strategy(bars: pd.DataFrame, cfg: StrategyConfig) -> list[dict]:
    n = len(bars)
    b = bars.to_dict("records")
    trades: list[dict] = []
    i = 0
    last_breakout_level: float | None = None
    while i < n - 1:
        bar = b[i]

        if pd.notna(bar["roll_high"]) and bar["high"] > bar["roll_high"]:
            last_breakout_level = bar["roll_high"]
            i += 1
            continue

        if last_breakout_level is not None and bar["low"] <= last_breakout_level and bar["pin_bar"]:
            last_breakout_level = None  # 這個突破水準的回測機會用掉了，不論是否成交

            nxt = b[i + 1]
            if not (cfg.entry_start <= nxt["time"] < cfg.entry_end):
                i += 1
                continue

            entry_price = nxt["open"]
            sl = bar["low"] - cfg.sl_buffer
            risk = entry_price - sl
            if not (0 < risk <= cfg.max_risk_points):
                i += 1
                continue

            exit_price = exit_reason = None
            exit_idx = None
            trailing_pending = False
            for j in range(i + 1, n):
                bj = b[j]
                if trailing_pending:
                    exit_price, exit_reason, exit_idx = bj["open"], "trailing_ma", j
                    break
                if bj["low"] <= sl:
                    exit_price, exit_reason, exit_idx = sl, "stop_loss", j
                    break
                if bj["time"] == cfg.force_close:
                    exit_price, exit_reason, exit_idx = bj["open"], "forced_close", j
                    break
                if pd.notna(bj["ma10"]) and bj["ma10"] > entry_price and bj["close"] < bj["ma10"]:
                    trailing_pending = True
            if exit_idx is None:
                exit_price, exit_reason, exit_idx = b[-1]["close"], "session_end_fallback", n - 1
            exit_dt = b[exit_idx]["datetime"]

            trades.append(
                dict(
                    strategy="breakout_retest",
                    session_date=bar["session_date"],
                    signal_dt=bar["datetime"],
                    entry_dt=nxt["datetime"],
                    entry_price=entry_price,
                    sl=sl,
                    tp=np.nan,
                    exit_dt=exit_dt,
                    exit_price=exit_price,
                    exit_reason=exit_reason,
                    risk_points=risk,
                )
            )
            i = exit_idx + 1
            continue

        i += 1
    return trades


def backtest(df: pd.DataFrame, cfg: StrategyConfig | None = None) -> pd.DataFrame:
    cfg = cfg or StrategyConfig()
    ind = compute_indicators(df, cfg)

    all_trades: list[dict] = []
    for _, session_bars in ind.groupby("session_date", sort=True):
        session_bars = session_bars.reset_index(drop=True)
        all_trades.extend(_simulate_range_strategy(session_bars, cfg))
        all_trades.extend(_simulate_breakout_strategy(session_bars, cfg))

    columns = [
        "strategy", "session_date", "signal_dt", "entry_dt", "entry_price", "sl", "tp",
        "exit_dt", "exit_price", "exit_reason", "risk_points", "pnl_points", "pnl_twd",
    ]
    if not all_trades:
        return pd.DataFrame(columns=columns)

    trades = pd.DataFrame(all_trades)

    trades = trades.sort_values("entry_dt").reset_index(drop=True)
    trades["pnl_points"] = trades["exit_price"] - trades["entry_price"]
    trades["pnl_twd"] = trades["pnl_points"] * cfg.point_value
    return trades
