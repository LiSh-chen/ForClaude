"""夜盤流動性熱區趨勢偵測（Night Liquid Window Trend Detection）。

背景：這次會話先前測過的夜盤版本（scripts/scan_night_session_trend.py）
把 VWAP 從整個夜盤開盤（15:00）開始累積，在幾個候選決策時點
（18:00/20:00/22:00/00:00/02:00）分別檢查，結果全部乾淨的零信號
（|t|<1）。但那個版本有個問題：VWAP 從15:00累積到22:00，中間會被
15:00-20:00這段「流動性偏低」的6小時稀釋——用這次會話量能剖析
（各小時平均每分鐘成交量）實際算出來的結果：

    時段(台灣時間)   平均每分鐘量
    21:00           79.6
    22:00          103.7  <- 全夜盤最高
    23:00           83.9
    15:00(夜盤開盤)  70.7
    18:00-19:00     ~30   <- 明顯偏低
    02:00-04:00    ~25-33 <- 全夜盤最低

21:00-23:00這段（跟美股現貨開盤時段高度重疊：夏令9:30ET=21:30台灣，
冬令=22:30台灣）量能是02:00-04:00清淡時段的3-4倍。這個模組直接把
「VWAP持續偏一側」這個核心概念重新套用，但**完全限定在這個流動性熱區
本身**（VWAP從熱區開盤window_start=21:00才開始累積，不是被稀釋過的
版本），決策時點跟出場都在熱區內完成，設計精神更接近已經驗證過的
日盤版本（VWAP從日盤開盤08:45累積、決策點11:00、出場前13:25，全部
在同一個流動性穩定的區間內），而不是像前一版本那樣橫跨流動性天差地遠
的整段夜盤。

刻意選在午夜前（21:00-23:45）完全結束，不跨零點，避開「時刻比較
不知道跨日」那個這次會話已經踩過一次的bug（scan_night_session_trend.py
最早版本的時間比較邏輯錯誤），交易日直接用K棒自己的日曆日期分組即可，
不需要處理跨日的session標記問題。

參數選擇說明：min_dominant_side_fraction 直接沿用日盤版本(frac=0.90)
已經驗證過的寬鬆值，**不是針對夜盤資料重新網格搜尋**——夜盤資料只有
2017-05-15起約6.5年，重新開一個網格搜尋在這麼小的樣本上做多重比較，
假陽性風險極高。這裡只用「移植已驗證參數」跟「原始嚴格版本(frac=1.0)」
兩個預先指定的版本直接比較，維持這次會話一貫的紀律。

規則跟 trend_day_strategy.py 幾乎一致，差異只在於：
- 只取 window_start~window_end 之間的K棒（不是整個日盤/夜盤）。
- VWAP 從 window_start 開始累積（不是從當天/當夜開盤）。
- decision_time/session_end 都改成落在這個熱區窗口內。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import time

import numpy as np
import pandas as pd

POINT_VALUE = 50.0


@dataclass
class NightLiquidWindowConfig:
    window_start: time = time(21, 0)
    decision_time: time = time(22, 30)
    window_end: time = time(23, 45)
    min_move_points: float = 20.0
    slippage_points: float = 1.0
    min_dominant_side_fraction: float = 1.0


def _window_frame(df: pd.DataFrame, cfg: NightLiquidWindowConfig) -> pd.DataFrame:
    t = df["datetime"].dt.time
    d = df[(t >= cfg.window_start) & (t <= cfg.window_end)].copy()
    d["trading_date"] = d["datetime"].dt.date
    return d


def backtest(df: pd.DataFrame, cfg: NightLiquidWindowConfig | None = None) -> pd.DataFrame:
    cfg = cfg or NightLiquidWindowConfig()
    win_df = _window_frame(df, cfg)

    trades = []
    for trading_date, g in win_df.groupby("trading_date"):
        g = g.sort_values("datetime").reset_index(drop=True)

        typical_price = (g["high"] + g["low"] + g["close"]) / 3
        cum_pv = (typical_price * g["volume"]).cumsum()
        cum_v = g["volume"].cumsum().replace(0, np.nan)
        g["ref_price"] = cum_pv / cum_v

        t = g["datetime"].dt.time
        decision_mask = t <= cfg.decision_time
        if not decision_mask.any():
            continue
        decision_idx = decision_mask[decision_mask].index[-1]
        if decision_idx + 1 >= len(g):
            continue

        pre = g.loc[: decision_idx]
        side = np.sign(pre["close"] - pre["ref_price"])
        side = side.replace(0, np.nan).dropna()
        if side.empty:
            continue

        counts = side.value_counts()
        dominant_sign = counts.idxmax()
        dominant_fraction = counts.max() / len(side)
        if dominant_fraction < cfg.min_dominant_side_fraction:
            continue

        direction_sign = dominant_sign
        window_open = g["open"].iloc[0]
        decision_close = g["close"].iloc[decision_idx]
        move = (decision_close - window_open) * direction_sign
        if move < cfg.min_move_points:
            continue

        direction = "long" if direction_sign > 0 else "short"
        entry_bar = g.iloc[decision_idx + 1]
        entry_price = entry_bar["open"] + (cfg.slippage_points if direction == "long" else -cfg.slippage_points)
        entry_dt = entry_bar["datetime"]

        exit_price = exit_dt = exit_reason = None
        for j in range(decision_idx + 1, len(g)):
            r = g.iloc[j]
            if t.iloc[j] >= cfg.window_end:
                break
            broke = (r["close"] < r["ref_price"]) if direction == "long" else (r["close"] > r["ref_price"])
            if broke:
                exit_price = r["close"] + (-cfg.slippage_points if direction == "long" else cfg.slippage_points)
                exit_dt, exit_reason = r["datetime"], "vwap_break"
                break

        if exit_price is None:
            close_mask = t >= cfg.window_end
            last_bar = g.loc[close_mask].iloc[0] if close_mask.any() else g.iloc[-1]
            exit_price = last_bar["open"] + (-cfg.slippage_points if direction == "long" else cfg.slippage_points)
            exit_dt, exit_reason = last_bar["datetime"], "window_close"

        pnl_points = (exit_price - entry_price) if direction == "long" else (entry_price - exit_price)
        trades.append(dict(
            trading_date=trading_date, direction=direction, decision_close=decision_close,
            move_at_decision=move, entry_dt=entry_dt, entry_price=entry_price,
            exit_dt=exit_dt, exit_price=exit_price, exit_reason=exit_reason, pnl_points=pnl_points,
        ))

    trades_df = pd.DataFrame(trades)
    if trades_df.empty:
        return trades_df
    trades_df["pnl_twd"] = trades_df["pnl_points"] * POINT_VALUE
    return trades_df.sort_values("entry_dt").reset_index(drop=True)
