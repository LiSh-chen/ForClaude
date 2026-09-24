"""趨勢日偵測（Trend Day Detection），日盤當沖、用 1 分鐘資料判斷「今天是
不是趨勢盤」再進場，跟開盤區間突破（ORB，已經驗證失敗——OOS轉負）根本
不同的篩選邏輯：

ORB 的問題：只要價格「摸到」開盤區間邊界一次就進場，抓到的是很多「早盤
噴出一下馬上被巴」的假突破日，這種日子在越來越有效率的市場裡越來越常見
（這次會話已經證實 ORB 的邊際逐年衰退到轉負）。

這裡改用更嚴格的篩選：不是「有沒有突破」，是「有沒有持續維持在盤中VWAP
（成交量加權均價，從當天開盤累積算）同一側，且已經走出有意義的幅度」。
邏輯是：真正的趨勢日，價格會早早站上（或跌破）VWAP 之後就不再回頭測試，
盤整/假突破日則會在VWAP兩側來回穿梭——用「到判斷時點為止有沒有回頭
穿越過 VWAP」當篩選器，比單純的區間邊界突破更嚴格，理論上能濾掉大部分
的假訊號日，只留下真正單邊控盤的那些日子。

下單機制：
- decision_time（預設11:00）之前，逐分鐘算「當天累積 VWAP」跟收盤價的
  相對位置；如果整段期間收盤價都在 VWAP 同一側（沒有反向穿越過），且
  從開盤到 decision_time 的累積移動幅度 >= min_move_points，判定為
  「趨勢日候選」，方向＝VWAP的那一側。
- 進場：decision_time 那根K棒收盤後，用下一根K棒的開盤價成交（市價單
  概念，讓決策跟成交分開，避免用同一根K棒的資訊進出場），扣滑價。
- 出場（趨勢失效停損）：進場後任何一分鐘，如果收盤價穿越回 VWAP 的
  反向那一側，視為趨勢假設失效，當根K棒收盤價成交出場（市價單，扣
  滑價）——VWAP 本身每分鐘都在變動，這是「移動中的停損水準」，不是
  固定停損單，是系統化策略常見的做法（跟固定停損單機制不同，這裡簡化
  用收盤價判斷觸發跟成交，不逐檔模擬停損單掛單）。
- 沒有觸發停損就抱到 session_end（收盤前強制平倉），市價出場、扣滑價。
- 同一天只交易一次（決策點只評估一次），沒有趨勢日候選的日子完全不交易。

簡化與已知限制：
- 只做日盤（08:45-13:45），不含夜盤。
- VWAP 停損用收盤價判斷+同根K棒收盤價成交，是簡化（沒有模擬停損單的
  跳空/滑價機制，因為 VWAP 是連續變動的水準，不適合套用固定水準的
  停損單模型）；正常滑價仍然扣在出場成交價上。
- 「有沒有回頭穿越VWAP」用收盤價逐分鐘檢查，不用高低點（用高低點會
  太敏感，稍微碰一下就判定失效，不符合「趨勢日」通常允許小幅拉回但
  不破均價的現實）。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import time

import numpy as np
import pandas as pd

POINT_VALUE = 50.0


@dataclass
class TrendDayConfig:
    decision_time: time = time(11, 0)
    min_move_points: float = 20.0
    session_end: time = time(13, 25)
    slippage_points: float = 1.0
    min_dominant_side_fraction: float = 1.0  # 1.0=原始版本「決策時點前完全沒穿越過VWAP」
    # <1.0＝放寬：允許決策時點前有一部分分鐘K棒收在VWAP反向側（雜訊型短暫拉回），
    # 只要「多數方向」那一側的比例達到這個門檻就算趨勢日候選，方向＝多數方向。


def _day_session_frame(df: pd.DataFrame) -> pd.DataFrame:
    t = df["datetime"].dt.time
    d = df[(t >= time(8, 45)) & (t <= time(13, 45))].copy()
    d["trading_date"] = d["datetime"].dt.date
    return d


def backtest(df: pd.DataFrame, cfg: TrendDayConfig | None = None) -> pd.DataFrame:
    cfg = cfg or TrendDayConfig()
    day_df = _day_session_frame(df)

    trades = []
    for trading_date, g in day_df.groupby("trading_date"):
        g = g.sort_values("datetime").reset_index(drop=True)
        typical_price = (g["high"] + g["low"] + g["close"]) / 3
        cum_pv = (typical_price * g["volume"]).cumsum()
        cum_v = g["volume"].cumsum().replace(0, np.nan)
        g["vwap"] = cum_pv / cum_v

        t = g["datetime"].dt.time
        decision_mask = t <= cfg.decision_time
        if not decision_mask.any():
            continue
        decision_idx = decision_mask[decision_mask].index[-1]
        if decision_idx + 1 >= len(g):
            continue  # 判斷時點之後沒有下一根K棒可以進場，跳過

        pre = g.loc[: decision_idx]
        side = np.sign(pre["close"] - pre["vwap"])
        side = side.replace(0, np.nan).dropna()
        if side.empty:
            continue

        counts = side.value_counts()
        dominant_sign = counts.idxmax()
        dominant_fraction = counts.max() / len(side)
        if dominant_fraction < cfg.min_dominant_side_fraction:
            continue  # 反向側的比例超過容許範圍，不是趨勢日候選

        direction_sign = dominant_sign
        day_open = g["open"].iloc[0]
        decision_close = g["close"].iloc[decision_idx]
        move = (decision_close - day_open) * direction_sign
        if move < cfg.min_move_points:
            continue

        direction = "long" if direction_sign > 0 else "short"
        entry_bar = g.iloc[decision_idx + 1]
        entry_price = entry_bar["open"] + (cfg.slippage_points if direction == "long" else -cfg.slippage_points)
        entry_dt = entry_bar["datetime"]

        exit_price = exit_dt = exit_reason = None
        for j in range(decision_idx + 1, len(g)):
            r = g.iloc[j]
            if t.iloc[j] >= cfg.session_end:
                break
            broke = (r["close"] < r["vwap"]) if direction == "long" else (r["close"] > r["vwap"])
            if broke:
                exit_price = r["close"] + (-cfg.slippage_points if direction == "long" else cfg.slippage_points)
                exit_dt, exit_reason = r["datetime"], "vwap_break"
                break

        if exit_price is None:
            close_mask = t >= cfg.session_end
            last_bar = g.loc[close_mask].iloc[0] if close_mask.any() else g.iloc[-1]
            exit_price = last_bar["open"] + (-cfg.slippage_points if direction == "long" else cfg.slippage_points)
            exit_dt, exit_reason = last_bar["datetime"], "session_close"

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
