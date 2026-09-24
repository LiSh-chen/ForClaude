"""「大做小」多時間週期均線排列策略：較大週期 MA 多頭排列 + 較小週期 MA 空頭
排列（順勢中的拉回）-> 做多；較大週期空頭排列 + 較小週期多頭排列（順勢中的
反彈）-> 做空。跟 hammer_signal_backtest.py 共用同一套風控/成本框架。

均線多頭/空頭排列定義：3 條均線 fast/mid/slow（預設週期 5/10/20，各自在自己
的時間週期上計算），fast>mid>slow 為多頭排列，fast<mid<slow 為空頭排列，其餘
（順序不一致）視為中性、不參與判斷。

訊號用「邊緣觸發」：大週期方向 + 小週期反向這個組合條件從「不成立」轉成
「成立」的那一刻才算一次訊號，同一段對齊期間不會每分鐘重複觸發。

進場：訊號成立後，於下一根 1 分鐘K棒開盤進場。
風控：停損 = 近 N 根 1 分鐘K棒的最低/最高點 -/+ sl_buffer（做多用最低點、
做空用最高點），risk = 進場價到 SL 的距離，TP = 進場價 + risk * r_multiple
（方向鏡射），若 risk 超過 max_risk_points 就放棄訊號。這個 N 根滾動視窗沒有
在每個連續盤中（block_id）重新歸零——只有每段盤剛開盤的前 N 根K棒會偶爾借用
到前一段盤收盤前的高低點，影響範圍很小（一個 session 開頭 N 分鐘/總長
~500 分鐘），未特別處理。

跟 hammer 策略最大的不同：這裡沒有「次根紅K」這種單根K棒條件，訊號完全由
大小週期的均線排列組合決定，出場沿用同一套 SL/TP/max_hold/block_end 引擎
（見 `tw_quant.hammer_signal_backtest.simulate` 的同款邏輯，這裡重新實作一份
是因為進場條件跟風控基準都不同，直接共用同一個函式反而會讓兩套完全不同的
訊號定義擠在一起、參數語意混淆）。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

MAX_GAP_MINUTES = 2


def _ma_alignment_state(close: pd.Series, fast: int, mid: int, slow: int) -> pd.Series:
    ma_fast = close.rolling(fast).mean()
    ma_mid = close.rolling(mid).mean()
    ma_slow = close.rolling(slow).mean()
    state = pd.Series(0, index=close.index, dtype=int)
    state[(ma_fast > ma_mid) & (ma_mid > ma_slow)] = 1
    state[(ma_fast < ma_mid) & (ma_mid < ma_slow)] = -1
    return state


def build_timeframe_state(df: pd.DataFrame, freq: str, fast: int = 5, mid: int = 10, slow: int = 20) -> np.ndarray:
    """把某個時間週期的均線排列狀態（1=多頭排列/-1=空頭排列/0=中性），依「該根
    K棒收盤後才可得知」的原則（merge_asof + allow_exact_matches=False）回填到
    跟 df 等長的 1 分鐘陣列。"""
    d = df.sort_values("datetime").reset_index(drop=True)
    tf_close = d.set_index("datetime")["close"].resample(freq, label="right", closed="left").last().dropna()
    state = _ma_alignment_state(tf_close, fast, mid, slow)
    state = state[state.index.isin(tf_close.dropna().index)]

    merged = pd.merge_asof(
        d[["datetime"]], state.rename("state").reset_index(),
        on="datetime", direction="backward", allow_exact_matches=False,
    )
    return merged["state"].fillna(0).to_numpy(dtype=int)


def build_base_arrays(df: pd.DataFrame, swing_lookback: int = 20) -> dict:
    d = df.sort_values("datetime").reset_index(drop=True)
    open_ = d["open"].to_numpy(dtype=float)
    high = d["high"].to_numpy(dtype=float)
    low = d["low"].to_numpy(dtype=float)
    close = d["close"].to_numpy(dtype=float)

    gap_min = d["datetime"].diff().dt.total_seconds().to_numpy() / 60
    new_block = (gap_min > MAX_GAP_MINUTES) | np.isnan(gap_min)
    block_id = np.cumsum(new_block)

    roll_low = pd.Series(low).shift(1).rolling(swing_lookback).min().to_numpy()
    roll_high = pd.Series(high).shift(1).rolling(swing_lookback).max().to_numpy()

    next_open = np.append(open_[1:], np.nan)
    next_block = np.append(block_id[1:], -1)

    return dict(
        n=len(d), open=open_, high=high, low=low, close=close, block_id=block_id,
        roll_low=roll_low, roll_high=roll_high, next_open=next_open, next_block=next_block,
    )


def simulate(
    arrays: dict,
    large_state: np.ndarray,
    small_state: np.ndarray,
    r_multiple: float,
    sl_buffer: float = 5.0,
    max_risk_points: float = 140.0,
    max_hold_minutes: int = 60,
) -> pd.DataFrame:
    n = arrays["n"]
    open_, high, low, close = arrays["open"], arrays["high"], arrays["low"], arrays["close"]
    block_id, next_block, next_open = arrays["block_id"], arrays["next_block"], arrays["next_open"]
    roll_low, roll_high = arrays["roll_low"], arrays["roll_high"]

    long_cond = (large_state == 1) & (small_state == -1)
    short_cond = (large_state == -1) & (small_state == 1)
    # 邊緣觸發：組合條件剛成立那一刻才算訊號
    long_edge = long_cond & ~np.concatenate(([False], long_cond[:-1]))
    short_edge = short_cond & ~np.concatenate(([False], short_cond[:-1]))

    direction_at = np.where(long_edge, "long", np.where(short_edge, "short", ""))
    cand_idx = np.flatnonzero(direction_at != "")

    trades = []
    blocked_until = -1
    for i in cand_idx:
        if i <= blocked_until:
            continue
        if next_block[i] != block_id[i]:
            continue  # 訊號那根已經是這段盤最後一根，次根開盤不在同一段連續盤

        direction = direction_at[i]
        entry_idx = i + 1
        entry_price = next_open[i]

        if direction == "long":
            if np.isnan(roll_low[i]):
                continue
            risk = entry_price - (roll_low[i] - sl_buffer)
        else:
            if np.isnan(roll_high[i]):
                continue
            risk = (roll_high[i] + sl_buffer) - entry_price

        if not (0 < risk <= max_risk_points):
            continue

        if direction == "long":
            sl = entry_price - risk
            tp = entry_price + risk * r_multiple
        else:
            sl = entry_price + risk
            tp = entry_price - risk * r_multiple

        limit = min(entry_idx + max_hold_minutes, n - 1)
        exit_idx = exit_price = exit_reason = None
        for j in range(entry_idx + 1, limit + 1):
            if block_id[j] != block_id[entry_idx]:
                exit_idx, exit_price, exit_reason = j - 1, close[j - 1], "block_end"
                break
            if direction == "long":
                if low[j] <= sl:
                    exit_idx, exit_price, exit_reason = j, sl, "stop_loss"
                    break
                if high[j] >= tp:
                    exit_idx, exit_price, exit_reason = j, tp, "take_profit"
                    break
            else:
                if high[j] >= sl:
                    exit_idx, exit_price, exit_reason = j, sl, "stop_loss"
                    break
                if low[j] <= tp:
                    exit_idx, exit_price, exit_reason = j, tp, "take_profit"
                    break
        if exit_idx is None:
            exit_idx, exit_price, exit_reason = limit, close[limit], "max_hold"

        pnl_points = (exit_price - entry_price) if direction == "long" else (entry_price - exit_price)
        trades.append(dict(
            signal_idx=i, direction=direction, entry_idx=entry_idx, exit_idx=exit_idx,
            entry_price=entry_price, exit_price=exit_price, sl=sl, tp=tp,
            risk_points=risk, exit_reason=exit_reason, pnl_points=pnl_points,
        ))
        blocked_until = exit_idx

    return pd.DataFrame(trades)
