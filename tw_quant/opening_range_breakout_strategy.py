"""開盤區間突破（Opening Range Breakout, ORB），日盤當沖、用 1 分鐘 K 棒抓
精確的突破時刻跟停損觸發時刻（跟這次會話稍早的日線唐奇安突破不同：那個
只看日 OHLC，不知道當天哪個時間點觸及水準；這裡直接用 1 分鐘資料，突破/
停損的成交時機是真的用分鐘級別掃出來的，不是日線近似）。

跟這次會話最早測過的「開盤上衝」（08:45 進場->09:00 出場，固定時刻、
不管價格走勢）根本不同路數：那是「不管市場怎麼走，固定時間進出場」，
這裡是「先觀察開盤一段時間的區間，價格真的突破區間才進場，讓獲利單一路
抱到收盤前」——邊際設計上就是要抓「一段時間內的價格延伸」而不是「固定
15分鐘的統計優勢」，理論上單筆邊際會大很多，這樣才禁得起 1 點滑價
（前面已經證實固定時刻策略的 1~2 點邊際完全禁不起滑價）。

下單機制：
- 開盤區間＝range_start~range_end 這段時間的最高/最低點。
- 區間結束後，用 1 分鐘 K 棒逐根掃描，第一根「高點 >= 區間上緣」的分鐘
  觸發做多、第一根「低點 <= 區間下緣」的分鐘觸發做空（同一天只認第一個
  觸發的方向，兩個方向同一根都觸發是極端情況，視為雜訊跳過不交易）。
- 進場用停損單概念，複用 donchian_breakout_strategy._fill_price 同一套
  「跳空用該分鐘開盤價、沒跳空用水準+滑價」邏輯，只是判斷單位從「日」
  換成「分鐘」，精確度更高。
- 停損＝開盤區間的另一端（多單的停損在區間下緣、空單在區間上緣），一樣
  逐分鐘掃描觸發，同一套成交邏輯。
- 沒有額外停利，讓獲利單抱到 session_end（收盤前的強制平倉時間，留一段
  緩衝避免收盤前流動性变差），到點用市價出場（有滑價）。
- 開盤區間太窄（min_range_points）代表當天盤整、雜訊突破風險高，直接
  跳過不交易；同一天只允許一筆部位。

簡化與已知限制：
- 只做日盤（08:45-13:45），不含夜盤——維持跟這次會話其餘策略一致的
  簡化，避免處理夜盤/日盤銜接的額外複雜度。
- 停損只用開盤區間本身的寬度，沒有另外疊加 ATR，是刻意的簡化：ORB
  原始設計就是用區間本身當風控單位，不是外加的波動率量測。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import time

import pandas as pd

from tw_quant.donchian_breakout_strategy import _fill_price

POINT_VALUE = 50.0


@dataclass
class OpeningRangeBreakoutConfig:
    range_start: time = time(8, 45)
    range_end: time = time(9, 15)
    session_end: time = time(13, 25)
    min_range_points: float = 5.0
    slippage_points: float = 1.0
    target_r_multiple: float | None = None  # None=不設固定停利，讓獲利單抱到收盤


def _day_session_frame(df: pd.DataFrame) -> pd.DataFrame:
    t = df["datetime"].dt.time
    d = df[(t >= time(8, 45)) & (t <= time(13, 45))].copy()
    d["trading_date"] = d["datetime"].dt.date
    return d


def backtest(df: pd.DataFrame, cfg: OpeningRangeBreakoutConfig | None = None) -> pd.DataFrame:
    cfg = cfg or OpeningRangeBreakoutConfig()
    day_df = _day_session_frame(df)

    trades = []
    for trading_date, g in day_df.groupby("trading_date"):
        g = g.sort_values("datetime").reset_index(drop=True)
        t = g["datetime"].dt.time

        range_mask = (t >= cfg.range_start) & (t < cfg.range_end)
        if not range_mask.any():
            continue
        range_high = g.loc[range_mask, "high"].max()
        range_low = g.loc[range_mask, "low"].min()
        if range_high - range_low < cfg.min_range_points:
            continue

        after_mask = (t >= cfg.range_end) & (t < cfg.session_end)
        after = g.loc[after_mask].reset_index(drop=True)
        if after.empty:
            continue

        long_trigger = after.index[after["high"] >= range_high]
        short_trigger = after.index[after["low"] <= range_low]
        long_idx = long_trigger[0] if len(long_trigger) else None
        short_idx = short_trigger[0] if len(short_trigger) else None

        if long_idx is None and short_idx is None:
            continue
        if long_idx is not None and short_idx is not None and long_idx == short_idx:
            continue  # 同一根分鐘K棒雙向都觸發，視為雜訊，不交易
        if short_idx is None or (long_idx is not None and long_idx < short_idx):
            direction, entry_idx, entry_level, stop_level = "long", long_idx, range_high, range_low
        else:
            direction, entry_idx, entry_level, stop_level = "short", short_idx, range_low, range_high

        entry_bar = after.iloc[entry_idx]
        side = "buy_stop" if direction == "long" else "sell_stop"
        entry_price, entry_reason = _fill_price(
            entry_level, entry_bar["open"], entry_bar["high"], entry_bar["low"], side, cfg.slippage_points,
        )
        entry_dt = entry_bar["datetime"]

        target = None
        if cfg.target_r_multiple is not None:
            risk = abs(entry_price - stop_level)
            target = entry_price + cfg.target_r_multiple * risk if direction == "long" \
                else entry_price - cfg.target_r_multiple * risk

        exit_price = exit_dt = exit_reason = None
        for j in range(entry_idx + 1, len(after)):
            bar = after.iloc[j]
            if direction == "long":
                if bar["low"] <= stop_level:
                    exit_price, exit_reason = _fill_price(stop_level, bar["open"], bar["high"], bar["low"], "sell_stop", cfg.slippage_points)
                    exit_dt = bar["datetime"]
                    break
                if target is not None and bar["high"] >= target:
                    exit_price, exit_reason = target, "target"
                    exit_dt = bar["datetime"]
                    break
            else:
                if bar["high"] >= stop_level:
                    exit_price, exit_reason = _fill_price(stop_level, bar["open"], bar["high"], bar["low"], "buy_stop", cfg.slippage_points)
                    exit_dt = bar["datetime"]
                    break
                if target is not None and bar["low"] <= target:
                    exit_price, exit_reason = target, "target"
                    exit_dt = bar["datetime"]
                    break

        if exit_price is None:
            close_mask = t >= cfg.session_end
            if not close_mask.any():
                last_bar = g.iloc[-1]
            else:
                last_bar = g.loc[close_mask].iloc[0]
            if direction == "long":
                exit_price = last_bar["open"] - cfg.slippage_points
            else:
                exit_price = last_bar["open"] + cfg.slippage_points
            exit_dt, exit_reason = last_bar["datetime"], "session_close"

        pnl_points = (exit_price - entry_price) if direction == "long" else (entry_price - exit_price)
        trades.append(dict(
            trading_date=trading_date, direction=direction, range_high=range_high, range_low=range_low,
            entry_dt=entry_dt, entry_price=entry_price, entry_reason=entry_reason,
            exit_dt=exit_dt, exit_price=exit_price, exit_reason=exit_reason, pnl_points=pnl_points,
        ))

    trades_df = pd.DataFrame(trades)
    if trades_df.empty:
        return trades_df
    trades_df["pnl_twd"] = trades_df["pnl_points"] * POINT_VALUE
    return trades_df.sort_values("entry_dt").reset_index(drop=True)
