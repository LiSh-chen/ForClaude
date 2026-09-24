"""唐奇安通道突破 + ATR 移動停損（趨勢跟隨，日線、多空雙向、可能持有數天到數週）。

跟這次會話前面所有「日內固定時刻」策略的根本差異：訊號跟出場點位是價格
突破算出來的，不是排定時刻，單筆邊際是幾十到上百點（ATR 量級），不是
1~2 點——這樣滑價（我們已經證實 1~2 點滑價就能吃光前面幾個策略的邊際）
佔的比例就小很多。

下單機制的具體假設（都用日盤 08:45-13:45 的日K，不含夜盤資料——夜盤價格
波動不會觸發或影響這裡的訊號/停損，這是簡化，見檔尾）：

- 進場／出場都用「停損單」概念：訊號一旦算出訊號水準（唐奇安上/下軌、
  ATR 停損價），就等同於已經掛在市場上的停損單，一旦當天價格觸及該水準
  就視為成交。
- 正常滑價：當天開盤沒有跳空穿過水準時，假設用「水準價 + slippage_points」
  成交（對做多進場/做空出場而言，slippage 讓成交價變差；方向依買賣鏡射）。
- 跳空滑價：如果當天開盤價本身已經跳空穿過水準（開盤前就已經有一堆停損單
  排隊，等第一筆成交時市場已經走掉），改用「當天開盤價」成交——這通常比
  「水準價+滑價」更差，是停損單最大的風險來源，且沒有額外再加滑價（開盤
  價本身已經反映當時的真實成交價，不重複扣）。
- 同一時間只允許一筆部位（多或空）；只有出場後才能因為新訊號再進場。

風控：
- 進場停損（初始）＝進場價 -/+ atr_stop_mult * ATR(atr_window)（多單減、空單加）。
- 出場另外有「吊燈移動停損」：多單停損只會往上調整（trail_lookback 天最高
  收盤 - atr_stop_mult*ATR），從不往下鬆動；空單鏡射。
- 沒有固定停利，讓獲利單靠移動停損自然出場（標準趨勢跟隨做法）；
  max_hold_days 是安全網，避免部位無限期卡著。

簡化與已知限制：
- 只用日盤日K；夜盤的價格波動、跳空完全沒有反映在訊號或停損觸發上，
  跟這次會話其餘策略一致採用的簡化。
- 判斷「今天有沒有觸及某個水準」只用當天的日 high/low 跟水準比較，不去查
  1 分鐘資料精確算「當天幾點觸及」——對日線持倉數天的策略，觸及當天用
  收盤附近價位近似即可，不影響策略本質（跟日內策略不同，日內策略的進出場
  時刻本身就是訊號的一部分，這裡不是）。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

POINT_VALUE = 50.0


@dataclass
class DonchianConfig:
    entry_window: int = 20      # 唐奇安突破：進場用的回看天數
    exit_window: int = 10       # 唐奇安出場：吊燈移動停損参考的回看天數
    atr_window: int = 20
    atr_stop_mult: float = 2.0  # 初始停損距離 = atr_stop_mult * ATR
    max_hold_days: int = 60
    slippage_points: float = 1.0  # 正常滑價（非跳空情況下，單邊）


def _atr(daily: pd.DataFrame, window: int) -> pd.Series:
    prev_close = daily["close"].shift(1)
    tr = pd.concat([
        daily["high"] - daily["low"],
        (daily["high"] - prev_close).abs(),
        (daily["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / window, adjust=False).mean()


def compute_indicators(daily: pd.DataFrame, cfg: DonchianConfig) -> pd.DataFrame:
    d = daily.sort_values("date").reset_index(drop=True).copy()
    d["atr"] = _atr(d, cfg.atr_window)
    # 全部用「前一天為止」的資料，不含當天，避免未來函數
    d["upper_entry"] = d["high"].shift(1).rolling(cfg.entry_window).max()
    d["lower_entry"] = d["low"].shift(1).rolling(cfg.entry_window).min()
    d["upper_exit_ref"] = d["high"].shift(1).rolling(cfg.exit_window).max()
    d["lower_exit_ref"] = d["low"].shift(1).rolling(cfg.exit_window).min()
    return d


def _fill_price(level: float, day_open: float, day_high: float, day_low: float,
                 side: str, slippage: float) -> tuple[float, str]:
    """side='buy_stop'（做多進場/空單出場觸及上方水準時買進）或
    'sell_stop'（做空進場/多單出場觸及下方水準時賣出）。回傳 (成交價, 原因)。"""
    if side == "buy_stop":
        if day_high < level:
            raise ValueError("未觸及水準，不該呼叫成交")
        if day_open >= level:
            return day_open, "gap_through"
        return level + slippage, "normal_slippage"
    else:
        if day_low > level:
            raise ValueError("未觸及水準，不該呼叫成交")
        if day_open <= level:
            return day_open, "gap_through"
        return level - slippage, "normal_slippage"


def backtest(df: pd.DataFrame, cfg: DonchianConfig | None = None) -> pd.DataFrame:
    from tw_quant.technical_indicators import build_daily_bars

    cfg = cfg or DonchianConfig()
    daily = compute_indicators(build_daily_bars(df), cfg)
    n = len(daily)

    trades = []
    i = 0
    while i < n:
        row = daily.iloc[i]
        if pd.isna(row["upper_entry"]) or pd.isna(row["atr"]):
            i += 1
            continue

        direction = None
        if row["high"] >= row["upper_entry"]:
            direction = "long"
            entry_level = row["upper_entry"]
        elif row["low"] <= row["lower_entry"]:
            direction = "short"
            entry_level = row["lower_entry"]

        if direction is None:
            i += 1
            continue

        side = "buy_stop" if direction == "long" else "sell_stop"
        entry_price, entry_reason = _fill_price(
            entry_level, row["open"], row["high"], row["low"], side, cfg.slippage_points,
        )
        entry_date = row["date"]
        atr_at_entry = row["atr"]

        if direction == "long":
            stop = entry_price - cfg.atr_stop_mult * atr_at_entry
        else:
            stop = entry_price + cfg.atr_stop_mult * atr_at_entry

        exit_price = exit_date = exit_reason = None
        j_limit = min(i + cfg.max_hold_days, n - 1)
        j = i + 1
        while j <= j_limit:
            r = daily.iloc[j]
            if direction == "long":
                if pd.notna(r["lower_exit_ref"]):
                    chandelier = r["lower_exit_ref"]
                    stop = max(stop, chandelier)  # 只會往上調整，不會鬆動
                if r["low"] <= stop:
                    exit_price, exit_reason = _fill_price(stop, r["open"], r["high"], r["low"], "sell_stop", cfg.slippage_points)
                    exit_date = r["date"]
                    break
            else:
                if pd.notna(r["upper_exit_ref"]):
                    chandelier = r["upper_exit_ref"]
                    stop = min(stop, chandelier)
                if r["high"] >= stop:
                    exit_price, exit_reason = _fill_price(stop, r["open"], r["high"], r["low"], "buy_stop", cfg.slippage_points)
                    exit_date = r["date"]
                    break
            j += 1

        if exit_price is None:
            last = daily.iloc[j_limit]
            exit_price, exit_date, exit_reason = last["close"], last["date"], "max_hold"
            exit_idx = j_limit
        else:
            exit_idx = int(daily.index[daily["date"] == exit_date][0])

        pnl_points = (exit_price - entry_price) if direction == "long" else (entry_price - exit_price)
        trades.append(dict(
            direction=direction, entry_date=entry_date, entry_price=entry_price, entry_reason=entry_reason,
            exit_date=exit_date, exit_price=exit_price, exit_reason=exit_reason, pnl_points=pnl_points,
        ))
        i = exit_idx + 1

    trades_df = pd.DataFrame(trades)
    if trades_df.empty:
        return trades_df
    trades_df["pnl_twd"] = trades_df["pnl_points"] * POINT_VALUE
    return trades_df.sort_values("entry_date").reset_index(drop=True)
