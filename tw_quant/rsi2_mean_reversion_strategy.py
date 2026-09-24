"""RSI(2) 短期均值回歸（Connors 風格），日線、多空雙向、通常持有數天。

跟這次會話目前為止測過的所有東西都不同路數：
- 跟開盤上衝/午盤放空/盤中翻多那三個「日內固定時刻」策略不同：這裡完全
  不管日內任何時間點，訊號跟出場都只看「日收盤」算出來，用隔天開盤
  執行，單筆邊際是幾十到上百點量級，不是 1~2 點——1 點滑價占的比例
  遠小於那些日內策略，理論上禁得起滑價。
- 跟唐奇安突破（donchian_breakout_strategy.py）不同：那是趨勢跟隨（順著
  突破方向進場），這裡是均值回歸（在多頭排列格局下找短期超跌反彈、在
  空頭排列格局下找短期超漲拉回）——訊號生成邏輯完全不同源，不是同一個
  策略的變形。

下單機制：訊號用「當天收盤」判斷完成後，只能在「隔天開盤」執行（收盤當下
不可能同時是隔天的開盤價，這是避免未來函數的基本要求）。開盤是市價單
概念，不是停損單，所以不會有「跳空穿越水準」這種情境，固定用「當天開盤
價 + slippage_points（買進時滑價讓成交價變差、賣出時反向）」模擬。

規則：
- 多單濾網：收盤價 > trend_ma_window 日均線（只在中長期多頭排列格局下
  找超跌反彈）。
- 多單訊號：RSI(rsi_period) < oversold。
- 多單出場：RSI(rsi_period) > overbought，或持有超過 max_hold_days 天，
  兩者皆用「隔天開盤」執行。
- 空單為多單鏡射（收盤 < 均線 且 RSI > 100-oversold 時放空，RSI 回落到
  100-overbought 或超過 max_hold_days 出場）。
- 同一時間只允許一筆部位；沒有額外停損（照 Connors 原始設計，靠均值回歸
  本身的高勝率跟短持有期控制風險，不额外加停損以免破壞訊號本身的統計
  特性——這是已知簡化，見檔尾）。

簡化與已知限制：
- 只用日盤日K（跟唐奇安突破模組一致的簡化，夜盤波動不影響訊號/出場）。
- 沒有額外停損，意味著單筆最大虧損理論上不封頂（只受 max_hold_days
  限制天數，不限金額）——如果這個策略要進一步演化成實盤版本，這是需要
  額外處理的風險點，但先照經典設計測出「有沒有邊際」再談風控加碼。
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

POINT_VALUE = 50.0


@dataclass
class Rsi2Config:
    rsi_period: int = 2
    oversold: float = 10.0
    overbought: float = 70.0
    trend_ma_window: int = 200
    max_hold_days: int = 10
    slippage_points: float = 1.0


def _rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, float("nan"))
    rsi = 100 - 100 / (1 + rs)
    return rsi.fillna(100)  # avg_loss=0 代表連續上漲，RSI 定義上應為 100


def compute_indicators(daily: pd.DataFrame, cfg: Rsi2Config) -> pd.DataFrame:
    d = daily.sort_values("date").reset_index(drop=True).copy()
    d["rsi"] = _rsi(d["close"], cfg.rsi_period)
    d["trend_ma"] = d["close"].rolling(cfg.trend_ma_window).mean()
    return d


def backtest(df: pd.DataFrame, cfg: Rsi2Config | None = None) -> pd.DataFrame:
    from tw_quant.technical_indicators import build_daily_bars

    cfg = cfg or Rsi2Config()
    daily = compute_indicators(build_daily_bars(df), cfg)
    n = len(daily)

    trades = []
    i = 0
    while i < n - 1:  # 最後一天沒有隔天可以進場，不掃描
        row = daily.iloc[i]
        if pd.isna(row["trend_ma"]) or pd.isna(row["rsi"]):
            i += 1
            continue

        direction = None
        if row["close"] > row["trend_ma"] and row["rsi"] < cfg.oversold:
            direction = "long"
        elif row["close"] < row["trend_ma"] and row["rsi"] > (100 - cfg.oversold):
            direction = "short"

        if direction is None:
            i += 1
            continue

        entry_idx = i + 1
        entry_row = daily.iloc[entry_idx]
        if direction == "long":
            entry_price = entry_row["open"] + cfg.slippage_points
        else:
            entry_price = entry_row["open"] - cfg.slippage_points
        entry_date = entry_row["date"]

        exit_idx = None
        j_limit = min(entry_idx + cfg.max_hold_days, n - 1)
        j = entry_idx
        while j < j_limit:
            r = daily.iloc[j]
            exit_signal = (direction == "long" and r["rsi"] > cfg.overbought) or \
                          (direction == "short" and r["rsi"] < (100 - cfg.overbought))
            if exit_signal:
                exit_idx = j + 1
                exit_reason = "rsi_reversion"
                break
            j += 1
        if exit_idx is None:
            exit_idx = j_limit
            exit_reason = "max_hold"

        exit_row = daily.iloc[exit_idx]
        if direction == "long":
            exit_price = exit_row["open"] - cfg.slippage_points
        else:
            exit_price = exit_row["open"] + cfg.slippage_points
        exit_date = exit_row["date"]

        pnl_points = (exit_price - entry_price) if direction == "long" else (entry_price - exit_price)
        trades.append(dict(
            direction=direction, signal_date=row["date"], entry_date=entry_date, entry_price=entry_price,
            exit_date=exit_date, exit_price=exit_price, exit_reason=exit_reason, pnl_points=pnl_points,
        ))
        i = exit_idx  # 出場當天才能再找下一筆訊號（同一時間只允許一筆部位）

    trades_df = pd.DataFrame(trades)
    if trades_df.empty:
        return trades_df
    trades_df["pnl_twd"] = trades_df["pnl_points"] * POINT_VALUE
    return trades_df.sort_values("entry_date").reset_index(drop=True)
