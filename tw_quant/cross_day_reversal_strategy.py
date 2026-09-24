"""跨日日盤報酬反轉（日盤當沖，完全不需要任何連續運算，只需要記住昨天一個
數字）：昨天整個日盤（08:45開盤->13:45收盤）報酬是正的，今天開盤做空；
昨天是負的，今天開盤做多；今天收盤前平倉。

這是這次會話目前為止結構最簡單的候選：不看任何當天內部走勢、不用任何
移動水準或指標，訊號在前一天收盤時就已經完全確定，今天開盤就能照表
操課——跟趨勢日系列（VWAP/開盤價/ATR）需要在當天盤中即時判斷完全不同，
是「跨日記憶」而不是「當天內部動態」的訊號來源。

探索過程（見 scripts/scan_cross_day_momentum.py）發現：整個日盤的跨日
自相關是負的（延續策略賠錢、反轉策略賺錢），也就是「今天noon前的方向
容易反轉昨天的方向」，這跟其他時段（開盤前15分鐘、午盤前半、尾盤）比
起來是唯一一個樣本內全樣本達到 |t|>=2 的候選，但子區間穩健性偏弱
（只有2001-2005顯著，其餘三個子區間弱但方向一致、沒有反過來），OOS
也還沒有到統計顯著（見下方回測腳本結果）——性質介於已經驗證的策略跟
純雜訊之間，是需要更多證據才能下定論的候選，不是穩健已驗證的策略。

下單機制：
- 訊號：昨天日盤報酬（昨收-昨開）的正負號，反向操作。
- 進場：今天08:45(或當天日盤第一根K棒)開盤，市價單，扣滑價。
- 出場：今天13:45前收盤，市價單，扣滑價（用session_end附近那根K棒的
  開盤價模擬，避免真的用到最後一筆的極端報價）。
- 每天都會有訊號（除非前一天沒有資料或幅度太小被min_abs_yesterday_return
  濾掉），是高頻率、低選擇性的設計，跟趨勢日系列的低頻高選擇性相反。

簡化與已知限制：
- 只看日盤，不含夜盤（夜盤到日盤之間的跳空完全沒有反映在訊號裡）。
- 沒有停損：理論上單筆虧損只受限於當天日盤的最大不利波動，沒有主動
  停損機制，是已知的簡化。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import time

import pandas as pd

POINT_VALUE = 50.0


@dataclass
class CrossDayReversalConfig:
    slippage_points: float = 1.0
    min_abs_yesterday_return: float = 0.0
    session_end: time = time(13, 44)


def _day_session_frame(df: pd.DataFrame) -> pd.DataFrame:
    t = df["datetime"].dt.time
    d = df[(t >= time(8, 45)) & (t <= time(13, 45))].copy()
    d["trading_date"] = d["datetime"].dt.date
    return d


def backtest(df: pd.DataFrame, cfg: CrossDayReversalConfig | None = None) -> pd.DataFrame:
    cfg = cfg or CrossDayReversalConfig()
    day_df = _day_session_frame(df)

    daily = day_df.groupby("trading_date").agg(open=("open", "first"), close=("close", "last")).reset_index()
    daily["trading_date"] = pd.to_datetime(daily["trading_date"])
    daily = daily.sort_values("trading_date").reset_index(drop=True)
    daily["day_return"] = daily["close"] - daily["open"]
    daily["prior_return"] = daily["day_return"].shift(1)

    trades = []
    for _, row in daily.iterrows():
        prior = row["prior_return"]
        if pd.isna(prior) or abs(prior) < cfg.min_abs_yesterday_return:
            continue
        direction = "short" if prior > 0 else "long"

        entry_price = row["open"] + (cfg.slippage_points if direction == "long" else -cfg.slippage_points)
        exit_price = row["close"] + (-cfg.slippage_points if direction == "long" else cfg.slippage_points)
        pnl_points = (exit_price - entry_price) if direction == "long" else (entry_price - exit_price)

        trades.append(dict(
            trading_date=row["trading_date"], direction=direction, prior_return=prior,
            entry_price=entry_price, exit_price=exit_price, pnl_points=pnl_points,
        ))

    trades_df = pd.DataFrame(trades)
    if trades_df.empty:
        return trades_df
    trades_df["pnl_twd"] = trades_df["pnl_points"] * POINT_VALUE
    return trades_df.sort_values("trading_date").reset_index(drop=True)
