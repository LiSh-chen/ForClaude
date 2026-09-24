"""流動性掃單反轉（ICT/SMC核心概念的可回測操作化版本：Liquidity Sweep +
Reclaim，日盤當沖偵測進場、可跨日持有）。

背景與誠實評估：
ICT（Inner Circle Trader）/SMC（Smart Money Concepts）是近年散戶圈流行
的交易框架，核心敘事是「模擬機構真實下單行為」，包含市場結構
（BOS/CHoCH）、流動性掃單（Liquidity Sweep）、訂單塊（Order Block）、
公允價值缺口（FVG）、溢價/折價區（Premium/Discount + Fibonacci OTE）、
時間窗（Kill Zone）等一整套概念。這套框架**沒有嚴謹的學術/量化實證
基礎**，敘事包裝了不少傳統技術分析的既有概念：Order Block本質上就是
支撐壓力回測（跟 support_resistance_fade_strategy.py 的 "fade" 模式
高度重疊，而那個模式已經在這次會話被系統性驗證為顯著負值，288組合網
格中27/28個fade候選全部顯著虧損），FVG本質上是缺口回補。

這個模組刻意只操作化「流動性掃單反轉」這個跟現有測試**不完全重疊**的
子概念：不是「碰到支撐/壓力就進場」（那是已經被推翻的fade邏輯），而是
「先真正刺穿水準一段距離（掃到停損池），然後在同一天內收回水準內
（確認假突破/停損獵殺），才反向進場」——多了「刺穿幅度」跟「同日收回」
兩個額外的確認條件，理論上有機會濾掉fade版本進場過早、被真突破巴頭
的那些虧損交易。BOS/CHoCH、Order Block回測進場、FVG、Fibonacci
OTE、Kill Zone時間窗等其餘ICT概念本次不納入（要嘛跟已驗證失敗的邏輯
重疊，要嘛定義太主觀不易嚴謹回測），如果這個核心子概念測出真實邊際，
再考慮疊加其他濾網。

規則：
- 用過去 channel_window 個交易日（不含當天）的最高/最低點當作「流動性池」
  水準（跟 support_resistance_fade_strategy.py 同一套資料結構）。
- 用當天 1 分鐘資料逐分鐘掃描（只做日盤 08:45-13:45，跟這次會話其餘
  日盤策略一致）：
  - 多方訊號：當天累積最低點跌破支撐水準達 sweep_buffer_points（真正
    刺穿一段距離，不是雜訊觸碰），且隨後某根K棒收盤價收回支撐之上
    （確認假跌破），觸發訊號。
  - 空方訊號：鏡射（突破壓力後收回壓力之下）。
  - 同一天如果多空訊號都出現，取時間較早的那個（同一天只進一筆）。
- 進場：訊號確認K棒的下一根K棒開盤價成交（市價單概念，決策與成交分開，
  避免用同一根K棒資訊進出場，跟 trend_day_strategy.py 同一套慣例），
  扣滑價。
- 停損：訊號確認前「掃單當下的極值」（多單＝當天目前為止的最低點）再
  加 stop_buffer_points 緩衝，跌破用停損單邏輯出場
  （donchian_breakout_strategy._fill_price，跳空用開盤價、沒跳空用
  水準+滑價）。
- 停利：風險距離（進場價-停損價）的 target_r_multiple 倍，限價單邏輯
  出場（support_resistance_fade_strategy._limit_fill，價格到了才成交、
  跳空穿越算價格改善，沒有滑價）。
- 進場當天剩餘的分鐘K棒先用1分鐘資料檢查停損/停利（精確捕捉當天內
  來回巴動），隔天之後改用日K的高低點檢查（跟其餘波段類策略一致的
  簡化），max_hold_days 到期還沒出場則用收盤價出場（市價單概念，扣
  滑價）。
- 同一時間只持有一筆部位。

簡化與已知限制：
- 只用日盤資料，不含夜盤（跟這次會話其餘日線/當沖策略一致的簡化，
  夜盤版本的VWAP趨勢日概念已經測過、結果是乾淨的零信號）。
- 「收回水準內」只看收盤價，不看是否真的出現典型反轉K棒型態（真正的
  ICT分析會再疊加CHoCH/K棒型態確認，這裡簡化成單純的價格收回，避免
  引入更多主觀判斷）。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import time

import pandas as pd

from tw_quant.donchian_breakout_strategy import _fill_price
from tw_quant.support_resistance_fade_strategy import _limit_fill

POINT_VALUE = 50.0


@dataclass
class LiquiditySweepConfig:
    channel_window: int = 20
    sweep_buffer_points: float = 10.0
    stop_buffer_points: float = 5.0
    target_r_multiple: float = 2.0
    max_hold_days: int = 10
    slippage_points: float = 1.0


def _day_session_frame(df: pd.DataFrame) -> pd.DataFrame:
    t = df["datetime"].dt.time
    d = df[(t >= time(8, 45)) & (t <= time(13, 45))].copy()
    d["trading_date"] = d["datetime"].dt.date
    return d


def _levels(df: pd.DataFrame, cfg: LiquiditySweepConfig) -> pd.DataFrame:
    from tw_quant.technical_indicators import build_daily_bars

    daily = build_daily_bars(df).sort_values("date").reset_index(drop=True)
    daily["resistance"] = daily["high"].shift(1).rolling(cfg.channel_window).max()
    daily["support"] = daily["low"].shift(1).rolling(cfg.channel_window).min()
    return daily


def backtest(df: pd.DataFrame, cfg: LiquiditySweepConfig | None = None) -> pd.DataFrame:
    cfg = cfg or LiquiditySweepConfig()
    day_df = _day_session_frame(df)
    daily = _levels(df, cfg)
    levels = daily.set_index(daily["date"].dt.date)

    date_groups = {d: g.sort_values("datetime").reset_index(drop=True) for d, g in day_df.groupby("trading_date")}
    dates = sorted(date_groups.keys())
    n_dates = len(dates)

    trades = []
    i = 0
    while i < n_dates:
        d = dates[i]
        if d not in levels.index:
            i += 1
            continue
        row = levels.loc[d]
        support, resistance = row["support"], row["resistance"]
        if pd.isna(support) or pd.isna(resistance):
            i += 1
            continue

        g = date_groups[d]
        if len(g) < 3:
            i += 1
            continue

        running_low = g["low"].cummin()
        running_high = g["high"].cummax()
        swept_support = running_low <= (support - cfg.sweep_buffer_points)
        swept_resistance = running_high >= (resistance + cfg.sweep_buffer_points)
        reclaim_long = swept_support & (g["close"] > support)
        reclaim_short = swept_resistance & (g["close"] < resistance)

        long_idx = reclaim_long[reclaim_long].index.min() if reclaim_long.any() else None
        short_idx = reclaim_short[reclaim_short].index.min() if reclaim_short.any() else None

        if long_idx is not None and (short_idx is None or long_idx <= short_idx):
            signal_idx, direction = long_idx, "long"
        elif short_idx is not None:
            signal_idx, direction = short_idx, "short"
        else:
            i += 1
            continue

        if signal_idx + 1 >= len(g):
            i += 1
            continue

        entry_bar = g.iloc[signal_idx + 1]
        entry_price = entry_bar["open"] + (cfg.slippage_points if direction == "long" else -cfg.slippage_points)
        entry_dt = entry_bar["datetime"]

        if direction == "long":
            sweep_extreme = g.loc[: signal_idx, "low"].min()
            stop = sweep_extreme - cfg.stop_buffer_points
        else:
            sweep_extreme = g.loc[: signal_idx, "high"].max()
            stop = sweep_extreme + cfg.stop_buffer_points

        if (direction == "long" and entry_price <= stop) or (direction == "short" and entry_price >= stop):
            i += 1
            continue  # 開盤已經跳空穿越停損，訊號失效，不進場（避免risk<=0的異常單）

        risk = abs(entry_price - stop)
        target = entry_price + cfg.target_r_multiple * risk if direction == "long" \
            else entry_price - cfg.target_r_multiple * risk

        exit_price = exit_dt = exit_reason = None
        exit_date = d

        for j in range(signal_idx + 2, len(g)):
            r = g.iloc[j]
            if direction == "long":
                if r["low"] <= stop:
                    exit_price, _ = _fill_price(stop, r["open"], r["high"], r["low"], "sell_stop", cfg.slippage_points)
                    exit_dt, exit_reason = r["datetime"], "stop"
                    break
                if r["high"] >= target:
                    exit_price = _limit_fill(target, r["open"], "sell_limit")
                    exit_dt, exit_reason = r["datetime"], "target"
                    break
            else:
                if r["high"] >= stop:
                    exit_price, _ = _fill_price(stop, r["open"], r["high"], r["low"], "buy_stop", cfg.slippage_points)
                    exit_dt, exit_reason = r["datetime"], "stop"
                    break
                if r["low"] <= target:
                    exit_price = _limit_fill(target, r["open"], "buy_limit")
                    exit_dt, exit_reason = r["datetime"], "target"
                    break

        j_date_idx = i
        hold_days_used = 0
        while exit_price is None and hold_days_used < cfg.max_hold_days:
            j_date_idx += 1
            if j_date_idx >= n_dates:
                break
            nd = dates[j_date_idx]
            if nd not in levels.index:
                continue
            drow = levels.loc[nd]
            hold_days_used += 1
            if direction == "long":
                if drow["low"] <= stop:
                    exit_price, _ = _fill_price(stop, drow["open"], drow["high"], drow["low"], "sell_stop", cfg.slippage_points)
                    exit_dt, exit_reason, exit_date = drow["date"], "stop", nd
                    break
                if drow["high"] >= target:
                    exit_price = _limit_fill(target, drow["open"], "sell_limit")
                    exit_dt, exit_reason, exit_date = drow["date"], "target", nd
                    break
            else:
                if drow["high"] >= stop:
                    exit_price, _ = _fill_price(stop, drow["open"], drow["high"], drow["low"], "buy_stop", cfg.slippage_points)
                    exit_dt, exit_reason, exit_date = drow["date"], "stop", nd
                    break
                if drow["low"] <= target:
                    exit_price = _limit_fill(target, drow["open"], "buy_limit")
                    exit_dt, exit_reason, exit_date = drow["date"], "target", nd
                    break

        if exit_price is None:
            last_date_idx = min(i + cfg.max_hold_days, n_dates - 1)
            last_date = dates[last_date_idx]
            last_close = levels.loc[last_date, "close"] if last_date in levels.index else entry_price
            exit_price = last_close - cfg.slippage_points if direction == "long" else last_close + cfg.slippage_points
            exit_dt, exit_reason, exit_date = pd.Timestamp(last_date), "max_hold", last_date
            j_date_idx = last_date_idx

        pnl_points = (exit_price - entry_price) if direction == "long" else (entry_price - exit_price)
        trades.append(dict(
            entry_date=d, direction=direction, entry_dt=entry_dt, entry_price=entry_price,
            stop=stop, target=target, exit_date=exit_date, exit_dt=exit_dt,
            exit_price=exit_price, exit_reason=exit_reason, pnl_points=pnl_points,
        ))
        i = j_date_idx + 1

    trades_df = pd.DataFrame(trades)
    if trades_df.empty:
        return trades_df
    trades_df["pnl_twd"] = trades_df["pnl_points"] * POINT_VALUE
    return trades_df.sort_values("entry_dt").reset_index(drop=True)
