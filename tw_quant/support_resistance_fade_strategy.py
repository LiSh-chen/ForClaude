"""壓力支撐區間逆勢操作（日線、多空雙向）：在「盤整區間」的下緣支撐找反彈
做多、上緣壓力找拉回放空，不是趨勢突破（跟 donchian_breakout_strategy.py
方向相反：那是「價格突破區間就順勢跟」，這裡是「價格碰到區間邊界就反著
做，賭它彈回區間內」），也跟 rsi2_mean_reversion_strategy.py 不同源（那個
用 RSI 動能指標定義超買超賣，這裡直接用價格區間的實際高低點定義支撐壓力，
是價格行為/型態分析，不是指標分析）。

核心設計是這次特別針對「滑價不可控」這個顧慮：進場跟停利都用「限價單」
概念——支撐/壓力本身就是我們設定的限價，價格自己走到那個價位才會成交，
不會因為滑價讓成交價變差（如果當天開盤已經跳空穿越水準，用開盤價成交，
反而是價格改善，不是變差）；只有「跌破支撐/漲破壓力後停損出場」跟「持有
超過上限天數被迫出場」這兩種情境是用市價/停損單概念，才會真的吃到滑價。
也就是說：**這個策略設計上，滑價只發生在虧損出場的那些交易，獲利出場
（碰到停利目標）完全不受滑價影響**——這是限價單天生的不對稱優勢，用來
直接回應「滑價不可控」這個顧慮，不是靠更大的點數邊際去稀釋滑價占比
（唐奇安突破、RSI均值回歸走的是後面這條路，這裡走的是前面這條路）。

規則：
- 用過去 channel_window 天（不含當天）的最高/最低點當作壓力/支撐（跟唐奇安
  用同一個資料結構，語意相反）。
- 只在「盤整格局」進場：(a) 通道寬度相對收盤價要夠寬（min_range_pct），
  太窄的區間碰到邊界的意義不大；(b) trend_ma_window 日均線在最近
  trend_lookback 天的斜率不能太陡（trend_slope_threshold_pct），太陡代表
  處於單邊趨勢，逆勢摸頂摸底風險過高，直接跳過不進場（經典交易常識：
  不要在強烈趨勢中逆勢對做）。
- 進場：當天最低點 <= 支撐 -> 限價買在支撐價（如果開盤已經跳空低於支撐，
  用開盤價成交，是價格改善）；當天最高點 >= 壓力 -> 限價放空在壓力價
  （鏡射）。
- 停利：目標訂在進場當下的「對面邊界」（支撐進場的多單，目標是當時的
  壓力價；反之亦然），限價出場，一樣是價格到了才成交、跳空穿越算價格
  改善。
- 停損：跌破支撐 stop_atr_mult*ATR（多單）/ 漲破壓力 stop_atr_mult*ATR
  （空單），停損單概念，用 donchian_breakout_strategy._fill_price 同一套
  「跳空用開盤價、沒跳空用水準+滑價」邏輯。
- 同一天如果停損跟停利理論上都可能觸發（當天高低點同時觸及兩個水準），
  保守假設先觸發停損（不能從日K反推當天哪個先發生，用悲觀假設避免高估
  績效）。
- max_hold_days 到期還沒出場，用收盤價出場（市價單概念，一樣有滑價）。

簡化與已知限制：
- 只用日盤日K，跟這次會話其餘日線策略一致的簡化。
- 支撐/壓力只用「N日高低點」定義，沒有做更細緻的「多次測試才算有效」
  分群（實務上支撐壓力通常要被測試過才算數，這裡簡化成單純的N日極值），
  這是已知的簡化，如果這個方向測出真實邊際，可以再加強水準的定義方式。
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from tw_quant.donchian_breakout_strategy import _atr, _fill_price

POINT_VALUE = 50.0


@dataclass
class SupportResistanceFadeConfig:
    channel_window: int = 20
    min_range_pct: float = 3.0
    trend_ma_window: int = 60
    trend_lookback: int = 20
    trend_slope_threshold_pct: float = 3.0
    stop_atr_mult: float = 1.0
    atr_window: int = 14
    max_hold_days: int = 20
    slippage_points: float = 1.0


def compute_indicators(daily: pd.DataFrame, cfg: SupportResistanceFadeConfig) -> pd.DataFrame:
    d = daily.sort_values("date").reset_index(drop=True).copy()
    d["atr"] = _atr(d, cfg.atr_window)
    d["resistance"] = d["high"].shift(1).rolling(cfg.channel_window).max()
    d["support"] = d["low"].shift(1).rolling(cfg.channel_window).min()
    d["range_pct"] = (d["resistance"] - d["support"]) / d["close"].shift(1) * 100
    # 跟支撐/壓力一樣不含當天：用「昨天為止」判斷是不是處於強烈趨勢格局，
    # 不然反轉當天自己的急跌/急漲會拉低當天算出來的斜率，反而削弱濾網在
    # 最需要擋單的反轉當天的效果。
    trend_ma = d["close"].shift(1).rolling(cfg.trend_ma_window).mean()
    d["trend_ma"] = trend_ma
    d["trend_slope_pct"] = trend_ma.pct_change(cfg.trend_lookback) * 100
    return d


def _limit_fill(level: float, day_open: float, side: str) -> float:
    """限價單成交邏輯：side='buy_limit' 買進限價、'sell_limit' 賣出限價。
    如果開盤就已經比限價更好（買進時開盤更低、賣出時開盤更高），用開盤價
    成交（價格改善，不是滑價）；否則用限價本身成交（完全沒有滑價）。"""
    if side == "buy_limit":
        return day_open if day_open < level else level
    return day_open if day_open > level else level


def backtest(df: pd.DataFrame, cfg: SupportResistanceFadeConfig | None = None) -> pd.DataFrame:
    from tw_quant.technical_indicators import build_daily_bars

    cfg = cfg or SupportResistanceFadeConfig()
    daily = compute_indicators(build_daily_bars(df), cfg)
    n = len(daily)

    trades = []
    i = 0
    while i < n:
        row = daily.iloc[i]
        if pd.isna(row["support"]) or pd.isna(row["trend_slope_pct"]) or pd.isna(row["atr"]):
            i += 1
            continue

        if row["range_pct"] < cfg.min_range_pct or abs(row["trend_slope_pct"]) > cfg.trend_slope_threshold_pct:
            i += 1
            continue  # 區間太窄或處於單邊趨勢，不逆勢進場

        direction = None
        if row["low"] <= row["support"]:
            direction = "long"
            entry_price = _limit_fill(row["support"], row["open"], "buy_limit")
            target = row["resistance"]
            stop = entry_price - cfg.stop_atr_mult * row["atr"]
        elif row["high"] >= row["resistance"]:
            direction = "short"
            entry_price = _limit_fill(row["resistance"], row["open"], "sell_limit")
            target = row["support"]
            stop = entry_price + cfg.stop_atr_mult * row["atr"]

        if direction is None:
            i += 1
            continue

        entry_date = row["date"]
        exit_price = exit_date = exit_reason = None
        j_limit = min(i + cfg.max_hold_days, n - 1)
        j = i + 1
        while j <= j_limit:
            r = daily.iloc[j]
            if direction == "long":
                stop_hit = r["low"] <= stop
                target_hit = r["high"] >= target
                if stop_hit:  # 同一天兩者都可能觸及時，保守假設停損先發生
                    exit_price, exit_reason = _fill_price(stop, r["open"], r["high"], r["low"], "sell_stop", cfg.slippage_points)
                    exit_date = r["date"]
                    break
                if target_hit:
                    exit_price = _limit_fill(target, r["open"], "sell_limit")
                    exit_reason = "target"
                    exit_date = r["date"]
                    break
            else:
                stop_hit = r["high"] >= stop
                target_hit = r["low"] <= target
                if stop_hit:
                    exit_price, exit_reason = _fill_price(stop, r["open"], r["high"], r["low"], "buy_stop", cfg.slippage_points)
                    exit_date = r["date"]
                    break
                if target_hit:
                    exit_price = _limit_fill(target, r["open"], "buy_limit")
                    exit_reason = "target"
                    exit_date = r["date"]
                    break
            j += 1

        if exit_price is None:
            last = daily.iloc[j_limit]
            if direction == "long":
                exit_price = last["close"] - cfg.slippage_points
            else:
                exit_price = last["close"] + cfg.slippage_points
            exit_date, exit_reason = last["date"], "max_hold"
            exit_idx = j_limit
        else:
            exit_idx = int(daily.index[daily["date"] == exit_date][0])

        pnl_points = (exit_price - entry_price) if direction == "long" else (entry_price - exit_price)
        trades.append(dict(
            direction=direction, entry_date=entry_date, entry_price=entry_price,
            exit_date=exit_date, exit_price=exit_price, exit_reason=exit_reason, pnl_points=pnl_points,
        ))
        i = exit_idx + 1

    trades_df = pd.DataFrame(trades)
    if trades_df.empty:
        return trades_df
    trades_df["pnl_twd"] = trades_df["pnl_points"] * POINT_VALUE
    return trades_df.sort_values("entry_date").reset_index(drop=True)
