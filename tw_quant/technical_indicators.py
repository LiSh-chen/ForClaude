"""給日內策略當進場過濾器用的日頻技術指標。

全部指標都用 `.shift(1)`：今天的過濾器只能看「昨天收盤後就已知」的值，避免用
到當天還沒發生的資訊。指標本身用日盤（08:45-13:45）聚合出的日K計算，不含
夜盤資料。
"""

from __future__ import annotations

from datetime import time

import numpy as np
import pandas as pd


def build_daily_bars(df: pd.DataFrame) -> pd.DataFrame:
    t = df["datetime"].dt.time
    day_df = df[(t >= time(8, 45)) & (t <= time(13, 45))].copy()
    day_df["date"] = day_df["datetime"].dt.date

    daily = day_df.groupby("date").agg(
        open=("open", "first"), high=("high", "max"), low=("low", "min"),
        close=("close", "last"), volume=("volume", "sum"),
    ).reset_index()
    daily["date"] = pd.to_datetime(daily["date"])
    return daily.sort_values("date").reset_index(drop=True)


def add_indicators(daily: pd.DataFrame) -> pd.DataFrame:
    d = daily.copy()

    delta = d["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / 14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / 14, adjust=False).mean()
    d["rsi14"] = 100 - 100 / (1 + avg_gain / avg_loss)

    ema12 = d["close"].ewm(span=12, adjust=False).mean()
    ema26 = d["close"].ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    d["macd_hist"] = macd - macd.ewm(span=9, adjust=False).mean()

    sma20 = d["close"].rolling(20).mean()
    std20 = d["close"].rolling(20).std()
    d["bb_pctb"] = (d["close"] - (sma20 - 2 * std20)) / (4 * std20)

    d["vol_ratio"] = d["volume"] / d["volume"].rolling(20).mean()

    # Stochastic KD(9,3,3)
    low9 = d["low"].rolling(9).min()
    high9 = d["high"].rolling(9).max()
    rsv = (d["close"] - low9) / (high9 - low9) * 100
    d["stoch_k"] = rsv.ewm(alpha=1 / 3, adjust=False).mean()
    d["stoch_d"] = d["stoch_k"].ewm(alpha=1 / 3, adjust=False).mean()

    # Williams %R(14)
    low14 = d["low"].rolling(14).min()
    high14 = d["high"].rolling(14).max()
    d["willr14"] = (high14 - d["close"]) / (high14 - low14) * -100

    # ATR(14)：真實區間的 14 日平均，衡量波動度水準（不是方向）
    prev_close = d["close"].shift(1)
    tr = pd.concat([
        d["high"] - d["low"],
        (d["high"] - prev_close).abs(),
        (d["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    d["atr14"] = tr.ewm(alpha=1 / 14, adjust=False).mean()
    d["atr_ratio"] = d["atr14"] / d["atr14"].rolling(60).mean()  # 相對近期波動水準

    # CCI(20)
    tp = (d["high"] + d["low"] + d["close"]) / 3
    tp_sma = tp.rolling(20).mean()
    tp_mad = tp.rolling(20).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    d["cci20"] = (tp - tp_sma) / (0.015 * tp_mad)

    for col in ["rsi14", "macd_hist", "bb_pctb", "vol_ratio", "stoch_k", "willr14", "atr_ratio", "cci20"]:
        d[f"{col}_lag1"] = d[col].shift(1)

    return d


LAG1_COLUMNS = [
    "rsi14_lag1", "macd_hist_lag1", "bb_pctb_lag1", "vol_ratio_lag1",
    "stoch_k_lag1", "willr14_lag1", "atr_ratio_lag1", "cci20_lag1",
]


def daily_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """回傳 date + `LAG1_COLUMNS` 過濾器欄位（date 是 `datetime.date`，方便
    直接跟 trades DataFrame 的 `trading_date` 合併）。"""
    d = add_indicators(build_daily_bars(df))
    d["date"] = d["date"].dt.date
    return d[["date"] + LAG1_COLUMNS]


def filter_trades_by_volume(trades: pd.DataFrame, df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """只留下「昨日成交量比率 >= threshold」那些交易日的交易。`threshold` 應該
    是用樣本內資料算出來、外部傳進來的固定值，這個函式本身不重新計算分位數
    ——避免在樣本外資料上重算門檻，變相用到未來資訊。"""
    return filter_trades_by_indicator(trades, df, "vol_ratio_lag1", threshold, "ge")


def filter_trades_by_indicator(
    trades: pd.DataFrame, df: pd.DataFrame, column: str, threshold: float, direction: str = "ge",
) -> pd.DataFrame:
    """通用版：只留下 `column`（`LAG1_COLUMNS` 之一）相對 `threshold` 方向
    （'ge'=大於等於／'le'=小於等於）成立的交易日。`threshold` 一律是外部傳入
    的固定值（通常用樣本內資料算出），這個函式本身不重新計算分位數。"""
    assert direction in ("ge", "le")
    indicators = daily_indicators(df)
    merged = trades.merge(indicators[["date", column]], left_on="trading_date", right_on="date", how="left")
    mask = merged[column] >= threshold if direction == "ge" else merged[column] <= threshold
    return merged[mask].drop(columns=["date", column])
