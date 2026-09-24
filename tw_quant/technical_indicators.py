"""給日內策略當進場過濾器用的日頻技術指標。

全部指標都用 `.shift(1)`：今天的過濾器只能看「昨天收盤後就已知」的值，避免用
到當天還沒發生的資訊。指標本身用日盤（08:45-13:45）聚合出的日K計算，不含
夜盤資料。
"""

from __future__ import annotations

from datetime import time

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

    for col in ["rsi14", "macd_hist", "bb_pctb", "vol_ratio"]:
        d[f"{col}_lag1"] = d[col].shift(1)

    return d


def daily_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """回傳 date + 四個 *_lag1 過濾器欄位（date 是 `datetime.date`，方便直接
    跟 trades DataFrame 的 `trading_date` 合併）。"""
    d = add_indicators(build_daily_bars(df))
    d["date"] = d["date"].dt.date
    return d[["date", "rsi14_lag1", "macd_hist_lag1", "bb_pctb_lag1", "vol_ratio_lag1"]]


def filter_trades_by_volume(trades: pd.DataFrame, df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """只留下「昨日成交量比率 >= threshold」那些交易日的交易。`threshold` 應該
    是用樣本內資料算出來、外部傳進來的固定值，這個函式本身不重新計算分位數
    ——避免在樣本外資料上重算門檻，變相用到未來資訊。"""
    indicators = daily_indicators(df)
    merged = trades.merge(indicators[["date", "vol_ratio_lag1"]], left_on="trading_date", right_on="date", how="left")
    return merged[merged["vol_ratio_lag1"] >= threshold].drop(columns=["date", "vol_ratio_lag1"])
