"""通用技術指標，全部以「每檔股票獨立計算」為前提（多檔股票長格式 DataFrame）。

輸入 DataFrame 慣例：至少含 group_col（預設 'stock_id'）與對應欄位，
每個 stock_id 內部須已依日期由舊到新排序，否則 rolling 結果無意義。

這裡刻意不做 shift(1)：所有指標回傳「當日可得」的值，是否要遞延一天
交由 signals.py 依規格書「強制使用 shift(1)」的規則統一處理，避免兩處各自
shift 造成邏輯不一致或重複遞延。
"""

from __future__ import annotations

import pandas as pd


def _grouped_rolling(df: pd.DataFrame, col: str, window: int, group_col: str, func):
    grouped = df.groupby(group_col, sort=False)[col].rolling(window, min_periods=window)
    result = func(grouped)
    return result.reset_index(level=0, drop=True).reindex(df.index)


def sma(df: pd.DataFrame, col: str, window: int, group_col: str = "stock_id") -> pd.Series:
    return _grouped_rolling(df, col, window, group_col, lambda r: r.mean())


def rolling_max(df: pd.DataFrame, col: str, window: int, group_col: str = "stock_id") -> pd.Series:
    return _grouped_rolling(df, col, window, group_col, lambda r: r.max())


def rolling_min(df: pd.DataFrame, col: str, window: int, group_col: str = "stock_id") -> pd.Series:
    return _grouped_rolling(df, col, window, group_col, lambda r: r.min())


def rolling_std(df: pd.DataFrame, col: str, window: int, group_col: str = "stock_id") -> pd.Series:
    return _grouped_rolling(df, col, window, group_col, lambda r: r.std())


def rolling_percentile_rank(
    df: pd.DataFrame, col: str, window: int, group_col: str = "stock_id"
) -> pd.Series:
    """回傳 0-100 的百分位排名：視窗最後一筆值，在過去 window 期樣本中的百分位。

    PR10 意指「目前值低於過去分布的第 10 百分位」，對應此函式回傳值 <= 10。
    PR90 意指「目前值高於過去分布的第 90 百分位」，對應此函式回傳值 >= 90。
    """
    pr = _grouped_rolling(df, col, window, group_col, lambda r: r.rank(pct=True))
    return pr * 100.0


def true_range(df: pd.DataFrame, group_col: str = "stock_id") -> pd.Series:
    prev_close = df.groupby(group_col, sort=False)["close"].shift(1)
    hl = df["high"] - df["low"]
    hc = (df["high"] - prev_close).abs()
    lc = (df["low"] - prev_close).abs()
    return pd.concat([hl, hc, lc], axis=1).max(axis=1)


def atr(df: pd.DataFrame, window: int, group_col: str = "stock_id") -> pd.Series:
    """Wilder's ATR：真實區間 TR 做 EWM 平滑（alpha = 1/window）。"""
    tr = true_range(df, group_col)
    tmp = pd.DataFrame({group_col: df[group_col].values, "tr": tr.values}, index=df.index)
    return tmp.groupby(group_col, sort=False)["tr"].transform(
        lambda s: s.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    )


def range_swing_pct(df: pd.DataFrame, window: int, group_col: str = "stock_id") -> pd.Series:
    """近 window 日高低價震幅：(區間最高價 - 區間最低價) / 區間最低價。"""
    hh = rolling_max(df, "high", window, group_col)
    ll = rolling_min(df, "low", window, group_col)
    return (hh - ll) / ll


def avg_volume(df: pd.DataFrame, window: int, group_col: str = "stock_id") -> pd.Series:
    return sma(df, "volume", window, group_col)


def shift_by_group(series: pd.Series, df: pd.DataFrame, periods: int = 1, group_col: str = "stock_id") -> pd.Series:
    """依規格書「強制使用 shift(1) 遞延一天」對任一序列做分組位移，避免跨股票污染。"""
    tmp = pd.DataFrame({group_col: df[group_col].values, "v": series.values}, index=df.index)
    return tmp.groupby(group_col, sort=False)["v"].shift(periods)
