"""參、雙軌進場邏輯 (SOP) 與日曆防禦

反未來函數原則：任何「用來篩選的門檻/濾網」一律以 T-1（含）以前的資料計算，
在程式上以「先算出當日原始值，再對整個序列 shift(1)」實作，等價於規格書所寫
「T-1 日之 XXX」。只有「點火扣板機」明訂為 T 日收盤價/成交量本身，因此不 shift，
但它比較的基準（過去 60 日最高價、過去 5 日均量）一樣要 shift(1)，
否則會把「今天」算進「過去」的視窗裡，造成資料洩漏。

輸出：一個 DataFrame，index 對齊輸入 prices，欄位為每個條件的布林遮罩，
以及最終 entry_signal_a / entry_signal_b。訊號代表「T 日收盤後產生，
T+1 日開盤進場」，實際下單時機在 risk.py / backtest.py 處理。
"""

from __future__ import annotations

import pandas as pd

from tw_quant.config import PoolConfig, SqueezeFilterConfig, IgnitionConfig, StrategyBConfig
from tw_quant import indicators as ind
from tw_quant.data_provider import compute_margin_short_ratio


def _eligible_mask(prices: pd.DataFrame, min_history_days: int) -> pd.Series:
    bars_seen = prices.groupby("stock_id", sort=False).cumcount() + 1
    return bars_seen >= min_history_days


def build_pool_mask(prices: pd.DataFrame, cfg: PoolConfig) -> pd.Series:
    """基礎魚池：T-1 日 5 日均量 > 門檻（張）且收盤價站上 60 日均線，排除新股。"""
    avg_vol_shares = ind.sma(prices, "volume", 5)
    ma60 = ind.sma(prices, "close", cfg.trend_ma_window)

    raw_pool = (avg_vol_shares > cfg.min_avg_volume_lots * 1000) & (prices["close"] > ma60)
    eligible = _eligible_mask(prices, cfg.min_history_days)
    raw_pool = raw_pool & eligible

    pool_t_minus_1 = ind.shift_by_group(raw_pool.astype(float), prices, periods=1)
    return pool_t_minus_1.fillna(0).astype(bool)


def build_squeeze_mask(prices: pd.DataFrame, cfg: SqueezeFilterConfig) -> pd.Series:
    """壓縮濾網：T-1 日「近 40 日震幅」低於過去 252 日 PR{threshold}。"""
    swing = ind.range_swing_pct(prices, cfg.range_window)
    tmp = prices[["stock_id"]].copy()
    tmp["swing"] = swing
    pr = ind.rolling_percentile_rank(tmp, "swing", cfg.pr_lookback)
    raw = pr <= cfg.pr_threshold
    shifted = ind.shift_by_group(raw.astype(float), prices, periods=1)
    return shifted.fillna(0).astype(bool)


def build_ignition_mask(prices: pd.DataFrame, cfg: IgnitionConfig) -> pd.Series:
    """點火扣板機：T 日收盤價突破「T-1 為止」過去 60 日最高價，
    且 T 日成交量 > 「T-1 為止」過去 5 日均量的 N 倍。
    """
    rolling_high = ind.rolling_max(prices, "high", cfg.breakout_window)
    rolling_high_prior = ind.shift_by_group(rolling_high, prices, periods=1)

    avg_vol = ind.sma(prices, "volume", cfg.volume_avg_window)
    avg_vol_prior = ind.shift_by_group(avg_vol, prices, periods=1)

    breakout = prices["close"] > rolling_high_prior
    volume_spike = prices["volume"] > (avg_vol_prior * cfg.volume_multiplier)
    return (breakout & volume_spike).fillna(False)


def build_calendar_defense_mask(prices: pd.DataFrame, cfg: StrategyBConfig) -> pd.Series:
    """日曆防禦：排除每年 3-4 月（股東會）與 6-8 月（除權息）的進場訊號。"""
    month = prices["date"].dt.month
    excluded = month.isin(cfg.excluded_months)
    return ~excluded


def build_margin_short_mask(
    prices: pd.DataFrame, margin_short: pd.DataFrame, cfg: StrategyBConfig
) -> pd.Series:
    """異常濾網：T-1 日券資比 > 過去 252 日 PR{threshold}。"""
    ms = margin_short.sort_values(["stock_id", "date"]).reset_index(drop=True)
    ms = ms.copy()
    ms["ratio"] = compute_margin_short_ratio(ms)
    pr = ind.rolling_percentile_rank(ms, "ratio", cfg.margin_short_ratio_pr_lookback)
    raw = pr >= cfg.margin_short_ratio_pr_threshold
    shifted = ind.shift_by_group(raw.astype(float), ms, periods=1)
    ms["margin_short_signal"] = shifted.fillna(0).astype(bool)

    merged = prices[["date", "stock_id"]].merge(
        ms[["date", "stock_id", "margin_short_signal"]], on=["date", "stock_id"], how="left"
    )
    return merged["margin_short_signal"].fillna(False).reset_index(drop=True).set_axis(prices.index)


def generate_strategy_a_signals(
    prices: pd.DataFrame,
    pool_cfg: PoolConfig,
    squeeze_cfg: SqueezeFilterConfig,
    ignition_cfg: IgnitionConfig,
) -> pd.DataFrame:
    pool = build_pool_mask(prices, pool_cfg)
    squeeze = build_squeeze_mask(prices, squeeze_cfg)
    ignition = build_ignition_mask(prices, ignition_cfg)

    out = prices[["date", "stock_id"]].copy()
    out["pool"] = pool.values
    out["squeeze"] = squeeze.values
    out["ignition"] = ignition.values
    out["entry_signal"] = out["pool"] & out["squeeze"] & out["ignition"]
    out["strategy"] = "A"
    return out


def generate_strategy_b_signals(
    prices: pd.DataFrame,
    margin_short: pd.DataFrame,
    pool_cfg: PoolConfig,
    ignition_cfg: IgnitionConfig,
    strategy_b_cfg: StrategyBConfig,
) -> pd.DataFrame:
    pool = build_pool_mask(prices, pool_cfg)
    ignition = build_ignition_mask(prices, ignition_cfg)
    calendar_ok = build_calendar_defense_mask(prices, strategy_b_cfg)
    margin_short_ok = build_margin_short_mask(prices, margin_short, strategy_b_cfg)

    out = prices[["date", "stock_id"]].copy()
    out["pool"] = pool.values
    out["margin_short_anomaly"] = margin_short_ok.values
    out["calendar_ok"] = calendar_ok.values
    out["ignition"] = ignition.values
    out["entry_signal"] = out["pool"] & out["margin_short_anomaly"] & out["calendar_ok"] & out["ignition"]
    out["strategy"] = "B"
    return out
