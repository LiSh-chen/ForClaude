"""可調參數版的訊號函式。

原本的研究腳本（scripts/test_*_from_db.py）把「已經找到的最佳參數」寫死成
模組層級常數（例如 BEST_DT_WIN = 20），因為那些腳本的目的是「重現/驗證
某一組已知結果」。這裡要做的事情相反：讓使用者在網頁上自由調整同一批
參數，所以把訊號邏輯改寫成「接受參數、回傳訊號函式」的 factory，數學算式
跟原本的腳本逐行對照完全一樣，只是把寫死的常數換成參數。

PEAD 相關的營收「意外」定義（次月幾號才算公開已知、用近幾個月平均當基準）
刻意不開放調整，直接重用 scripts/test_pead_revenue_drift_from_db.py 的
既有實作——那組數字是反未來函數安全邊界的假設，不是單純的績效調參數，
隨便放寬有可能不小心引入未來函數，不適合在網頁上讓人隨手改。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tw_quant import indicators as ind
from tw_quant.signals import build_pool_mask


def make_mss_signal_fn(downtrend_window: int, swing_window: int, volume_mult: float, volume_window: int = 20):
    """結構轉折 C：跌破前波低點後、又收復前一段 swing high，且帶量確認。
    逐行對照 scripts/test_holiday_overlay_from_db.py 的 mss_signal_fn。
    """

    def _fn(master: pd.DataFrame, prices: pd.DataFrame, margin_short: pd.DataFrame, cfg) -> tuple:
        pool = build_pool_mask(master, cfg.pool)
        low_min_now = ind.rolling_min(master, "low", downtrend_window)
        low_min_earlier = ind.shift_by_group(low_min_now, master, periods=downtrend_window)
        low_min_now_prior = ind.shift_by_group(low_min_now, master, periods=1)
        low_min_earlier_prior = ind.shift_by_group(low_min_earlier, master, periods=1)
        downtrend_context = (low_min_now_prior < low_min_earlier_prior).fillna(False)

        swing_high = ind.rolling_max(master, "high", swing_window)
        swing_high_prior = ind.shift_by_group(swing_high, master, periods=1)
        breaks_structure = (master["close"] > swing_high_prior).fillna(False)

        avg_vol = ind.sma(master, "volume", volume_window)
        avg_vol_prior = ind.shift_by_group(avg_vol, master, periods=1)
        volume_ok = (master["volume"] > avg_vol_prior * volume_mult).fillna(False)

        entry = pool & downtrend_context & breaks_structure & volume_ok
        return entry.values, np.zeros(len(master), dtype=bool)

    return _fn


def _compute_rsi(master: pd.DataFrame, window: int) -> pd.Series:
    """Wilder's RSI，逐行對照 scripts/test_rsi_bollinger_reversion_from_db.py。"""
    delta = master.groupby("stock_id", sort=False)["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.groupby(master["stock_id"], sort=False).transform(
        lambda s: s.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    )
    avg_loss = loss.groupby(master["stock_id"], sort=False).transform(
        lambda s: s.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    )
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - 100 / (1 + rs)

    both_zero = (avg_gain == 0) & (avg_loss == 0)
    only_loss_zero = (avg_loss == 0) & ~both_zero
    rsi = rsi.mask(only_loss_zero, 100.0)
    rsi = rsi.mask(both_zero, 50.0)
    return rsi.fillna(50.0)


def make_rsi_signal_fn(window: int):
    def _fn(master: pd.DataFrame) -> pd.Series:
        rsi = _compute_rsi(master, window)
        return ind.shift_by_group(rsi, master, periods=1)

    return _fn


def make_bollinger_signal_fn(window: int, require_volume_spike: bool, volume_mult: float):
    def _fn(master: pd.DataFrame) -> pd.Series:
        sma = ind.sma(master, "close", window)
        std = ind.rolling_std(master, "close", window)
        z = (master["close"] - sma) / std.replace(0, np.nan)
        if require_volume_spike:
            avg_vol = ind.sma(master, "volume", window)
            volume_ok = master["volume"] > avg_vol * volume_mult
            z = z.where(volume_ok.fillna(False), np.nan)
        return ind.shift_by_group(z, master, periods=1)

    return _fn


def make_breakout_combo_signal_fn(
    breakout_window: int, volume_mult: float, revenue_flag_lookup: pd.Series
):
    """營收動能+價量突破，逐行對照 scripts/test_pead_breakout_combo_from_db.py
    的 make_combo_signal_fn。revenue_flag_lookup 由呼叫端先用固定的反未來
    函數安全邏輯（attach_revenue_momentum_flag）算好再傳進來。
    """

    def _fn(master: pd.DataFrame, prices: pd.DataFrame, margin_short: pd.DataFrame, cfg) -> tuple:
        pool = build_pool_mask(master, cfg.pool)

        idx = pd.MultiIndex.from_arrays([master["stock_id"], master["date"]])
        revenue_ok = pd.Series(revenue_flag_lookup.reindex(idx).fillna(False).values, index=master.index)

        rolling_high = ind.rolling_max(master, "high", breakout_window)
        rolling_high_prior = ind.shift_by_group(rolling_high, master, periods=1)
        price_breakout = (master["close"] > rolling_high_prior).fillna(False)

        avg_vol_5 = ind.sma(master, "volume", 5)
        avg_vol_5_prior = ind.shift_by_group(avg_vol_5, master, periods=1)
        volume_spike = (master["volume"] > avg_vol_5_prior * volume_mult).fillna(False)

        entry = pool & revenue_ok & price_breakout & volume_spike
        return entry.values, np.zeros(len(master), dtype=bool)

    return _fn
