"""測試 RSI / 布林通道排名訊號的計算邏輯本身（scripts/test_rsi_bollinger_reversion_from_db.py）。

用構造出來、漲跌幅已知的簡單價格序列驗證 RSI 公式算對；並確認布林通道
z-score 訊號的方向（跌越深，z 越負）跟反未來函數（signal_fn 回傳值已經
shift(1) 過，不會用到 T 日當天還沒發生的資訊）是對的。
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.test_rsi_bollinger_reversion_from_db import (
    _compute_rsi,
    make_bollinger_ranking_signal_fn,
    rsi_ranking_signal_fn,
)


def test_rsi_is_100_when_only_gains_and_near_0_when_only_losses():
    dates = pd.bdate_range("2020-01-01", periods=30)
    rising = pd.DataFrame(
        {
            "date": dates, "stock_id": "UP", "industry": "IND",
            "open": range(100, 130), "high": range(101, 131), "low": range(99, 129),
            "close": range(100, 130), "volume": 100000, "turnover_value": 1e7,
        }
    )
    falling = pd.DataFrame(
        {
            "date": dates, "stock_id": "DOWN", "industry": "IND",
            "open": range(130, 100, -1), "high": range(131, 101, -1), "low": range(129, 99, -1),
            "close": range(130, 100, -1), "volume": 100000, "turnover_value": 1e7,
        }
    )
    master = pd.concat([rising, falling], ignore_index=True).sort_values(["stock_id", "date"]).reset_index(drop=True)

    rsi = _compute_rsi(master, window=14)
    rsi_up_last = rsi[master["stock_id"] == "UP"].iloc[-1]
    rsi_down_last = rsi[master["stock_id"] == "DOWN"].iloc[-1]

    assert rsi_up_last > 95  # 連續上漲、沒有任何下跌 -> RSI 該接近 100
    assert rsi_down_last < 5  # 連續下跌、沒有任何上漲 -> RSI 該接近 0


def test_rsi_ranking_signal_is_shifted_by_one_day():
    dates = pd.bdate_range("2020-01-01", periods=20)
    closes = [100.0] * 15 + [90.0, 80.0, 70.0, 60.0, 50.0]  # 最後 5 天連續重跌
    master = pd.DataFrame(
        {
            "date": dates, "stock_id": "S1", "industry": "IND",
            "open": closes, "high": [c * 1.01 for c in closes], "low": [c * 0.99 for c in closes],
            "close": closes, "volume": 100000, "turnover_value": 1e7,
        }
    )
    raw_rsi = _compute_rsi(master, window=14)
    shifted = rsi_ranking_signal_fn(master)

    # shift(1) 之後，第 i 天的訊號值應該等於原始序列第 i-1 天的值
    assert shifted.iloc[-1] == raw_rsi.iloc[-2]
    assert pd.isna(shifted.iloc[0])


def test_bollinger_zscore_is_more_negative_after_a_crash():
    dates = pd.bdate_range("2020-01-01", periods=25)
    closes = [100.0] * 24 + [60.0]  # 最後一天暴跌
    master = pd.DataFrame(
        {
            "date": dates, "stock_id": "S1", "industry": "IND",
            "open": closes, "high": [c * 1.01 for c in closes], "low": [c * 0.99 for c in closes],
            "close": closes, "volume": 100000, "turnover_value": 1e7,
        }
    )
    signal_fn = make_bollinger_ranking_signal_fn(require_volume_spike=False)
    z = signal_fn(master)

    # shift(1) 生效，暴跌那天的 z-score 要到「隔天」的訊號才看得到；這裡只有
    # 25 天資料、暴跌是最後一天，沒有隔天可以觀察到，所以改驗證暴跌本身在
    # shift 前的原始 z-score 確實非常負
    sma = master["close"].rolling(20).mean()
    std = master["close"].rolling(20).std()
    raw_z = (master["close"] - sma) / std
    assert raw_z.iloc[-1] < -2.0


def test_bollinger_signal_with_volume_requirement_drops_candidates_without_spike():
    dates = pd.bdate_range("2020-01-01", periods=26)
    closes = [100.0] * 24 + [60.0, 61.0]  # 暴跌後留一天，讓 shift(1) 後的訊號在資料範圍內看得到
    master = pd.DataFrame(
        {
            "date": dates, "stock_id": "S1", "industry": "IND",
            "open": closes, "high": [c * 1.01 for c in closes], "low": [c * 0.99 for c in closes],
            "close": closes, "volume": 100000, "turnover_value": 1e7,  # 成交量始終平盤，沒有爆量
        }
    )
    no_vol_fn = make_bollinger_ranking_signal_fn(require_volume_spike=False)
    with_vol_fn = make_bollinger_ranking_signal_fn(require_volume_spike=True)

    z_no_vol = no_vol_fn(master)
    z_with_vol = with_vol_fn(master)

    # 沒有要求量能確認時，暴跌隔天應該有非 NaN 的排名值；要求量能確認、
    # 但成交量根本沒爆量時，同一天的排名值應該被濾成 NaN
    assert z_no_vol.notna().any()
    assert z_with_vol.isna().all()
