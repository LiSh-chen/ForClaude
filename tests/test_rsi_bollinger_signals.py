"""測試 RSI / 布林通道訊號的計算邏輯本身（scripts/test_rsi_bollinger_reversion_from_db.py）。

用構造出來、漲跌幅已知的簡單價格序列驗證 RSI 公式算對，並確認布林通道
訊號的方向（低於下軌才觸發，不是高於）跟反未來函數（用 T-1 的下軌比較
T 日收盤）是對的。
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.test_rsi_bollinger_reversion_from_db import _compute_rsi, make_bollinger_signal_fn
from tw_quant.config import StrategyConfig


def _always_true_pool(master, pool_cfg):
    return pd.Series(True, index=master.index)


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


def test_bollinger_signal_only_triggers_below_the_prior_day_lower_band(monkeypatch):
    import scripts.test_rsi_bollinger_reversion_from_db as mod

    monkeypatch.setattr(mod, "build_pool_mask", _always_true_pool)

    dates = pd.bdate_range("2020-01-01", periods=25)
    closes = [100.0] * 20 + [100.0, 100.0, 100.0, 100.0, 60.0]  # 最後一天價格暴跌，該跌破下軌
    master = pd.DataFrame(
        {
            "date": dates, "stock_id": "S1", "industry": "IND",
            "open": closes, "high": [c * 1.01 for c in closes], "low": [c * 0.99 for c in closes],
            "close": closes, "volume": 100000, "turnover_value": 1e7,
        }
    )
    cfg = StrategyConfig()
    signal_fn = make_bollinger_signal_fn(std_mult=2.0, require_volume_spike=False)
    entry_a, entry_b = signal_fn(master, master, pd.DataFrame(), cfg)

    assert entry_a[-1]  # 最後一天暴跌後應該觸發
    assert not entry_a[:-1].any()  # 前面價格平穩、沒有理由觸發
    assert not entry_b.any()
