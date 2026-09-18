"""測試月營收動能 + 價量突破組合策略的訊號邏輯
（scripts/test_pead_breakout_combo_from_db.py）。
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.test_pead_breakout_combo_from_db import YOY_THRESHOLD, attach_revenue_momentum_flag, make_combo_signal_fn
from tw_quant.config import StrategyConfig


def _always_true_pool(master, pool_cfg):
    return pd.Series(True, index=master.index)


def _make_revenue(stock_id: str, monthly_values: dict[tuple[int, int], float]) -> pd.DataFrame:
    rows = [
        {"date": pd.Timestamp(year=y, month=m, day=1), "stock_id": stock_id, "revenue": v, "revenue_year": y, "revenue_month": m}
        for (y, m), v in monthly_values.items()
    ]
    return pd.DataFrame(rows)


def _declining_then_spike_monthly(spike_value: float) -> dict[tuple[int, int], float]:
    """前 13 個月平盤 100（用來滿足 12 個月暖身），接著 11 個月逐月遞減
    （每月都創「新低」而不是新高，YoY 也是負的，確保 momentum_ok 在這段
    期間確定是 False，避免「一路平盤 = 每個月都跟自己打平的新高」這種
    邊界情況把測試結果弄混），最後一個月跳升到 spike_value。
    """
    monthly = {(2021, m): 100.0 for m in range(1, 13)}
    monthly[(2022, 1)] = 100.0
    for i, m in enumerate(range(2, 13)):
        monthly[(2022, m)] = 100.0 - (i + 1) * 2.0  # 98, 96, ..., 78
    monthly[(2023, 1)] = spike_value
    return monthly


def test_revenue_momentum_flag_true_when_yoy_above_threshold():
    dates = pd.bdate_range("2023-01-01", "2023-06-30")
    prices = pd.DataFrame(
        {
            "date": dates, "stock_id": "S1", "industry": "IND",
            "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0,
            "volume": 100000, "turnover_value": 1e7,
        }
    )
    revenue = _make_revenue("S1", _declining_then_spike_monthly(120.0))  # yoy = 0.2 > 15% 門檻，known_date = 2023-02-15

    master = attach_revenue_momentum_flag(prices, revenue)
    known_date = pd.Timestamp("2023-02-15")

    before = master[(master["date"] > known_date - pd.Timedelta(days=10)) & (master["date"] < known_date)]
    after = master[master["date"] > known_date]

    assert not before["revenue_momentum_ok_shifted"].any()
    assert after["revenue_momentum_ok_shifted"].any()


def test_revenue_momentum_flag_false_when_yoy_below_threshold():
    dates = pd.bdate_range("2023-01-01", "2023-06-30")
    prices = pd.DataFrame(
        {
            "date": dates, "stock_id": "S1", "industry": "IND",
            "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0,
            "volume": 100000, "turnover_value": 1e7,
        }
    )
    # 95 比前 11 個月遞減區間裡的最高點（98）還低，所以不會被判定成「近12個月新高」；
    # yoy = 95/100-1 = -0.05，也遠低於 15% 門檻
    revenue = _make_revenue("S1", _declining_then_spike_monthly(95.0))

    master = attach_revenue_momentum_flag(prices, revenue)
    assert not master["revenue_momentum_ok_shifted"].any()


def test_combo_signal_requires_all_four_conditions(monkeypatch):
    import scripts.test_pead_breakout_combo_from_db as mod

    monkeypatch.setattr(mod, "build_pool_mask", _always_true_pool)

    dates = pd.bdate_range("2023-01-01", periods=30)
    closes = [100.0] * 25 + [101.0, 102.0, 103.0, 104.0, 130.0]  # 最後一天大漲突破
    volumes = [100000] * 25 + [100000, 100000, 100000, 100000, 500000]  # 最後一天爆量
    master = pd.DataFrame(
        {
            "date": dates, "stock_id": "S1", "industry": "IND",
            "open": closes, "high": [c * 1.01 for c in closes], "low": [c * 0.99 for c in closes],
            "close": closes, "volume": volumes, "turnover_value": 1e7,
        }
    )
    master["revenue_momentum_ok_shifted"] = False  # 全部關閉營收條件

    cfg = StrategyConfig()
    signal_fn = make_combo_signal_fn(breakout_window=20, volume_mult=2.0, master_with_revenue=master)
    entry_a, entry_b = signal_fn(master, master, pd.DataFrame(), cfg)

    # 就算價量突破條件都滿足，營收動能條件是 False，entry 應該全部是 False
    assert not entry_a.any()
    assert not entry_b.any()

    # 打開營收動能條件後，最後一天（真的突破+爆量）應該觸發。make_combo_signal_fn
    # 在呼叫當下就把 revenue_momentum_ok_shifted 這欄做成查表用的 Series，所以
    # 修改欄位值後要重新呼叫一次拿新的 signal_fn，才能反映新的欄位狀態。
    master["revenue_momentum_ok_shifted"] = True
    signal_fn2 = make_combo_signal_fn(breakout_window=20, volume_mult=2.0, master_with_revenue=master)
    entry_a2, _ = signal_fn2(master, master, pd.DataFrame(), cfg)
    assert entry_a2[-1]
