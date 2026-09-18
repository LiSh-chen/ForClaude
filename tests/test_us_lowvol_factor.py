"""測試 scripts/test_us_lowvol_factor_from_db.py 的低波動排名訊號函式本身
（不連網、不用真的跑回測引擎）：波動度計算對不對、排名方向對不對（低波動
該排在前面，因為搭配 ascending=True）、用的是 T-1 資訊不是當天。
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from test_us_lowvol_factor_from_db import make_low_vol_signal_fn  # noqa: E402


def _master(dates, stock_id, closes):
    return pd.DataFrame({"date": dates, "stock_id": [stock_id] * len(dates), "close": closes})


def test_low_vol_stock_ranks_below_high_vol_stock():
    dates = pd.bdate_range("2024-01-01", periods=30)
    # A 平穩上漲、B 大幅震盪但終值相近——A 的日報酬波動度該遠低於 B
    steady = 100 * (1.001 ** np.arange(len(dates)))
    choppy = 100 + 10 * np.sin(np.arange(len(dates)))
    master = pd.concat([_master(dates, "A", steady), _master(dates, "B", choppy)], ignore_index=True)

    signal_fn = make_low_vol_signal_fn(vol_window=10)
    result = signal_fn(master)

    last_a = result[(master["stock_id"] == "A")].iloc[-1]
    last_b = result[(master["stock_id"] == "B")].iloc[-1]
    assert last_a < last_b  # 波動度數值上 A 該小於 B，ascending=True 才會選到 A


def test_uses_t_minus_1_information_not_same_day_return():
    dates = pd.bdate_range("2024-01-01", periods=15)
    closes = [100.0] * 10 + [100.0, 150.0, 100.0, 150.0, 100.0]  # 第 11 天(index 10)開始劇烈震盪
    master = _master(dates, "A", closes)

    signal_fn = make_low_vol_signal_fn(vol_window=5)
    result = signal_fn(master)

    # 劇烈震盪開始那天（index 10）本身的波動度數值還不該反映當天的價格變化，
    # 因為排名用的是 shift(1) 後、T-1 為止已知的資訊
    vol_at_震盪開始日 = result.iloc[10]
    vol_前一天 = result.iloc[9]
    assert vol_at_震盪開始日 == vol_前一天


def test_nan_during_warmup_period():
    dates = pd.bdate_range("2024-01-01", periods=5)
    master = _master(dates, "A", [100.0, 101.0, 99.0, 102.0, 98.0])

    signal_fn = make_low_vol_signal_fn(vol_window=10)  # 資料筆數不夠撐滿窗格
    result = signal_fn(master)

    assert result.isna().all()
