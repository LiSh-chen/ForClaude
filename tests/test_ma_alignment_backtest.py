"""測試 tw_quant/ma_alignment_backtest.py：大週期 MA 多頭/空頭排列 + 小週期
反向排列（拉回/反彈）進場，固定風報比停損停利。

用構造出的均線排列狀態陣列（跳過大小週期重取樣本身，直接指定 large_state/
small_state）測 simulate() 的訊號、風控、邊緣觸發邏輯；另外用小段真實 K棒
測 build_timeframe_state 的排列判定跟無未來函數。
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tw_quant.ma_alignment_backtest import (  # noqa: E402
    _ma_alignment_state, build_base_arrays, build_timeframe_state, simulate,
)

FLAT = (17000.0, 17000.0, 17000.0, 17000.0)


def _bars(specs: list[tuple[float, float, float, float]], start: str = "2021-06-01 21:00:00") -> pd.DataFrame:
    times = pd.date_range(start, periods=len(specs), freq="1min")
    rows = [{"datetime": ts, "open": o, "high": h, "low": l, "close": c} for ts, (o, h, l, c) in zip(times, specs)]
    return pd.DataFrame(rows)


def test_long_entry_when_large_bull_small_bear_take_profit():
    specs = [FLAT] * 30
    # 塑造 swing low 在 index 20 附近，讓 roll_low(idx=25) 抓到明確低點
    specs[20] = (17000, 17001, 16980, 16999)
    for i in range(21, 25):
        specs[i] = FLAT
    df = _bars(specs)
    arrays = build_base_arrays(df, swing_lookback=10)

    n = len(specs)
    large_state = np.zeros(n, dtype=int)
    small_state = np.zeros(n, dtype=int)
    large_state[25] = 1   # 大週期多頭排列
    small_state[25] = -1  # 小週期空頭排列（拉回）-> 應該在 index25 觸發做多訊號

    # index26 是進場根（用它的開盤價進場），index27 才開始掃停損停利
    df.loc[26, ["open", "high", "low", "close"]] = [17010, 17015, 17005, 17012]
    df.loc[27, ["open", "high", "low", "close"]] = [17015, 17200, 17010, 17190]  # 大漲觸及停利

    arrays = build_base_arrays(df, swing_lookback=10)
    trades = simulate(arrays, large_state, small_state, r_multiple=2.0, max_hold_minutes=10)

    assert len(trades) == 1
    t = trades.iloc[0]
    assert t["direction"] == "long"
    assert t["entry_idx"] == 26
    assert t["sl"] == t["entry_price"] - t["risk_points"]
    assert t["exit_reason"] == "take_profit"


def test_short_entry_when_large_bear_small_bull():
    specs = [FLAT] * 30
    specs[20] = (17000, 17020, 16999, 17001)  # swing high 明確
    df = _bars(specs)

    n = len(specs)
    large_state = np.zeros(n, dtype=int)
    small_state = np.zeros(n, dtype=int)
    large_state[25] = -1  # 大週期空頭排列
    small_state[25] = 1   # 小週期多頭排列（反彈）-> 應該觸發做空訊號

    df.loc[26, ["open", "high", "low", "close"]] = [17000, 17005, 16995, 17002]
    df.loc[27, ["open", "high", "low", "close"]] = [17000, 17005, 16800, 16810]  # 大跌觸及停利

    arrays = build_base_arrays(df, swing_lookback=10)
    trades = simulate(arrays, large_state, small_state, r_multiple=2.0, max_hold_minutes=10)

    assert len(trades) == 1
    t = trades.iloc[0]
    assert t["direction"] == "short"
    assert t["sl"] == t["entry_price"] + t["risk_points"]
    assert t["exit_reason"] == "take_profit"


def test_no_signal_when_states_agree():
    specs = [FLAT] * 15
    df = _bars(specs)
    arrays = build_base_arrays(df, swing_lookback=5)
    n = len(specs)
    large_state = np.full(n, 1)
    small_state = np.full(n, 1)  # 大小週期同向，不該有訊號（策略要的是反向拉回）
    trades = simulate(arrays, large_state, small_state, r_multiple=2.0)
    assert trades.empty


def test_edge_trigger_only_fires_once_per_episode():
    specs = [FLAT] * 15
    df = _bars(specs)
    arrays = build_base_arrays(df, swing_lookback=5)
    n = len(specs)
    large_state = np.zeros(n, dtype=int)
    small_state = np.zeros(n, dtype=int)
    # 條件從 index 6 開始連續成立到 index 10（同一段對齊期間）
    large_state[6:11] = 1
    small_state[6:11] = -1
    trades = simulate(arrays, large_state, small_state, r_multiple=2.0, max_hold_minutes=2)
    # 不管中間補幾筆，訊號應該只在 index6 的邊緣觸發那一次（後續被 blocked_until 或
    # 邊緣觸發條件本身擋掉，不會每根都重複進場）
    assert (trades["signal_idx"] == 6).sum() == 1
    assert (trades["signal_idx"].isin([7, 8, 9, 10])).sum() == 0


def test_build_timeframe_state_alignment_and_no_lookahead():
    # 5 分鐘週期、fast=2/mid=3/slow=4，手算驗證排列判定跟無未來函數
    times = pd.date_range("2021-06-01 00:00:00", periods=25, freq="1min")
    # 讓價格持續上漲，5 分鐘收盤序列會是遞增的，均線排列最終應呈現多頭排列
    closes = [100 + i * 0.5 for i in range(25)]
    df = pd.DataFrame({"datetime": times, "open": closes, "high": closes, "low": closes, "close": closes})

    state = build_timeframe_state(df, freq="5min", fast=2, mid=3, slow=4)
    # 前段（還不夠 4 根 5 分鐘K棒算慢均線）必為中性 0
    assert (state[:20] == 0).all()
    # 上漲序列足夠久之後應該呈現多頭排列（fast>mid>slow）
    assert state[-1] == 1


def test_ma_alignment_state_bearish():
    close = pd.Series([10, 9, 8, 7, 6, 5, 4, 3, 2, 1], dtype=float)
    state = _ma_alignment_state(close, fast=2, mid=3, slow=4)
    assert state.iloc[-1] == -1
