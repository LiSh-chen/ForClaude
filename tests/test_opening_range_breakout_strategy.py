"""測試 tw_quant/opening_range_breakout_strategy.py：開盤區間突破（ORB）。"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tw_quant.opening_range_breakout_strategy import (  # noqa: E402
    OpeningRangeBreakoutConfig, backtest,
)


def _minute_bars(date: str, start_time: str, n: int, **kw) -> list[dict]:
    ts = pd.date_range(f"{date} {start_time}", periods=n, freq="min")
    return [dict(datetime=t, **kw) for t in ts]


def test_long_breakout_then_stop_loss_hit():
    rows = []
    rows += _minute_bars("2021-01-04", "08:45", 30, open=100, high=101, low=99, close=100, volume=100)
    rows.append(dict(datetime=pd.Timestamp("2021-01-04 09:15"), open=100.5, high=105, low=100, close=104, volume=100))
    rows.append(dict(datetime=pd.Timestamp("2021-01-04 09:16"), open=103, high=104, low=95, close=96, volume=100))
    rows += _minute_bars("2021-01-04", "09:17", 200, open=96, high=97, low=95, close=96, volume=100)
    df = pd.DataFrame(rows)

    cfg = OpeningRangeBreakoutConfig(min_range_points=1.0, slippage_points=1.0)
    trades = backtest(df, cfg)

    assert len(trades) == 1
    t = trades.iloc[0]
    assert t["direction"] == "long"
    assert t["entry_price"] == 102.0  # 區間上緣101 + 1滑價
    assert t["exit_price"] == 98.0    # 區間下緣99 - 1滑價
    assert t["exit_reason"] == "normal_slippage"
    assert t["pnl_points"] == -4.0


def test_short_breakout_gap_through_entry():
    rows = []
    rows += _minute_bars("2021-01-04", "08:45", 30, open=100, high=101, low=99, close=100, volume=100)
    # 突破棒直接跳空穿越下緣（開盤價就已經在水準之下）
    rows.append(dict(datetime=pd.Timestamp("2021-01-04 09:15"), open=95, high=96, low=94, close=94.5, volume=100))
    rows += _minute_bars("2021-01-04", "09:16", 200, open=94, high=95, low=93, close=94, volume=100)
    df = pd.DataFrame(rows)

    cfg = OpeningRangeBreakoutConfig(min_range_points=1.0, slippage_points=1.0)
    trades = backtest(df, cfg)

    assert len(trades) == 1
    t = trades.iloc[0]
    assert t["direction"] == "short"
    assert t["entry_price"] == 95.0  # 開盤已跳空穿越下緣99，用開盤價成交
    assert t["entry_reason"] == "gap_through"


def test_no_stop_hit_exits_at_session_close_with_slippage():
    rows = []
    rows += _minute_bars("2021-01-04", "08:45", 30, open=100, high=101, low=99, close=100, volume=100)
    rows.append(dict(datetime=pd.Timestamp("2021-01-04 09:15"), open=100.5, high=105, low=100, close=104, volume=100))
    rows += _minute_bars("2021-01-04", "09:16", 200, open=103, high=104, low=102, close=103, volume=100)  # 不碰停損99
    rows += _minute_bars("2021-01-04", "13:25", 20, open=106, high=107, low=105, close=106, volume=100)
    df = pd.DataFrame(rows)

    cfg = OpeningRangeBreakoutConfig(min_range_points=1.0, slippage_points=1.0)
    trades = backtest(df, cfg)

    assert len(trades) == 1
    t = trades.iloc[0]
    assert t["exit_reason"] == "session_close"
    assert t["exit_price"] == 105.0  # 13:25 開盤 106 - 1 滑價
    assert t["pnl_points"] == 3.0    # 105 - 102(進場)


def test_narrow_opening_range_is_skipped():
    # 開盤區間寬度只有 2 點，預設門檻 5 點，應該整天跳過不交易
    rows = _minute_bars("2021-01-04", "08:45", 300, open=100, high=101, low=99, close=100, volume=100)
    df = pd.DataFrame(rows)
    trades = backtest(df, OpeningRangeBreakoutConfig())
    assert trades.empty


def test_target_r_multiple_exits_at_fixed_target():
    rows = []
    rows += _minute_bars("2021-01-04", "08:45", 30, open=100, high=101, low=99, close=100, volume=100)
    rows.append(dict(datetime=pd.Timestamp("2021-01-04 09:15"), open=100.5, high=105, low=100, close=104, volume=100))
    # 進場價 102、風險=進場-停損99=3點，1倍R停利目標=105
    rows.append(dict(datetime=pd.Timestamp("2021-01-04 09:16"), open=103, high=106, low=102, close=105, volume=100))
    rows += _minute_bars("2021-01-04", "09:17", 200, open=105, high=106, low=104, close=105, volume=100)
    df = pd.DataFrame(rows)

    cfg = OpeningRangeBreakoutConfig(min_range_points=1.0, slippage_points=1.0, target_r_multiple=1.0)
    trades = backtest(df, cfg)

    assert len(trades) == 1
    t = trades.iloc[0]
    assert t["exit_reason"] == "target"
    assert t["exit_price"] == 105.0
