"""測試 tw_quant/momentum_checkpoint_strategy.py：動能檢查點 + ATR固定停損停利
（完全不需要連續指標運算的版本）。"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tw_quant.momentum_checkpoint_strategy import (  # noqa: E402
    MomentumCheckpointConfig, backtest,
)


def _quiet_day(date: str, price: float = 100, n: int = 300, start: str = "08:45") -> list[dict]:
    ts = pd.date_range(f"{date} {start}", periods=n, freq="min")
    return [dict(datetime=tm, open=price, high=price + 1, low=price - 1, close=price, volume=100) for tm in ts]


def _rising_day(date: str, start: str = "08:45", target_at_135: float = 125.0) -> list[dict]:
    rows = []
    ts = pd.date_range(f"{date} {start}", periods=300, freq="min")
    for i, tm in enumerate(ts):
        c = 100 + i * ((target_at_135 - 100) / 135) if i <= 135 else target_at_135
        rows.append(dict(datetime=tm, open=c - 0.1, high=c + 0.2, low=c - 0.2, close=c, volume=100))
    return rows


def test_no_trade_without_prior_day_atr():
    # 只有第一天資料，沒有前一天可以算ATR，不該交易
    df = pd.DataFrame(_rising_day("2021-01-04"))
    trades = backtest(df, MomentumCheckpointConfig())
    assert trades.empty


def test_valid_setup_enters_long_and_rides_to_session_close():
    rows = _quiet_day("2021-01-04") + _rising_day("2021-01-05")
    df = pd.DataFrame(rows)
    trades = backtest(df, MomentumCheckpointConfig(target_r_multiple=None))
    assert len(trades) == 1
    t = trades.iloc[0]
    assert t["direction"] == "long"
    assert t["exit_reason"] == "session_close"


def test_checkpoint1_move_too_small_skips_day():
    rows = _quiet_day("2021-01-04") + _rising_day("2021-01-05", target_at_135=105.0)  # 幅度太小
    df = pd.DataFrame(rows)
    trades = backtest(df, MomentumCheckpointConfig())
    assert trades.empty


def _day_with_post_entry_move(date: str, post_entry_price: float, start: str = "08:45") -> list[dict]:
    rows = []
    ts = pd.date_range(f"{date} {start}", periods=300, freq="min")
    for i, tm in enumerate(ts):
        if i <= 135:
            c = 100 + i * (25 / 135)
        elif i == 136:
            c = 125.2
        elif i == 137:
            c = post_entry_price
        else:
            c = post_entry_price
        rows.append(dict(datetime=tm, open=c - 0.1, high=c + 0.2, low=c - 0.2, close=c, volume=100))
    return rows


def test_atr_stop_hit_after_entry():
    rows = _quiet_day("2021-01-04") + _day_with_post_entry_move("2021-01-05", post_entry_price=115.0)
    df = pd.DataFrame(rows)
    trades = backtest(df, MomentumCheckpointConfig(target_r_multiple=2.0))
    assert len(trades) == 1
    t = trades.iloc[0]
    assert t["pnl_points"] < 0
    assert t["exit_reason"] in ("gap_through", "normal_slippage")


def test_atr_target_hit_after_entry():
    rows = _quiet_day("2021-01-04") + _day_with_post_entry_move("2021-01-05", post_entry_price=140.0)
    df = pd.DataFrame(rows)
    trades = backtest(df, MomentumCheckpointConfig(target_r_multiple=2.0))
    assert len(trades) == 1
    t = trades.iloc[0]
    assert t["exit_reason"] == "target"
    assert t["pnl_points"] > 0


def test_short_direction_setup():
    rows = []
    ts = pd.date_range("2021-01-05 08:45", periods=300, freq="min")
    for i, tm in enumerate(ts):
        c = 100 - i * (25 / 135) if i <= 135 else 75.0
        rows.append(dict(datetime=tm, open=c + 0.1, high=c + 0.2, low=c - 0.2, close=c, volume=100))
    rows = _quiet_day("2021-01-04") + rows
    df = pd.DataFrame(rows)
    trades = backtest(df, MomentumCheckpointConfig(target_r_multiple=None))
    assert len(trades) == 1
    assert trades.iloc[0]["direction"] == "short"
