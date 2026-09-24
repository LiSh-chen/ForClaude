"""測試 tw_quant/trend_day_strategy.py：VWAP持續性趨勢日偵測。"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tw_quant.trend_day_strategy import TrendDayConfig, backtest  # noqa: E402


def _day_bars(date: str, prices: list[float], start: str = "08:45", volume: float = 100) -> list[dict]:
    ts = pd.date_range(f"{date} {start}", periods=len(prices), freq="min")
    return [dict(datetime=tm, open=c - 0.05, high=c + 0.1, low=c - 0.1, close=c, volume=volume)
            for tm, c in zip(ts, prices)]


def test_monotonic_uptrend_day_triggers_long_and_rides_to_close():
    prices = [100 + i * 0.2 for i in range(300)]
    df = pd.DataFrame(_day_bars("2021-01-04", prices))
    trades = backtest(df, TrendDayConfig(slippage_points=1.0))

    assert len(trades) == 1
    t = trades.iloc[0]
    assert t["direction"] == "long"
    assert t["exit_reason"] == "session_close"
    assert t["pnl_points"] > 0


def test_monotonic_downtrend_day_triggers_short():
    prices = [200 - i * 0.2 for i in range(300)]
    df = pd.DataFrame(_day_bars("2021-01-04", prices))
    trades = backtest(df, TrendDayConfig(slippage_points=1.0))

    assert len(trades) == 1
    assert trades.iloc[0]["direction"] == "short"
    assert trades.iloc[0]["pnl_points"] > 0


def test_choppy_day_crossing_vwap_is_not_a_trend_candidate():
    import math
    prices = [100 + 3 * math.sin(2 * math.pi * i / 20) for i in range(300)]
    df = pd.DataFrame(_day_bars("2021-01-04", prices))
    trades = backtest(df, TrendDayConfig())
    assert trades.empty


def test_insufficient_move_is_skipped():
    # 單邊但幅度太小：300分鐘漲15點，decision_time(11:00,第135分鐘)時只有約6.75點，< 預設20點門檻
    prices = [100 + i * 0.05 for i in range(300)]
    df = pd.DataFrame(_day_bars("2021-01-04", prices))
    trades = backtest(df, TrendDayConfig())
    assert trades.empty


def test_vwap_break_after_entry_triggers_early_exit():
    rise = [100 + i * 0.2 for i in range(150)]
    fall = [130 - (i - 150) * 0.5 for i in range(150, 300)]
    prices = rise + fall
    df = pd.DataFrame(_day_bars("2021-01-04", prices))
    trades = backtest(df, TrendDayConfig(slippage_points=1.0))

    assert len(trades) == 1
    t = trades.iloc[0]
    assert t["exit_reason"] == "vwap_break"
    assert t["pnl_points"] < 0


def test_relaxed_dominant_fraction_allows_one_early_noise_crossing():
    prices = [100 + i * 0.2 for i in range(300)]
    prices[5] = prices[5] - 2  # 早盤一根雜訊拉回，跌破當時還很貼近價格的VWAP
    df = pd.DataFrame(_day_bars("2021-01-04", prices))

    strict = backtest(df, TrendDayConfig())
    relaxed = backtest(df, TrendDayConfig(min_dominant_side_fraction=0.99))

    assert strict.empty
    assert len(relaxed) == 1
    assert relaxed.iloc[0]["direction"] == "long"


def test_no_trade_when_no_bar_after_decision_time():
    # 只到decision_time(11:00)為止就沒資料了，沒有下一根K棒可以進場
    prices = [100 + i * 0.2 for i in range(136)]  # 08:45 + 135分鐘 = 11:00 是最後一根
    df = pd.DataFrame(_day_bars("2021-01-04", prices))
    trades = backtest(df, TrendDayConfig())
    assert trades.empty
