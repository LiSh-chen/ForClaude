"""測試 tw_quant/cross_day_reversal_strategy.py：跨日日盤報酬反轉。"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tw_quant.cross_day_reversal_strategy import CrossDayReversalConfig, backtest  # noqa: E402


def _day_bars(date: str, open_price: float, close_price: float) -> list[dict]:
    ts = pd.date_range(f"{date} 08:45", periods=3, freq="90min")  # 08:45, 10:15, 11:45（只需要開收盤，中間隨意）
    return [
        dict(datetime=ts[0], open=open_price, high=max(open_price, close_price) + 1,
             low=min(open_price, close_price) - 1, close=(open_price + close_price) / 2, volume=100),
        dict(datetime=ts[1], open=(open_price + close_price) / 2, high=max(open_price, close_price) + 1,
             low=min(open_price, close_price) - 1, close=(open_price + close_price) / 2, volume=100),
        dict(datetime=ts[2], open=(open_price + close_price) / 2, high=max(open_price, close_price) + 1,
             low=min(open_price, close_price) - 1, close=close_price, volume=100),
    ]


def test_positive_prior_day_triggers_short_today():
    rows = _day_bars("2021-01-04", 100, 110) + _day_bars("2021-01-05", 111, 115)  # day1漲10點，day2也漲
    df = pd.DataFrame(rows)
    trades = backtest(df, CrossDayReversalConfig(slippage_points=1.0))

    assert len(trades) == 1  # day1沒有前一天資料，跳過；只有day2有訊號
    t = trades.iloc[0]
    assert t["direction"] == "short"
    assert t["prior_return"] == 10
    assert t["entry_price"] == 111 - 1  # 開盤價-滑價（放空進場滑價讓成交價變差）
    assert t["exit_price"] == 115 + 1  # 收盤價+滑價（回補滑價讓成交價變差）


def test_negative_prior_day_triggers_long_today():
    rows = _day_bars("2021-01-04", 100, 90) + _day_bars("2021-01-05", 89, 95)  # day1跌10點
    df = pd.DataFrame(rows)
    trades = backtest(df, CrossDayReversalConfig(slippage_points=1.0))

    assert len(trades) == 1
    t = trades.iloc[0]
    assert t["direction"] == "long"
    assert t["prior_return"] == -10
    assert t["entry_price"] == 89 + 1
    assert t["exit_price"] == 95 - 1


def test_first_day_has_no_prior_return_and_is_skipped():
    rows = _day_bars("2021-01-04", 100, 110)
    df = pd.DataFrame(rows)
    trades = backtest(df, CrossDayReversalConfig())
    assert trades.empty


def test_min_abs_yesterday_return_filters_small_moves():
    rows = _day_bars("2021-01-04", 100, 102) + _day_bars("2021-01-05", 103, 108)  # day1只漲2點
    df = pd.DataFrame(rows)
    trades = backtest(df, CrossDayReversalConfig(min_abs_yesterday_return=5.0))
    assert trades.empty

    trades_lenient = backtest(df, CrossDayReversalConfig(min_abs_yesterday_return=0.0))
    assert len(trades_lenient) == 1
