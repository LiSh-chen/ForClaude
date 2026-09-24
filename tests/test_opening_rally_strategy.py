"""測試 tw_quant/opening_rally_strategy.py：開盤 08:45-09:00 做多。"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tw_quant.opening_rally_strategy import OpeningRallyConfig, backtest  # noqa: E402

FLAT = (17000.0, 17000.0, 17000.0, 17000.0)


def _day_bars(specs: dict[str, tuple[float, float, float, float]], d: str = "2021-06-01") -> pd.DataFrame:
    times = pd.date_range(f"{d} 08:45:00", f"{d} 13:45:00", freq="1min")
    rows = []
    for ts in times:
        key = ts.strftime("%H:%M")
        o, h, l, c = specs.get(key, FLAT)
        rows.append({"datetime": ts, "open": o, "high": h, "low": l, "close": c})
    return pd.DataFrame(rows)


def test_entry_and_exit_use_scheduled_bar_opens():
    specs = {
        "08:45": (17000, 17010, 16995, 17005),
        "09:00": (17020, 17025, 17015, 17022),
    }
    df = _day_bars(specs)
    trades = backtest(df)

    assert len(trades) == 1
    t = trades.iloc[0]
    assert t["entry_price"] == 17000
    assert t["exit_price"] == 17020
    assert t["pnl_points"] == 20
    assert t["pnl_twd"] == 1000.0


def test_configurable_entry_and_exit_times():
    from datetime import time
    specs = {
        "09:00": (17000, 17005, 16995, 17002),
        "09:30": (17040, 17045, 17035, 17042),
    }
    df = _day_bars(specs)
    trades = backtest(df, OpeningRallyConfig(entry=time(9, 0), exit=time(9, 30)))

    t = trades.iloc[0]
    assert t["entry_price"] == 17000
    assert t["exit_price"] == 17040
    assert t["pnl_points"] == 40


def test_day_missing_exit_bar_is_skipped():
    times = pd.date_range("2021-06-01 08:45:00", "2021-06-01 08:55:00", freq="1min")
    df = pd.DataFrame([{"datetime": ts, "open": 17000, "high": 17000, "low": 17000, "close": 17000} for ts in times])
    trades = backtest(df)
    assert trades.empty


def test_multiple_days_produce_one_trade_each():
    df1 = _day_bars({}, d="2021-06-01")
    df2 = _day_bars({}, d="2021-06-02")
    df = pd.concat([df1, df2], ignore_index=True)
    trades = backtest(df)
    assert len(trades) == 2
