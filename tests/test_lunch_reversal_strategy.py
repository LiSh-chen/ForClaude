"""測試 tw_quant/lunch_reversal_strategy.py：午盤下殺放空(12:00-12:30) +
盤中反彈做多(12:30-13:00)。"""

import sys
from datetime import date
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tw_quant.lunch_reversal_strategy import LunchReversalConfig, backtest  # noqa: E402

FLAT = (17000.0, 17000.0, 17000.0, 17000.0)


def _day_bars(specs: dict[str, tuple[float, float, float, float]], d: str = "2021-06-01") -> pd.DataFrame:
    times = pd.date_range(f"{d} 08:45:00", f"{d} 13:45:00", freq="1min")
    rows = []
    for ts in times:
        key = ts.strftime("%H:%M")
        o, h, l, c = specs.get(key, FLAT)
        rows.append({"datetime": ts, "open": o, "high": h, "low": l, "close": c})
    return pd.DataFrame(rows)


def test_two_legs_use_correct_entry_exit_prices():
    specs = {
        "12:00": (17000, 17005, 16995, 16998),
        "12:30": (16980, 16985, 16975, 16982),
        "13:00": (16990, 16995, 16985, 16992),
    }
    df = _day_bars(specs)
    trades = backtest(df)

    assert len(trades) == 2
    short_leg = trades[trades["leg"] == "short_lunch_dip"].iloc[0]
    long_leg = trades[trades["leg"] == "long_afternoon_rebound"].iloc[0]

    assert short_leg["entry_price"] == 17000  # 12:00 開盤價
    assert short_leg["exit_price"] == 16980   # 12:30 開盤價
    assert short_leg["exit_reason"] == "scheduled"
    assert short_leg["pnl_points"] == 20      # 空單：17000-16980

    assert long_leg["entry_price"] == 16980   # 12:30 開盤價（跟空單出場同一個價）
    assert long_leg["exit_price"] == 16990    # 13:00 開盤價
    assert long_leg["pnl_points"] == 10       # 多單：16990-16980


def test_protective_stop_triggers_on_short_leg():
    specs = {
        "12:00": (17000, 17005, 16995, 16998),
        "12:10": (17000, 17060, 16995, 17055),  # 大漲，觸及停損
        "12:30": (16980, 16985, 16975, 16982),
        "13:00": (16990, 16995, 16985, 16992),
    }
    df = _day_bars(specs)
    trades = backtest(df, LunchReversalConfig(max_loss_points=50))

    short_leg = trades[trades["leg"] == "short_lunch_dip"].iloc[0]
    assert short_leg["exit_reason"] == "stop_loss"
    assert short_leg["exit_price"] == 17050  # entry(17000) + max_loss(50)
    assert short_leg["pnl_points"] == -50


def test_day_missing_a_required_bar_is_skipped():
    # 只給到 12:20，沒有 12:30/13:00 的資料
    times = pd.date_range("2021-06-01 08:45:00", "2021-06-01 12:20:00", freq="1min")
    df = pd.DataFrame([{"datetime": ts, "open": 17000, "high": 17000, "low": 17000, "close": 17000} for ts in times])
    trades = backtest(df)
    assert trades.empty


def test_two_trading_days_produce_four_trades():
    df1 = _day_bars({"12:00": (17000, 17001, 16999, 17000)}, d="2021-06-01")
    df2 = _day_bars({"12:00": (18000, 18001, 17999, 18000)}, d="2021-06-02")
    df = pd.concat([df1, df2], ignore_index=True)
    trades = backtest(df)
    assert len(trades) == 4
    assert set(trades["trading_date"]) == {date(2021, 6, 1), date(2021, 6, 2)}
