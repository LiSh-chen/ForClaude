"""測試 tw_quant/technical_indicators.py：日頻指標 + 落後一天避免未來函數。"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tw_quant.technical_indicators import (  # noqa: E402
    add_indicators, build_daily_bars, daily_indicators, filter_trades_by_volume,
)


def _minute_bars_for_days(closes_by_day: dict[str, float]) -> pd.DataFrame:
    rows = []
    for d, c in closes_by_day.items():
        times = pd.date_range(f"{d} 08:45:00", f"{d} 13:45:00", freq="1min")
        for i, ts in enumerate(times):
            rows.append({"datetime": ts, "open": c, "high": c + 1, "low": c - 1, "close": c, "volume": 10 + i % 3})
    return pd.DataFrame(rows)


def test_build_daily_bars_aggregates_ohlcv_from_day_session_only():
    df = _minute_bars_for_days({"2021-06-01": 17000.0, "2021-06-02": 17050.0})
    daily = build_daily_bars(df)
    assert len(daily) == 2
    assert daily.loc[0, "close"] == 17000.0
    assert daily.loc[1, "close"] == 17050.0
    assert daily["volume"].iloc[0] > 0


def test_lag1_columns_are_nan_on_first_available_day():
    closes = {f"2021-06-{d:02d}": 17000 + d for d in range(1, 25)}
    df = _minute_bars_for_days(closes)
    daily = add_indicators(build_daily_bars(df))
    assert pd.isna(daily.loc[0, "rsi14_lag1"])
    assert pd.isna(daily.loc[0, "macd_hist_lag1"])
    # 任一天的 lag1 都應該等於「前一天當天」算出來的值，不是自己當天的
    for i in [5, 10, 15, 20]:
        assert daily.loc[i, "macd_hist_lag1"] == daily.loc[i - 1, "macd_hist"]
    # vol_ratio 需要 20 天滾動視窗才有值，只在視窗補滿後檢查
    assert daily.loc[20, "vol_ratio_lag1"] == daily.loc[19, "vol_ratio"]


def test_rsi_bounds_and_uptrend_gives_high_rsi():
    closes = {f"2021-06-{d:02d}": 17000 + d * 10 for d in range(1, 25)}  # 持續上漲
    df = _minute_bars_for_days(closes)
    daily = add_indicators(build_daily_bars(df))
    rsi = daily["rsi14"].dropna()
    assert (rsi >= 0).all() and (rsi <= 100).all()
    assert rsi.iloc[-1] > 70  # 持續上漲應該接近超買


def test_filter_trades_by_volume_keeps_only_days_at_or_above_threshold():
    closes = {f"2021-06-{d:02d}": 17000.0 for d in range(1, 26)}
    df = _minute_bars_for_days(closes)
    indicators = daily_indicators(df)

    trades = pd.DataFrame({
        "trading_date": indicators["date"],
        "pnl_points": range(len(indicators)),
    })

    threshold = 1.05
    filtered = filter_trades_by_volume(trades, df, threshold)

    kept_dates = set(filtered["trading_date"])
    expected_dates = set(indicators.loc[indicators["vol_ratio_lag1"] >= threshold, "date"])
    assert kept_dates == expected_dates
    assert "vol_ratio_lag1" not in filtered.columns  # 過濾用欄位不該外洩到回傳結果


def test_daily_indicators_returns_date_and_lag1_columns_keyed_by_date():
    closes = {f"2021-06-{d:02d}": 17000 + d for d in range(1, 25)}
    df = _minute_bars_for_days(closes)
    out = daily_indicators(df)
    assert set(out.columns) == {"date", "rsi14_lag1", "macd_hist_lag1", "bb_pctb_lag1", "vol_ratio_lag1"}
    assert isinstance(out["date"].iloc[0], type(pd.Timestamp("2021-01-01").date()))
