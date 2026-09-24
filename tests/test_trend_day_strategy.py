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


def _two_day_bars(prices_day2: list[float]) -> pd.DataFrame:
    # day1 用比 _day_bars 更寬的高低點範圍，模擬合理量級的日線ATR
    # （若沿用 _day_bars 的窄範圍，算出的ATR只有0.2點，遠小於1點滑價，
    # 會讓移動停損在進場當下就被滑價本身觸發，是測試資料的問題不是策略邏輯）
    ts1 = pd.date_range("2021-01-04 08:45", periods=300, freq="min")
    day1 = [dict(datetime=tm, open=100, high=101, low=99, close=100, volume=100) for tm in ts1]
    day2 = _day_bars("2021-01-05", prices_day2)
    return pd.DataFrame(day1 + day2)


def test_min_move_pct_scales_with_price_level():
    # 開盤價1000，決策時點漲幅約13.5點（1.35%）：
    # 絕對點數門檻20點不會通過，相對百分比門檻1%會通過
    prices = [1000 + i * 0.1 for i in range(300)]
    df = pd.DataFrame(_day_bars("2021-01-04", prices))

    by_points = backtest(df, TrendDayConfig(min_move_points=20))
    by_pct = backtest(df, TrendDayConfig(min_move_pct=0.01))

    assert by_points.empty
    assert len(by_pct) == 1


def test_trailing_atr_rides_to_close_on_clean_uptrend():
    prices = [100 + i * 0.2 for i in range(300)]
    df = _two_day_bars(prices)
    trades = backtest(df, TrendDayConfig(exit_mode="trailing_atr", atr_stop_mult=1.5))

    assert len(trades) == 1
    t = trades.iloc[0]
    assert t["direction"] == "long"
    assert t["exit_reason"] == "session_close"
    assert t["pnl_points"] > 0


def test_trailing_atr_exits_early_on_pullback_after_entry():
    prices = [100 + i * 0.2 if i <= 150 else 130 - (i - 150) * 0.3 for i in range(300)]
    df = _two_day_bars(prices)
    trades = backtest(df, TrendDayConfig(exit_mode="trailing_atr", atr_stop_mult=1.5))

    assert len(trades) == 1
    t = trades.iloc[0]
    assert t["exit_reason"] == "trailing_stop"
    assert t["pnl_points"] < 0


def test_trailing_atr_skipped_when_no_prior_day_atr():
    # 只有一天資料，沒有前一天可以算ATR
    prices = [100 + i * 0.2 for i in range(300)]
    df = pd.DataFrame(_day_bars("2021-01-05", prices))
    trades = backtest(df, TrendDayConfig(exit_mode="trailing_atr"))
    assert trades.empty


def test_open_reference_monotonic_uptrend_matches_vwap_reference():
    prices = [100 + i * 0.2 for i in range(300)]
    df = pd.DataFrame(_day_bars("2021-01-04", prices))
    trades = backtest(df, TrendDayConfig(reference="open"))

    assert len(trades) == 1
    t = trades.iloc[0]
    assert t["direction"] == "long"
    assert t["exit_reason"] == "session_close"


def test_open_reference_exit_uses_fixed_day_open_not_moving_vwap():
    # 上漲後回落，開盤價（固定）比VWAP（跟著價格墊高）更晚被跌破，
    # 兩種參考水準的出場點必須明顯不同，證明真的各自獨立運作
    rise = [100 + i * 0.2 for i in range(150)]
    fall = [130 - (i - 150) * 0.3 for i in range(150, 300)]
    prices = rise + fall
    df = pd.DataFrame(_day_bars("2021-01-04", prices))

    open_trades = backtest(df, TrendDayConfig(reference="open"))
    vwap_trades = backtest(df, TrendDayConfig(reference="vwap"))

    assert open_trades.iloc[0]["exit_reason"] == "open_break"
    assert vwap_trades.iloc[0]["exit_reason"] == "vwap_break"
    assert open_trades.iloc[0]["exit_price"] < vwap_trades.iloc[0]["exit_price"]  # open版本停損比較晚觸發


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
