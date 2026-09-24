"""測試 tw_quant/liquidity_sweep_reversal_strategy.py：流動性掃單反轉。"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tw_quant.liquidity_sweep_reversal_strategy import LiquiditySweepConfig, backtest  # noqa: E402

_TEST_CFG = LiquiditySweepConfig(
    channel_window=5, sweep_buffer_points=2.0, stop_buffer_points=1.0,
    target_r_multiple=2.0, max_hold_days=5, slippage_points=0.5,
)


def _day(date_str: str, bars: list[dict]) -> pd.DataFrame:
    base = pd.Timestamp(f"{date_str} 08:45:00")
    rows = []
    for i, b in enumerate(bars):
        rows.append(dict(datetime=base + pd.Timedelta(minutes=i), volume=1000, **b))
    return pd.DataFrame(rows)


def _quiet_day(date_str: str, price: float, n: int = 5) -> pd.DataFrame:
    bars = [dict(open=price, high=price + 0.5, low=price - 0.5, close=price) for _ in range(n)]
    return _day(date_str, bars)


def _quiet_history(price: float, n_days: int = 6) -> pd.DataFrame:
    dates = pd.bdate_range("2021-01-04", periods=n_days)
    return pd.concat([_quiet_day(d.strftime("%Y-%m-%d"), price) for d in dates], ignore_index=True)


def test_sweep_and_reclaim_triggers_long_entry_next_bar_open():
    hist = _quiet_history(100.0, n_days=6)  # support/resistance算出來會是99.5~100.5左右
    next_date = pd.bdate_range("2021-01-04", periods=7)[-1].strftime("%Y-%m-%d")
    signal_day = _day(next_date, [
        dict(open=99.5, high=99.8, low=96.5, close=97.0),   # 刺穿支撐(99.5-2=97.5門檻)，收盤未收回
        dict(open=97.0, high=99.8, low=96.8, close=99.8),   # 收盤收回支撐之上，確認訊號
        dict(open=99.9, high=100.2, low=99.7, close=100.0), # 進場K棒（訊號下一根）
        dict(open=100.0, high=100.3, low=99.9, close=100.1),
    ])
    df = pd.concat([hist, signal_day], ignore_index=True)
    trades = backtest(df, _TEST_CFG)

    assert len(trades) == 1
    t = trades.iloc[0]
    assert t["direction"] == "long"
    assert t["entry_price"] == 99.9 + 0.5  # 下一根K棒開盤+滑價


def test_sweep_without_reclaim_no_trade():
    hist = _quiet_history(100.0, n_days=6)
    next_date = pd.bdate_range("2021-01-04", periods=7)[-1].strftime("%Y-%m-%d")
    signal_day = _day(next_date, [
        dict(open=99.5, high=99.8, low=96.5, close=97.0),  # 刺穿支撐但整天沒收回
        dict(open=97.0, high=97.5, low=96.0, close=96.5),
        dict(open=96.5, high=97.0, low=96.0, close=96.8),
    ])
    df = pd.concat([hist, signal_day], ignore_index=True)
    trades = backtest(df, _TEST_CFG)
    assert trades.empty


def test_no_sweep_no_trade():
    hist = _quiet_history(100.0, n_days=10)
    trades = backtest(hist, _TEST_CFG)
    assert trades.empty


def test_stop_loss_incurs_slippage_via_fill_price():
    hist = _quiet_history(100.0, n_days=6)
    d0, d1 = pd.bdate_range("2021-01-04", periods=8)[-2:]
    signal_day = _day(d0.strftime("%Y-%m-%d"), [
        dict(open=99.5, high=99.8, low=96.5, close=97.0),
        dict(open=97.0, high=99.8, low=96.8, close=99.8),   # 訊號確認
        dict(open=99.9, high=100.2, low=99.7, close=100.0), # 進場價100.4，停損=96.5-1=95.5
        dict(open=99.9, high=99.9, low=94.0, close=94.5),   # 同一天內急殺跌破停損
    ])
    df = pd.concat([hist, signal_day], ignore_index=True)
    trades = backtest(df, _TEST_CFG)

    assert len(trades) == 1
    t = trades.iloc[0]
    assert t["exit_reason"] == "stop"
    assert t["pnl_points"] < 0


def test_target_hit_uses_limit_fill_no_slippage():
    hist = _quiet_history(100.0, n_days=6)
    d0 = pd.bdate_range("2021-01-04", periods=7)[-1]
    signal_day = _day(d0.strftime("%Y-%m-%d"), [
        dict(open=99.5, high=99.8, low=96.5, close=97.0),
        dict(open=97.0, high=99.8, low=96.8, close=99.8),
        dict(open=99.9, high=100.2, low=99.7, close=100.0),  # 進場價100.4，風險=100.4-95.5=4.9，停利=100.4+9.8=110.2
        dict(open=100.0, high=111.0, low=99.9, close=110.5),  # 大漲觸及停利
    ])
    df = pd.concat([hist, signal_day], ignore_index=True)
    trades = backtest(df, _TEST_CFG)

    assert len(trades) == 1
    t = trades.iloc[0]
    assert t["exit_reason"] == "target"
    assert t["exit_price"] == t["target"]  # 限價出場，剛好在停利價，沒有滑價
    assert t["pnl_points"] > 0


def test_max_hold_fallback_when_neither_stop_nor_target_hit():
    hist = _quiet_history(100.0, n_days=6)
    dates = pd.bdate_range("2021-01-04", periods=20)
    signal_date = dates[5]
    signal_day = _day(signal_date.strftime("%Y-%m-%d"), [
        dict(open=99.5, high=99.8, low=96.5, close=97.0),
        dict(open=97.0, high=99.8, low=96.8, close=99.8),
        dict(open=99.9, high=100.2, low=99.7, close=100.0),  # 進場
        dict(open=100.0, high=100.3, low=99.9, close=100.1),
    ])
    later_quiet = pd.concat(
        [_quiet_day(d.strftime("%Y-%m-%d"), 100.0) for d in dates[6:12]], ignore_index=True,
    )
    df = pd.concat([hist, signal_day, later_quiet], ignore_index=True)
    trades = backtest(df, _TEST_CFG)

    assert len(trades) == 1
    assert trades.iloc[0]["exit_reason"] == "max_hold"
