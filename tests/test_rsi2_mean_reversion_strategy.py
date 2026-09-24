"""測試 tw_quant/rsi2_mean_reversion_strategy.py：RSI(2) 均值回歸。"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tw_quant.rsi2_mean_reversion_strategy import (  # noqa: E402
    Rsi2Config, _rsi, backtest, compute_indicators,
)


def test_rsi_all_gains_is_100():
    close = pd.Series([100, 101, 102, 103, 104])
    rsi = _rsi(close, period=2)
    assert rsi.iloc[-1] == 100


def test_rsi_all_losses_is_0():
    close = pd.Series([104, 103, 102, 101, 100])
    rsi = _rsi(close, period=2)
    assert rsi.iloc[-1] < 5


def _daily(rows: list[dict]) -> pd.DataFrame:
    dates = pd.date_range("2021-01-04", periods=len(rows), freq="B")
    df = pd.DataFrame(rows)
    df["date"] = dates
    return df[["date", "open", "high", "low", "close", "volume"]]


def _bars_from_daily(daily: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in daily.iterrows():
        ts = pd.Timestamp(r["date"]).replace(hour=8, minute=45)
        rows.append(dict(datetime=ts, open=r["open"], high=r["high"], low=r["low"], close=r["close"], volume=r["volume"]))
    return pd.DataFrame(rows)


def _uptrend_rows(n: int, base: float = 100.0, step: float = 0.5) -> list[dict]:
    """緩步上漲，確保 close > 長期均線的多頭濾網成立。"""
    rows = []
    for i in range(n):
        c = base + i * step
        rows.append(dict(open=c - 0.2, high=c + 0.3, low=c - 0.3, close=c, volume=1000))
    return rows


def test_compute_indicators_trend_ma_no_lookahead():
    rows = _uptrend_rows(30)
    daily = _daily(rows)
    ind = compute_indicators(daily, Rsi2Config(trend_ma_window=10))
    # 第 10 天（index 9）的均線只能用前 10 天（index 0~9）算，不含未來
    expected = daily["close"].iloc[0:10].mean()
    assert abs(ind.loc[9, "trend_ma"] - expected) < 1e-9


def test_backtest_long_entry_uses_next_day_open_with_slippage():
    # 先建立足夠長的多頭排列歷史讓 trend_ma（20日）穩定成立，訊號日刻意急殺
    # 製造 RSI 超賣，但跌幅控制在還能維持 close > trend_ma 的範圍內
    rows = _uptrend_rows(40, base=100, step=1.0)
    rows.append(dict(open=140, high=140, low=131, close=131, volume=1000))  # 訊號日：急殺
    rows.append(dict(open=132, high=136, low=131.5, close=135, volume=1000))  # 隔天開盤進場
    rows += _uptrend_rows(15, base=135, step=1.0)  # 之後持續上漲，RSI 應該很快回升觸發出場

    daily = _daily(rows)
    cfg = Rsi2Config(rsi_period=2, oversold=15, overbought=70, trend_ma_window=20, max_hold_days=10, slippage_points=1.0)
    trades = backtest(_bars_from_daily(daily), cfg)

    assert len(trades) >= 1
    t = trades.iloc[0]
    assert t["direction"] == "long"
    # 進場日就是訊號日的隔一天，開盤價 132 + 1 滑價 = 133
    assert t["entry_price"] == 133.0


def test_max_hold_fallback_when_rsi_never_recovers():
    # 訊號後持續盤整不回升，RSI 一直卡在低檔，應該觸發 max_hold 出場
    rows = _uptrend_rows(40, base=100, step=1.0)
    rows.append(dict(open=140, high=140, low=131, close=131, volume=1000))
    for i in range(20):
        rows.append(dict(open=131, high=131.5, low=130.5, close=131, volume=1000))

    daily = _daily(rows)
    cfg = Rsi2Config(rsi_period=2, oversold=15, overbought=70, trend_ma_window=20, max_hold_days=5, slippage_points=1.0)
    trades = backtest(_bars_from_daily(daily), cfg)

    assert len(trades) >= 1
    assert trades.iloc[0]["exit_reason"] == "max_hold"
