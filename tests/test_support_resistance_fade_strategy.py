"""測試 tw_quant/support_resistance_fade_strategy.py：壓力支撐區間逆勢操作。"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tw_quant.support_resistance_fade_strategy import (  # noqa: E402
    SupportResistanceFadeConfig, _limit_fill, backtest, compute_indicators,
)


def test_limit_fill_buy_no_gap_fills_at_level():
    assert _limit_fill(level=100.0, day_open=101.0, side="buy_limit") == 100.0


def test_limit_fill_buy_gap_improvement_fills_at_open():
    # 開盤已經比限價更低（更便宜），用開盤價成交，是價格改善不是滑價
    assert _limit_fill(level=100.0, day_open=95.0, side="buy_limit") == 95.0


def test_limit_fill_sell_no_gap_fills_at_level():
    assert _limit_fill(level=100.0, day_open=99.0, side="sell_limit") == 100.0


def test_limit_fill_sell_gap_improvement_fills_at_open():
    assert _limit_fill(level=100.0, day_open=105.0, side="sell_limit") == 105.0


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


def _quiet_rows(price: float, n: int) -> list[dict]:
    return [dict(open=price, high=price + 1 - i * 0.001, low=price - 1 + i * 0.001, close=price, volume=1000)
            for i in range(n)]


_TEST_CFG = SupportResistanceFadeConfig(
    channel_window=5, min_range_pct=1.0, trend_ma_window=8, trend_lookback=3,
    trend_slope_threshold_pct=50, stop_atr_mult=1.0, atr_window=5, max_hold_days=5, slippage_points=1.0,
)


def test_compute_indicators_support_resistance_no_lookahead():
    rows = _quiet_rows(100, 30)
    rows[25] = dict(open=100, high=150, low=99, close=149, volume=1000)  # 極端值那天
    daily = _daily(rows)
    ind = compute_indicators(daily, _TEST_CFG)
    # 極端值那天自己算出的 resistance 不該含當天的 high=150
    assert ind.loc[25, "resistance"] < 150
    # 隔天的 5 日回看窗口才會納入這根 150
    assert ind.loc[26, "resistance"] >= 150


def test_backtest_long_entry_at_support_hits_target_via_limit_fill():
    rows = _quiet_rows(100, 20)
    rows.append(dict(open=100, high=100, low=94, close=97, volume=1000))  # 支撐觸碰日：進場
    rows.append(dict(open=97, high=98, low=97, close=97, volume=1000))    # 平靜，不碰停損
    rows.append(dict(open=97, high=110, low=97, close=108, volume=1000))  # 反彈觸及壓力

    daily = _daily(rows)
    trades = backtest(_bars_from_daily(daily), _TEST_CFG)

    assert len(trades) == 1
    t = trades.iloc[0]
    assert t["direction"] == "long"
    assert t["exit_reason"] == "target"
    assert t["entry_price"] == 99.015  # 進場用限價（支撐水準本身），不是市價，沒有滑價
    assert t["exit_price"] == 100.985  # 停利同樣是限價（對面壓力），沒有滑價
    assert t["pnl_points"] > 0


def test_backtest_long_entry_stop_loss_incurs_slippage():
    rows = _quiet_rows(100, 20)
    rows.append(dict(open=100, high=100, low=94, close=97, volume=1000))  # 支撐觸碰日：進場
    rows.append(dict(open=95, high=96, low=90, close=91, volume=1000))    # 急殺跌破停損

    daily = _daily(rows)
    trades = backtest(_bars_from_daily(daily), _TEST_CFG)

    assert len(trades) == 1
    t = trades.iloc[0]
    assert t["exit_reason"] in ("gap_through", "normal_slippage")  # 停損走市價/停損單邏輯，才會有滑價
    assert t["pnl_points"] < 0


def test_trend_filter_blocks_fade_during_strong_trend():
    # 強烈上升趨勢中製造一個急殺觸碰支撐的訊號日，濾網應該擋掉，不進場
    def trend_rows(n, start=80.0, step=1.0):
        out = []
        for i in range(n):
            c = start + i * step
            out.append(dict(open=c - 0.2, high=c + 0.5 - i * 0.0001, low=c - 0.5 + i * 0.0001, close=c, volume=1000))
        return out

    rows = trend_rows(25, start=80.0, step=1.0)
    rows.append(dict(open=104, high=105, low=95, close=98, volume=1000))  # 急殺觸碰支撐
    daily = _daily(rows)
    cfg = SupportResistanceFadeConfig(
        channel_window=5, min_range_pct=1.0, trend_ma_window=8, trend_lookback=3,
        trend_slope_threshold_pct=3.0, stop_atr_mult=1.0, atr_window=5, max_hold_days=5, slippage_points=1.0,
    )
    trades = backtest(_bars_from_daily(daily), cfg)
    assert trades.empty


def test_breakout_mode_long_entry_via_stop_order_rides_to_max_hold():
    rows = _quiet_rows(100, 20)
    rows.append(dict(open=100, high=106, low=100, close=104, volume=1000))  # 突破壓力做多
    rows += _quiet_rows(110, 10)  # 站穩高檔，不跌破停損
    daily = _daily(rows)
    cfg = SupportResistanceFadeConfig(
        channel_window=5, min_range_pct=1.0, trend_ma_window=8, trend_lookback=3,
        trend_slope_threshold_pct=50, stop_atr_mult=1.0, atr_window=5, max_hold_days=5,
        slippage_points=1.0, direction_mode="breakout",
    )
    trades = backtest(_bars_from_daily(daily), cfg)

    assert len(trades) == 1
    t = trades.iloc[0]
    assert t["direction"] == "long"
    assert t["exit_reason"] == "max_hold"
    assert t["pnl_points"] > 0


def test_breakout_mode_stop_loss_on_false_breakout():
    rows = _quiet_rows(100, 20)
    rows.append(dict(open=100, high=106, low=100, close=104, volume=1000))  # 突破壓力做多
    rows.append(dict(open=103, high=104, low=95, close=96, volume=1000))  # 假突破急殺破停損
    daily = _daily(rows)
    cfg = SupportResistanceFadeConfig(
        channel_window=5, min_range_pct=1.0, trend_ma_window=8, trend_lookback=3,
        trend_slope_threshold_pct=50, stop_atr_mult=1.0, atr_window=5, max_hold_days=5,
        slippage_points=1.0, direction_mode="breakout",
    )
    trades = backtest(_bars_from_daily(daily), cfg)

    assert len(trades) == 1
    t = trades.iloc[0]
    assert t["exit_reason"] in ("gap_through", "normal_slippage")
    assert t["pnl_points"] < 0


def test_breakout_mode_entry_gap_through_uses_open_price():
    rows = _quiet_rows(100, 20)
    rows.append(dict(open=108, high=110, low=107, close=109, volume=1000))  # 開盤即跳空穿越壓力
    daily = _daily(rows)
    cfg = SupportResistanceFadeConfig(
        channel_window=5, min_range_pct=1.0, trend_ma_window=8, trend_lookback=3,
        trend_slope_threshold_pct=50, stop_atr_mult=1.0, atr_window=5, max_hold_days=5,
        slippage_points=1.0, direction_mode="breakout",
    )
    trades = backtest(_bars_from_daily(daily), cfg)

    assert len(trades) == 1
    assert trades.iloc[0]["entry_price"] == 108.0  # 跳空穿越，用開盤價成交


def test_breakout_mode_target_r_multiple_exits_at_computed_target():
    rows = _quiet_rows(100, 20)
    rows.append(dict(open=100, high=106, low=100, close=104, volume=1000))  # 突破壓力做多，進場106+1滑價=107
    rows.append(dict(open=107, high=200, low=107, close=150, volume=1000))  # 隔天噴出，遠遠超過停利目標
    daily = _daily(rows)
    cfg = SupportResistanceFadeConfig(
        channel_window=5, min_range_pct=1.0, trend_ma_window=8, trend_lookback=3,
        trend_slope_threshold_pct=50, stop_atr_mult=1.0, atr_window=5, max_hold_days=5,
        slippage_points=1.0, direction_mode="breakout", target_r_multiple=2.0,
    )
    trades = backtest(_bars_from_daily(daily), cfg)

    assert len(trades) == 1
    t = trades.iloc[0]
    assert t["exit_reason"] == "target"
    # target = entry + target_r_multiple*risk，遠低於隔天noise high=200，用限價出場（沒有滑價），
    # 確認真的是用target_r_multiple算出來的目標提早出場，不是抱到收盤/max_hold
    assert t["exit_price"] < 150  # 遠低於當天最高，確認是限價停利成交、不是收盤價
    assert t["pnl_points"] > 0


def test_breakout_mode_target_r_multiple_none_keeps_original_behavior():
    # target_r_multiple=None（預設）時，行為要跟舊版一致：完全沒有停利，
    # 即使噴出很多也繼續抱著到max_hold_days
    rows = _quiet_rows(100, 20)
    rows.append(dict(open=100, high=106, low=100, close=104, volume=1000))
    rows.append(dict(open=107, high=200, low=107, close=150, volume=1000))
    rows += _quiet_rows(150, 10)
    daily = _daily(rows)
    cfg = SupportResistanceFadeConfig(
        channel_window=5, min_range_pct=1.0, trend_ma_window=8, trend_lookback=3,
        trend_slope_threshold_pct=50, stop_atr_mult=1.0, atr_window=5, max_hold_days=5,
        slippage_points=1.0, direction_mode="breakout", target_r_multiple=None,
    )
    trades = backtest(_bars_from_daily(daily), cfg)
    assert len(trades) == 1
    assert trades.iloc[0]["exit_reason"] == "max_hold"


def test_max_hold_fallback_when_neither_target_nor_stop_hit():
    rows = _quiet_rows(100, 20)
    rows.append(dict(open=100, high=100, low=94, close=97, volume=1000))  # 進場
    for _ in range(10):
        rows.append(dict(open=97, high=98, low=97, close=97.5, volume=1000))  # 卡在中間，不碰任何一邊

    daily = _daily(rows)
    trades = backtest(_bars_from_daily(daily), _TEST_CFG)

    assert len(trades) >= 1
    assert trades.iloc[0]["exit_reason"] == "max_hold"
