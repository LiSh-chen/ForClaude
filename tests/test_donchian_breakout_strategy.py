"""測試 tw_quant/donchian_breakout_strategy.py：唐奇安突破 + ATR 移動停損。"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tw_quant.donchian_breakout_strategy import (  # noqa: E402
    DonchianConfig, _atr, _fill_price, backtest, compute_indicators,
)
from tw_quant.technical_indicators import build_daily_bars  # noqa: E402


def test_fill_price_normal_slippage_buy_stop():
    price, reason = _fill_price(level=100.0, day_open=98.0, day_high=101.0, day_low=97.0,
                                 side="buy_stop", slippage=1.5)
    assert reason == "normal_slippage"
    assert price == 101.5  # level + slippage，滑價讓買進成交價變差


def test_fill_price_gap_through_buy_stop():
    price, reason = _fill_price(level=100.0, day_open=105.0, day_high=110.0, day_low=104.0,
                                 side="buy_stop", slippage=1.5)
    assert reason == "gap_through"
    assert price == 105.0  # 開盤已經跳空穿過水準，用開盤價成交（不額外加滑價）


def test_fill_price_normal_slippage_sell_stop():
    price, reason = _fill_price(level=100.0, day_open=102.0, day_high=103.0, day_low=99.0,
                                 side="sell_stop", slippage=1.5)
    assert reason == "normal_slippage"
    assert price == 98.5  # level - slippage，滑價讓賣出成交價變差


def test_fill_price_gap_through_sell_stop():
    price, reason = _fill_price(level=100.0, day_open=95.0, day_high=96.0, day_low=94.0,
                                 side="sell_stop", slippage=1.5)
    assert reason == "gap_through"
    assert price == 95.0


def _daily(rows: list[dict]) -> pd.DataFrame:
    dates = pd.date_range("2021-01-04", periods=len(rows), freq="B")
    df = pd.DataFrame(rows)
    df["date"] = dates
    return df[["date", "open", "high", "low", "close", "volume"]]


def _quiet_rows(price: float, n: int) -> list[dict]:
    """平盤填充用的分鐘：每天 high/low 都微幅內縮一點點，避免高低點完全重複
    （因為唐奇安突破判斷用 >=，完全平盤重複的資料會在滾動視窗邊界產生假突破
    ——這是測試資料的邊界問題，不是策略邏輯本身的 bug，這裡刻意讓每天都
    嚴格比之前的滾動最高/最低點更窄，確保平盤期間不會意外觸發訊號）。"""
    return [dict(open=price, high=price + 1 - i * 0.001, low=price - 1 + i * 0.001, close=price, volume=1000)
            for i in range(n)]


def test_compute_indicators_no_lookahead_on_entry_bands():
    rows = _quiet_rows(100, 30)
    rows[25] = dict(open=100, high=150, low=99, close=149, volume=1000)  # 中間一根極端值
    daily = _daily(rows)
    ind = compute_indicators(daily, DonchianConfig(entry_window=20, exit_window=10, atr_window=14))
    # 第 26 天（index 25，剛好是極端值那根）的 upper_entry 不該含當天自己的 high=150
    assert ind.loc[25, "upper_entry"] < 150
    # 但緊接著的下一天，20 日回看窗口裡就會含到這根 150，upper_entry 應該大幅跳升
    assert ind.loc[26, "upper_entry"] >= 150


def test_backtest_long_breakout_and_atr_initial_stop_exit():
    rows = _quiet_rows(100, 25)  # TR≈2 每天 -> ATR≈2
    rows.append(dict(open=100, high=110, low=100, close=108, volume=1000))  # 突破日
    rows += _quiet_rows(100, 15)  # 之後回到平盤（不會再破前波低點，先確保停損不會提早誤觸發)
    rows.append(dict(open=99, high=99, low=90, close=91, volume=1000))  # 急殺破前波低點，觸發初始 ATR 停損

    daily = _daily(rows)
    cfg = DonchianConfig(entry_window=20, exit_window=10, atr_window=14, atr_stop_mult=2.0, slippage_points=1.0)
    trades = backtest(_bars_from_daily(daily), cfg)

    assert len(trades) >= 1
    t = trades.iloc[0]
    assert t["direction"] == "long"
    assert t["entry_reason"] == "normal_slippage"
    assert t["pnl_points"] < 0  # 後面急殺，這筆應該是虧損出場


def _bars_from_daily(daily: pd.DataFrame) -> pd.DataFrame:
    """把日K展開成該模組 backtest() 需要的 1 分鐘K棒格式（用 build_daily_bars 反向對照，
    這裡直接構造「每天只有一根 08:45 的 1 分鐘K棒」，數值等於當天日K，
    足夠讓 build_daily_bars 聚合回同樣的 open/high/low/close/volume）。"""
    rows = []
    for _, r in daily.iterrows():
        ts = pd.Timestamp(r["date"]).replace(hour=8, minute=45)
        rows.append(dict(datetime=ts, open=r["open"], high=r["high"], low=r["low"], close=r["close"], volume=r["volume"]))
    return pd.DataFrame(rows)


def test_max_hold_fallback_when_stop_never_hit():
    rows = _quiet_rows(100, 25)
    rows.append(dict(open=100, high=110, low=100, close=108, volume=1000))
    # low 緩步墊高：滾動 10 日最低點（不含當天）永遠低於當天 low，移動停損追得到但碰不到
    for i in range(80):
        base = 107 + i * 0.2
        rows.append(dict(open=base, high=base + 2, low=base, close=base + 1, volume=1000))
    daily = _daily(rows)
    cfg = DonchianConfig(entry_window=20, exit_window=10, atr_window=14, atr_stop_mult=5.0, max_hold_days=30)
    trades = backtest(_bars_from_daily(daily), cfg)

    assert len(trades) >= 1
    assert trades.iloc[0]["exit_reason"] == "max_hold"
