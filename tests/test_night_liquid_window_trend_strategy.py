"""測試 tw_quant/night_liquid_window_trend_strategy.py：夜盤流動性熱區趨勢偵測。"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tw_quant.night_liquid_window_trend_strategy import NightLiquidWindowConfig, backtest  # noqa: E402

_TEST_CFG = NightLiquidWindowConfig(
    window_start=pd.Timestamp("2021-01-04 21:00").time(),
    decision_time=pd.Timestamp("2021-01-04 21:05").time(),
    window_end=pd.Timestamp("2021-01-04 21:15").time(),
    min_move_points=5.0, slippage_points=1.0,
)


def _bars(date_str: str, start: str, rows: list[dict]) -> pd.DataFrame:
    base = pd.Timestamp(f"{date_str} {start}")
    out = []
    for i, r in enumerate(rows):
        out.append(dict(datetime=base + pd.Timedelta(minutes=i), volume=1000, **r))
    return pd.DataFrame(out)


def test_persistent_one_side_triggers_long_entry_next_bar():
    rows = [dict(open=100 + i, high=100 + i + 0.5, low=100 + i - 0.5, close=100 + i) for i in range(6)]
    # 決策點(21:05,第6根)收盤=105，開盤=100，幅度5>=門檻，全程在VWAP之上
    rows += [dict(open=105, high=105.5, low=104.8, close=105.2) for _ in range(5)]
    df = _bars("2021-01-04", "21:00", rows)
    trades = backtest(df, _TEST_CFG)

    assert len(trades) == 1
    t = trades.iloc[0]
    assert t["direction"] == "long"
    assert t["entry_price"] == 105 + 1.0  # 決策點下一根K棒開盤+滑價


def test_whipsaw_across_reference_blocks_entry():
    rows = [dict(open=100, high=102, low=98, close=101),
            dict(open=101, high=101, low=97, close=98),   # 收盤跌破VWAP
            dict(open=98, high=103, low=98, close=102),    # 又收盤站上VWAP
            dict(open=102, high=104, low=101, close=103),
            dict(open=103, high=105, low=102, close=104),
            dict(open=104, high=106, low=103, close=105)]
    df = _bars("2021-01-04", "21:00", rows)
    trades = backtest(df, _TEST_CFG)
    assert trades.empty


def test_move_below_threshold_no_trade():
    rows = [dict(open=100 + i * 0.1, high=100 + i * 0.1 + 0.2, low=100 + i * 0.1 - 0.2, close=100 + i * 0.1)
            for i in range(6)]
    rows += [dict(open=100.5, high=100.6, low=100.4, close=100.5) for _ in range(5)]
    df = _bars("2021-01-04", "21:00", rows)
    trades = backtest(df, _TEST_CFG)
    assert trades.empty


def test_vwap_break_exits_with_slippage():
    rows = [dict(open=100 + i, high=100 + i + 0.5, low=100 + i - 0.5, close=100 + i) for i in range(6)]
    rows.append(dict(open=105, high=105.5, low=105, close=105.2))  # 進場
    rows.append(dict(open=105, high=105, low=95, close=96))  # 急殺跌破VWAP
    df = _bars("2021-01-04", "21:00", rows)
    trades = backtest(df, _TEST_CFG)

    assert len(trades) == 1
    t = trades.iloc[0]
    assert t["exit_reason"] == "vwap_break"
    assert t["pnl_points"] < 0


def test_bars_outside_window_are_ignored():
    rows = [dict(open=100 + i, high=100 + i + 0.5, low=100 + i - 0.5, close=100 + i) for i in range(6)]
    rows += [dict(open=105, high=105.5, low=104.8, close=105.2) for _ in range(5)]
    df = _bars("2021-01-04", "21:00", rows)
    # 加一根窗口外(20:00)的資料，不該影響結果
    outside = pd.DataFrame([dict(datetime=pd.Timestamp("2021-01-04 20:00"), open=999, high=999, low=999, close=999, volume=1)])
    df = pd.concat([outside, df], ignore_index=True)
    trades = backtest(df, _TEST_CFG)
    assert len(trades) == 1
