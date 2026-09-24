"""測試 MTX 夜盤長下影線策略（tw_quant/txf_night_pinbar.py）。

用手動構造的合成 1 分鐘 K 棒（單一夜盤 session，15:00~23:30），驗證：
- 震盪策略：長下影線觸及前波低點後，次根開盤進場，1:2 停利/停損正確觸發
- 風控檢核：停損距離 > 140 點時放棄該筆訊號
- 突破策略：突破前波高點後拉回觸及、形成長下影線 -> 次根開盤進場，
  10MA 上穿後跌破 -> 次根開盤停利出場
- 強制平倉：23:30 仍在場內時，以該根開盤價出場
- 進場時間窗過濾：次根不在 21:30<=t<23:30 內就不成交

注意：body=0（開盤=收盤）時 lower_shadow=0 也會滿足 `lower_shadow >= 2*body` 這個
字面定義，等於零實體的平盤分鐘會「trivially」成立長下影線——這是規格公式本身在
body=0 時的邊界情況，不是實作 bug。因此下面所有測試的「填充用」分鐘一律給非零
實體（open != close）、且不觸及前波高低點，確保背景分鐘不會意外觸發訊號、掩蓋
真正要測的那一根。
"""

import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tw_quant.txf_night_pinbar import StrategyConfig, backtest, compute_indicators  # noqa: E402

# 非零實體、高=17002/低=16998 的「範圍界定」用分鐘：body=2, lower_shadow=1，不成立長下影線
RANGE_FILLER = {"open": 17001, "high": 17002, "low": 16998, "close": 16999}


def _session_bars(session_date: str, overrides: dict[str, dict]) -> pd.DataFrame:
    """建立一個 15:00~23:30 的完整夜盤 session。預設每根都是 RANGE_FILLER
    （非零實體、不觸發長下影線的窄幅盤整），避免 body=0 的平盤分鐘意外觸發訊號；
    再用 overrides（key='HH:MM'）覆寫指定分鐘的 OHLCV。"""
    times = pd.date_range(f"{session_date} 15:00:00", f"{session_date} 23:30:00", freq="1min")
    rows = []
    for ts in times:
        key = ts.strftime("%H:%M")
        o = overrides.get(key, RANGE_FILLER)
        rows.append({"datetime": ts, "open": o["open"], "high": o["high"],
                     "low": o["low"], "close": o["close"], "volume": o.get("volume", 10)})
    return pd.DataFrame(rows)


def test_range_strategy_hits_take_profit():
    cfg = StrategyConfig(swing_lookback=5)
    overrides = {}
    # 21:40 訊號 K 棒：長下影線，低點跌破前波低點 16998
    overrides["21:40"] = {"open": 17000, "high": 17001, "low": 16985, "close": 16999}
    # 21:41 次根開盤進場： entry=16999, sl = 16985-5=16980, risk=19, tp=16999+38=17037
    overrides["21:41"] = {"open": 16999, "high": 17000, "low": 16998, "close": 17000}
    # 隨後價格上漲觸及 17037 停利
    overrides["21:45"] = {"open": 17010, "high": 17040, "low": 17008, "close": 17038}

    df = _session_bars("2021-06-01", overrides)
    trades = backtest(df, cfg)

    range_trades = trades[trades["strategy"] == "range_hammer"]
    assert len(range_trades) == 1
    t = range_trades.iloc[0]
    assert t["entry_price"] == 16999
    assert t["sl"] == 16980
    assert t["tp"] == 17037
    assert t["exit_reason"] == "take_profit"
    assert t["exit_price"] == 17037
    assert t["pnl_points"] == 38


def test_range_strategy_r_multiple_is_configurable():
    """range_r_multiple 改成 1:3 時，TP 應該跟著變，不是寫死 1:2。"""
    cfg = StrategyConfig(swing_lookback=5, range_r_multiple=3.0)
    overrides = {}
    overrides["21:40"] = {"open": 17000, "high": 17001, "low": 16985, "close": 16999}
    overrides["21:41"] = {"open": 16999, "high": 17000, "low": 16998, "close": 17000}
    overrides["21:45"] = {"open": 17010, "high": 17057, "low": 17008, "close": 17055}

    df = _session_bars("2021-06-01", overrides)
    trades = backtest(df, cfg)

    t = trades[trades["strategy"] == "range_hammer"].iloc[0]
    assert t["entry_price"] == 16999
    assert t["sl"] == 16980
    assert t["tp"] == 16999 + 3 * 19  # risk=19, r_multiple=3 -> tp=17056
    assert t["exit_reason"] == "take_profit"
    assert t["pnl_points"] == 57


def test_range_strategy_skips_signal_when_risk_exceeds_max():
    cfg = StrategyConfig(swing_lookback=5)
    overrides = {}
    # 下影線極長，停損距離會遠超過 140 點
    overrides["21:40"] = {"open": 17000, "high": 17001, "low": 16800, "close": 16999}
    overrides["21:41"] = {"open": 16999, "high": 17000, "low": 16998, "close": 17000}

    df = _session_bars("2021-06-01", overrides)
    trades = backtest(df, cfg)

    assert trades[trades["strategy"] == "range_hammer"].empty


def test_range_strategy_stop_loss_triggers_before_take_profit_same_bar():
    cfg = StrategyConfig(swing_lookback=5)
    overrides = {}
    overrides["21:40"] = {"open": 17000, "high": 17001, "low": 16985, "close": 16999}
    overrides["21:41"] = {"open": 16999, "high": 17000, "low": 16998, "close": 17000}
    # entry=16999, sl=16980, tp=17037；此根同時觸及兩者，應以停損優先
    overrides["21:45"] = {"open": 16999, "high": 17040, "low": 16970, "close": 17000}

    df = _session_bars("2021-06-01", overrides)
    trades = backtest(df, cfg)

    t = trades[trades["strategy"] == "range_hammer"].iloc[0]
    assert t["exit_reason"] == "stop_loss"
    assert t["exit_price"] == 16980


def test_breakout_strategy_retest_pinbar_then_trailing_ma_exit():
    cfg = StrategyConfig(swing_lookback=5, ma_window=3)
    overrides = {}
    # 21:25 突破前波高點 17002
    overrides["21:25"] = {"open": 17000, "high": 17010, "low": 16999, "close": 17008}
    # 21:26~21:27 拉回但還沒踩到長下影線
    overrides["21:26"] = {"open": 17008, "high": 17008, "low": 17004, "close": 17005}
    overrides["21:27"] = {"open": 17005, "high": 17005, "low": 17002, "close": 17003}
    # 21:28 觸及但不成立長下影線（body 夠大），不消耗突破水位、繼續等待
    overrides["21:28"] = {"open": 17005, "high": 17006, "low": 17001, "close": 17002}
    # 21:29 拉回觸及被突破的前波高點（17002），且形成長下影線 -> 訊號成立
    overrides["21:29"] = {"open": 17003, "high": 17004, "low": 16995, "close": 17002}
    # 21:30 次根開盤進場（在允許進場時間內）：entry=17003, sl=16995-5=16990
    overrides["21:30"] = {"open": 17003, "high": 17010, "low": 17001, "close": 17009}
    # 之後價格上漲，10MA 隨之墊高並超過進場價，接著收黑跌破 10MA 觸發移動停利
    overrides["21:31"] = {"open": 17009, "high": 17015, "low": 17008, "close": 17012}
    overrides["21:32"] = {"open": 17012, "high": 17020, "low": 17010, "close": 17018}
    overrides["21:33"] = {"open": 17018, "high": 17018, "low": 16994, "close": 16995}  # 收盤跌破 10MA（低點仍高於 SL=16990）
    overrides["21:34"] = {"open": 17000, "high": 17002, "low": 16998, "close": 17001}  # 次根開盤出場

    df = _session_bars("2021-06-01", overrides)
    trades = backtest(df, cfg)

    bt = trades[trades["strategy"] == "breakout_retest"]
    assert len(bt) == 1
    t = bt.iloc[0]
    assert t["entry_price"] == 17003
    assert t["sl"] == 16990
    assert t["exit_reason"] == "trailing_ma"
    assert t["exit_price"] == 17000
    assert t["entry_dt"] == datetime(2021, 6, 1, 21, 30)


def test_forced_close_at_2330_uses_that_bars_open():
    cfg = StrategyConfig(swing_lookback=5)
    overrides = {}
    overrides["23:20"] = {"open": 17000, "high": 17001, "low": 16985, "close": 16999}
    overrides["23:21"] = {"open": 16999, "high": 17000, "low": 16998, "close": 17000}
    # 進場後價格盤整，直到 23:30 都沒碰到 SL/TP（entry=16999, sl=16980, tp=17037）
    for m in range(22, 30):
        overrides[f"23:{m:02d}"] = {"open": 16999, "high": 17001, "low": 16997, "close": 16999}
    overrides["23:30"] = {"open": 17005, "high": 17006, "low": 17004, "close": 17005}

    df = _session_bars("2021-06-01", overrides)
    trades = backtest(df, cfg)

    t = trades[trades["strategy"] == "range_hammer"].iloc[0]
    assert t["exit_reason"] == "forced_close"
    assert t["exit_price"] == 17005


def test_entry_outside_time_window_is_ignored():
    """訊號成立但次根時間不在 21:30<=t<23:30 內，不應該有交易。"""
    cfg = StrategyConfig(swing_lookback=5)
    overrides = {}
    overrides["15:20"] = {"open": 17000, "high": 17001, "low": 16985, "close": 16999}
    overrides["15:21"] = {"open": 16999, "high": 17000, "low": 16998, "close": 17000}

    df = _session_bars("2021-06-01", overrides)
    trades = backtest(df, cfg)

    assert trades.empty


def test_compute_indicators_pin_bar_flag():
    df = _session_bars("2021-06-01", {
        "21:00": {"open": 17000, "high": 17001, "low": 16990, "close": 17000},  # lower_shadow=10 >= 2*body(0)
        "21:01": {"open": 17000, "high": 17005, "low": 16997, "close": 17004},  # lower_shadow=3 < 2*body(4)
    })
    ind = compute_indicators(df, StrategyConfig())
    row0 = ind[ind["time"].astype(str) == "21:00:00"].iloc[0]
    row1 = ind[ind["time"].astype(str) == "21:01:00"].iloc[0]
    assert bool(row0["pin_bar"]) is True
    assert bool(row1["pin_bar"]) is False
