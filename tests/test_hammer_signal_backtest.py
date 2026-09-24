"""測試 tw_quant/hammer_signal_backtest.py：長下影線 + 次根紅K 訊號接固定
風報比停損停利的交易模擬引擎，以及成本模型。

用手動構造的 1 分鐘K棒序列驗證：
- 做多：停利/停損正確觸發，風險/停利點位計算正確
- 做空：跟做多共用同一個 risk（點數），只是停損停利方向相反
- 停損距離 > max_risk_points 時放棄該筆訊號
- 都沒碰到 SL/TP 時，max_hold_minutes 後強制出場
- 訊號後緊接著一段 >2 分鐘的間隔（跨盤）時，以間隔前最後一根收盤價出場
- 同一時間只允許一筆未平倉部位，訊號出現時若已在場內則忽略
- 成本模型：稅金正確依名目契約價值計算，net = gross - tax - commission
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tw_quant.hammer_signal_backtest import (  # noqa: E402
    TradeCost, apply_costs, build_base_arrays, build_htf_trend, simulate,
)

FLAT = (17000.0, 17000.0, 17000.0, 17000.0)  # open, high, low, close（body=0，不會被當訊號）


def _bars(specs: list[tuple[float, float, float, float]], start: str = "2021-06-01 21:00:00") -> pd.DataFrame:
    times = pd.date_range(start, periods=len(specs), freq="1min")
    rows = [{"datetime": ts, "open": o, "high": h, "low": l, "close": c} for ts, (o, h, l, c) in zip(times, specs)]
    return pd.DataFrame(rows)


def test_long_take_profit():
    specs = [FLAT] * 10
    # 訊號K棒：長下影線，low=16985
    specs[0] = (17000, 17001, 16985, 16999)
    # 次根紅K，收盤=17000 -> 進場價
    specs[1] = (16999, 17000, 16998, 17000)
    # 之後價格上漲觸及停利
    specs[2] = (17010, 17045, 17005, 17040)

    df = _bars(specs)
    arrays = build_base_arrays(df)
    trades = simulate(arrays, ratio_threshold=0.5, r_multiple=2.0, direction="long")

    assert len(trades) == 1
    t = trades.iloc[0]
    assert t["entry_price"] == 17000
    assert t["risk_points"] == 20  # 17000 - (16985 - 5)
    assert t["sl"] == 16980
    assert t["tp"] == 17040
    assert t["exit_reason"] == "take_profit"
    assert t["exit_price"] == 17040
    assert t["pnl_points"] == 40


def test_long_stop_loss():
    specs = [FLAT] * 10
    specs[0] = (17000, 17001, 16985, 16999)
    specs[1] = (16999, 17000, 16998, 17000)
    specs[2] = (16999, 17001, 16975, 16999)  # low 跌破 SL=16980

    df = _bars(specs)
    arrays = build_base_arrays(df)
    trades = simulate(arrays, ratio_threshold=0.5, r_multiple=2.0, direction="long")

    t = trades.iloc[0]
    assert t["exit_reason"] == "stop_loss"
    assert t["exit_price"] == 16980
    assert t["pnl_points"] == -20


def test_short_mirrors_long_risk_but_opposite_direction():
    specs = [FLAT] * 10
    specs[0] = (17000, 17001, 16985, 16999)
    specs[1] = (16999, 17000, 16998, 17000)
    # 空單停損在 entry+risk=17020 之上；此根觸及 17020 應停損出場
    specs[2] = (17000, 17025, 16999, 17010)

    df = _bars(specs)
    arrays = build_base_arrays(df)
    trades = simulate(arrays, ratio_threshold=0.5, r_multiple=2.0, direction="short")

    t = trades.iloc[0]
    assert t["entry_price"] == 17000
    assert t["risk_points"] == 20
    assert t["sl"] == 17020  # entry + risk（做空停損在上方）
    assert t["tp"] == 16960  # entry - risk*2（做空停利在下方）
    assert t["exit_reason"] == "stop_loss"
    assert t["exit_price"] == 17020
    assert t["pnl_points"] == -20  # entry(17000) - exit(17020)


def test_max_risk_filter_skips_signal():
    specs = [FLAT] * 5
    specs[0] = (17000, 17001, 16800, 16999)  # 下影線極長，risk 遠超過 140
    specs[1] = (16999, 17000, 16998, 17000)

    df = _bars(specs)
    arrays = build_base_arrays(df)
    trades = simulate(arrays, ratio_threshold=0.5, r_multiple=2.0, direction="long", max_risk_points=140)

    assert trades.empty


def test_max_hold_fallback_forces_exit():
    specs = [FLAT] * 8
    specs[0] = (17000, 17001, 16985, 16999)
    specs[1] = (16999, 17000, 16998, 17000)  # entry=17000, sl=16980, tp(r=2)=17040
    # 進場後盤整、都沒碰到 SL/TP
    for i in range(2, 8):
        specs[i] = (17000, 17010, 16990, 17005)

    df = _bars(specs)
    arrays = build_base_arrays(df)
    trades = simulate(arrays, ratio_threshold=0.5, r_multiple=2.0, direction="long", max_hold_minutes=3)

    t = trades.iloc[0]
    assert t["exit_reason"] == "max_hold"
    # 訊號在 index 0，進場在 entry_idx=1，max_hold=3 -> exit_idx = 1+3 = 4
    assert t["exit_idx"] == 4
    assert t["exit_price"] == specs[4][3]


def test_block_end_fallback_on_session_gap():
    specs = [FLAT] * 6
    specs[0] = (17000, 17001, 16985, 16999)
    specs[1] = (16999, 17000, 16998, 17000)
    specs[2] = (17000, 17010, 16990, 17005)  # 進場後這根盤整，緊接著發生跳盤

    times = pd.date_range("2021-06-01 21:00:00", periods=3, freq="1min")
    # 第 3 根之後直接跳到很晚的時間（間隔 > 2 分鐘），形成新的連續區塊
    times_gap = pd.date_range("2021-06-01 23:00:00", periods=3, freq="1min")
    all_times = list(times) + list(times_gap)
    all_specs = specs[:3] + [FLAT, FLAT, FLAT]
    df = pd.DataFrame([
        {"datetime": ts, "open": o, "high": h, "low": l, "close": c}
        for ts, (o, h, l, c) in zip(all_times, all_specs)
    ])

    arrays = build_base_arrays(df)
    trades = simulate(arrays, ratio_threshold=0.5, r_multiple=2.0, direction="long", max_hold_minutes=10)

    t = trades.iloc[0]
    assert t["exit_reason"] == "block_end"
    assert t["exit_price"] == specs[2][3]  # 跳盤前最後一根（index 2）的收盤價


def test_one_trade_at_a_time_blocks_overlapping_signal():
    specs = [FLAT] * 12
    specs[0] = (17000, 17001, 16985, 16999)
    specs[1] = (16999, 17000, 16998, 17000)  # entry, sl=16980, tp(r=2)=17040
    # 場內期間出現另一個技術上合格的訊號，應該被忽略（因為還沒平倉）
    specs[3] = (17000, 17001, 16985, 16999)
    specs[4] = (16999, 17010, 16998, 17005)
    for i in range(2, 12):
        if i not in (3, 4):
            specs[i] = (17000, 17010, 16990, 17005)  # 盤整，不觸發 SL/TP

    df = _bars(specs)
    arrays = build_base_arrays(df)
    trades = simulate(arrays, ratio_threshold=0.5, r_multiple=2.0, direction="long", max_hold_minutes=20)

    assert len(trades) == 1  # 第二個訊號被場內部位擋掉


def test_build_htf_trend_has_no_lookahead():
    """5 分鐘窗口、SMA(2) 手算驗證：一根 1 分鐘K棒絕對不能用到「跟自己同一時刻
    才剛收盤」的高週期K棒——這根K棒本身必須落在下一個高週期窗口才看得到它。"""
    times = pd.date_range("2021-06-01 00:00:00", periods=20, freq="1min")
    closes = [101 + i for i in range(20)]
    df = pd.DataFrame({"datetime": times, "open": closes, "high": closes, "low": closes, "close": closes})

    htf = build_htf_trend(df, freq="5min", sma_window=2)

    # 前 11 根（00:00~00:10）都还没有可用的高週期趨勢值（不夠 2 根 5 分鐘K棒算 SMA，
    # 且 00:10 這根本身不能拿來用在 00:10 自己身上）
    assert (htf[:11] == 0).all()
    # 00:11 起，00:10 那根 5 分鐘K棒（收盤 111，SMA(106,111)=108.5）已經收盤可用 -> 多頭
    assert (htf[11:] == 1).all()


def test_simulate_trend_filter_with_and_against():
    specs = [FLAT] * 8
    specs[0] = (17000, 17001, 16985, 16999)
    specs[1] = (16999, 17000, 16998, 17000)
    specs[2] = (17010, 17045, 17005, 17040)  # 觸及多單停利

    df = _bars(specs)
    arrays = build_base_arrays(df)

    # 訊號K棒（index 0）當下高週期趨勢設為「多頭(1)」
    htf_up = pd.Series(1, index=range(len(specs))).to_numpy()

    # long + with(順勢，多單只在高週期多頭進場) -> 應該成交
    t_with = simulate(arrays, ratio_threshold=0.5, r_multiple=2.0, direction="long",
                       htf_trend=htf_up, trend_mode="with")
    assert len(t_with) == 1

    # long + against(逆勢，多單只在高週期空頭進場) -> 高週期是多頭，應該被過濾掉
    t_against = simulate(arrays, ratio_threshold=0.5, r_multiple=2.0, direction="long",
                          htf_trend=htf_up, trend_mode="against")
    assert t_against.empty


def test_apply_costs_tax_and_net():
    trades = pd.DataFrame([
        {"entry_price": 17000.0, "exit_price": 17040.0, "pnl_points": 40.0},
        {"entry_price": 17000.0, "exit_price": 16980.0, "pnl_points": -20.0},
    ])
    cost = TradeCost("測試", commission_round_trip=60.0, tax_rate_per_side=0.00002, point_value=50.0)
    priced = apply_costs(trades, cost)

    expected_tax_0 = 0.00002 * 50.0 * (17000.0 + 17040.0)
    assert priced.loc[0, "gross_twd"] == 40.0 * 50.0
    assert abs(priced.loc[0, "cost_twd"] - (expected_tax_0 + 60.0)) < 1e-9
    assert abs(priced.loc[0, "net_twd"] - (2000.0 - expected_tax_0 - 60.0)) < 1e-9


def test_apply_costs_includes_slippage_when_set():
    trades = pd.DataFrame([{"entry_price": 17000.0, "exit_price": 17040.0, "pnl_points": 40.0}])
    no_slip = TradeCost("無滑價", commission_round_trip=60.0, point_value=50.0)
    with_slip = TradeCost("2點滑價", commission_round_trip=60.0, point_value=50.0, slippage_points_round_trip=2.0)

    priced_no_slip = apply_costs(trades, no_slip)
    priced_with_slip = apply_costs(trades, with_slip)

    assert priced_with_slip.loc[0, "cost_twd"] - priced_no_slip.loc[0, "cost_twd"] == 2.0 * 50.0
    assert priced_no_slip.loc[0, "net_twd"] - priced_with_slip.loc[0, "net_twd"] == 100.0
