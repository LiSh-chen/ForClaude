"""測試 tw_quant/event_drift_backtest.py 的事件驅動固定持有期回測邏輯
（純合成數字，手算驗證，不連資料庫）。"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tw_quant.config import CostConfig, StrategyConfig
from tw_quant.event_drift_backtest import EventDriftConfig, _build_entry_exit_candidates, run_event_drift_backtest


def _flat_price_df(stock_id: str, dates: pd.DatetimeIndex, price: float = 100.0) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": dates, "stock_id": stock_id, "industry": "IND",
            "open": price, "high": price * 1.01, "low": price * 0.99, "close": price,
            "volume": 100000, "turnover_value": price * 100000,
        }
    )


def _zero_cost_cfg() -> StrategyConfig:
    cfg = StrategyConfig()
    cfg.costs = CostConfig(tax_rate=0.0, fee_rate=0.0, per_share_fee=0.0, exit_slippage_ticks=0)
    cfg.sizing.lot_size = 1
    return cfg


def test_build_candidates_respects_entry_lag_anti_lookahead():
    dates = pd.bdate_range("2024-01-01", periods=30)
    stock_df = _flat_price_df("S1", dates).set_index("date")[["open", "close"]]
    by_stock = {"S1": stock_df}
    events = pd.DataFrame([{"stock_id": "S1", "known_date": dates[5], "signal": 0.2}])
    drift_cfg = EventDriftConfig(entry_lag_days=3, holding_days=10)

    candidates = _build_entry_exit_candidates(events, by_stock, drift_cfg)

    assert len(candidates) == 1
    # 事件已知日期是 dates[5]，延遲 3 個交易日進場 -> dates[5+3]
    assert candidates[0]["entry_date"] == dates[8]
    assert candidates[0]["entry_date"] != dates[5]  # 事件當天絕對不能是進場日


def test_build_candidates_exits_after_exact_holding_days():
    dates = pd.bdate_range("2024-01-01", periods=60)
    stock_df = _flat_price_df("S1", dates).set_index("date")[["open", "close"]]
    by_stock = {"S1": stock_df}
    events = pd.DataFrame([{"stock_id": "S1", "known_date": dates[0], "signal": 0.2}])
    drift_cfg = EventDriftConfig(entry_lag_days=1, holding_days=20)

    candidates = _build_entry_exit_candidates(events, by_stock, drift_cfg)

    assert candidates[0]["entry_date"] == dates[1]
    assert candidates[0]["exit_date"] == dates[1 + 20]
    assert not candidates[0]["forced_exit_at_data_end"]


def test_build_candidates_filters_below_threshold():
    dates = pd.bdate_range("2024-01-01", periods=30)
    by_stock = {"S1": _flat_price_df("S1", dates).set_index("date")[["open", "close"]]}
    events = pd.DataFrame(
        [{"stock_id": "S1", "known_date": dates[0], "signal": 0.01}, {"stock_id": "S1", "known_date": dates[10], "signal": 0.5}]
    )
    drift_cfg = EventDriftConfig(entry_lag_days=1, holding_days=5, signal_threshold=0.1)

    candidates = _build_entry_exit_candidates(events, by_stock, drift_cfg)

    assert len(candidates) == 1
    assert candidates[0]["known_date"] == dates[10]


def test_build_candidates_forces_early_exit_when_holding_period_exceeds_data():
    dates = pd.bdate_range("2024-01-01", periods=15)
    by_stock = {"S1": _flat_price_df("S1", dates).set_index("date")[["open", "close"]]}
    events = pd.DataFrame([{"stock_id": "S1", "known_date": dates[0], "signal": 0.2}])
    drift_cfg = EventDriftConfig(entry_lag_days=1, holding_days=100)  # 遠超過可用資料長度

    candidates = _build_entry_exit_candidates(events, by_stock, drift_cfg)

    assert len(candidates) == 1
    assert candidates[0]["exit_date"] == dates[-1]
    assert candidates[0]["forced_exit_at_data_end"] is True


def test_build_candidates_skips_events_with_no_room_to_enter():
    dates = pd.bdate_range("2024-01-01", periods=10)
    by_stock = {"S1": _flat_price_df("S1", dates).set_index("date")[["open", "close"]]}
    events = pd.DataFrame([{"stock_id": "S1", "known_date": dates[-1], "signal": 0.2}])  # 最後一天才知道，沒有明天可以進場
    drift_cfg = EventDriftConfig(entry_lag_days=1, holding_days=5)

    candidates = _build_entry_exit_candidates(events, by_stock, drift_cfg)
    assert candidates == []


def test_run_backtest_zero_cost_matches_hand_calc_for_single_trade():
    dates = pd.bdate_range("2024-01-01", periods=30)
    prices = _flat_price_df("S1", dates, price=100.0)
    prices.loc[prices["date"] >= dates[5], ["open", "close"]] = 110.0  # 進場後價格漲 10%

    events = pd.DataFrame([{"stock_id": "S1", "known_date": dates[0], "signal": 0.2}])
    cfg = _zero_cost_cfg()
    cfg.initial_capital = 10_000.0
    drift_cfg = EventDriftConfig(entry_lag_days=1, holding_days=10, max_concurrent_positions=1)

    result = run_event_drift_backtest(prices, events, cfg, drift_cfg)

    assert len(result.trades) == 1
    trade = result.trades.iloc[0]
    assert trade["entry_date"] == dates[1]
    assert trade["exit_date"] == dates[1 + 10]
    # 進場價 100（dates[1] 還沒漲），出場價 110（dates[11] 已經漲了）
    assert trade["entry_price"] == pytest.approx(100.0)
    assert trade["exit_price"] == pytest.approx(110.0)
    assert trade["pnl_pct"] == pytest.approx(0.10, rel=1e-6)
    assert (result.equity_curve["equity"] > 0).all()


def test_run_backtest_skips_duplicate_position_in_same_stock():
    dates = pd.bdate_range("2024-01-01", periods=60)
    prices = _flat_price_df("S1", dates)
    events = pd.DataFrame(
        [
            {"stock_id": "S1", "known_date": dates[0], "signal": 0.2},
            {"stock_id": "S1", "known_date": dates[3], "signal": 0.5},  # 第一筆還持有中，這筆該被跳過
        ]
    )
    cfg = _zero_cost_cfg()
    drift_cfg = EventDriftConfig(entry_lag_days=1, holding_days=30, max_concurrent_positions=5)

    result = run_event_drift_backtest(prices, events, cfg, drift_cfg)

    assert len(result.trades) + len(result.open_positions) == 1
    assert any(r["stage"] == "duplicate_position" for r in result.rejected_log)


def test_run_backtest_respects_max_concurrent_positions_capacity():
    dates = pd.bdate_range("2024-01-01", periods=30)
    prices = pd.concat([_flat_price_df(f"S{i}", dates) for i in range(5)], ignore_index=True)
    events = pd.DataFrame([{"stock_id": f"S{i}", "known_date": dates[0], "signal": 0.2} for i in range(5)])
    cfg = _zero_cost_cfg()
    cfg.initial_capital = 100_000.0
    drift_cfg = EventDriftConfig(entry_lag_days=1, holding_days=20, max_concurrent_positions=2, capital_per_position=10_000.0)

    result = run_event_drift_backtest(prices, events, cfg, drift_cfg)

    n_positions_opened = len(result.trades) + len(result.open_positions)
    assert n_positions_opened == 2  # 5 檔同時觸發，額度只有 2 個
    assert sum(1 for r in result.rejected_log if r["stage"] == "capacity") == 3


def test_run_backtest_prefers_higher_signal_when_capacity_limited_same_day():
    dates = pd.bdate_range("2024-01-01", periods=30)
    prices = pd.concat([_flat_price_df("LOW", dates), _flat_price_df("HIGH", dates)], ignore_index=True)
    events = pd.DataFrame(
        [{"stock_id": "LOW", "known_date": dates[0], "signal": 0.05}, {"stock_id": "HIGH", "known_date": dates[0], "signal": 0.50}]
    )
    cfg = _zero_cost_cfg()
    drift_cfg = EventDriftConfig(entry_lag_days=1, holding_days=10, max_concurrent_positions=1)

    result = run_event_drift_backtest(prices, events, cfg, drift_cfg)

    opened = set(result.trades["stock_id"]) | set(result.open_positions.keys())
    assert opened == {"HIGH"}


def test_run_backtest_freed_capacity_same_day_can_be_reused():
    dates = pd.bdate_range("2024-01-01", periods=30)
    prices = pd.concat([_flat_price_df("S1", dates), _flat_price_df("S2", dates)], ignore_index=True)
    # S1 在第1天進場，持有5天，第6天出場；S2 剛好在第6天知道消息，隔天(第7天)進場
    events = pd.DataFrame(
        [{"stock_id": "S1", "known_date": dates[0], "signal": 0.2}, {"stock_id": "S2", "known_date": dates[5], "signal": 0.3}]
    )
    cfg = _zero_cost_cfg()
    drift_cfg = EventDriftConfig(entry_lag_days=1, holding_days=5, max_concurrent_positions=1)

    result = run_event_drift_backtest(prices, events, cfg, drift_cfg)

    # S1: known dates[0] -> entry dates[1] -> exit dates[1+5]=dates[6]
    # S2: known dates[5] -> entry dates[6] -> 跟 S1 出場同一天，額度應該夠用
    opened = set(result.trades["stock_id"]) | set(result.open_positions.keys())
    assert opened == {"S1", "S2"}
    assert not any(r["stage"] == "capacity" for r in result.rejected_log)


def test_run_backtest_start_end_date_restricts_which_events_can_trigger():
    dates = pd.bdate_range("2024-01-01", periods=60)
    prices = _flat_price_df("S1", dates)
    events = pd.DataFrame(
        [{"stock_id": "S1", "known_date": dates[0], "signal": 0.2}, {"stock_id": "S1", "known_date": dates[40], "signal": 0.3}]
    )
    cfg = _zero_cost_cfg()
    drift_cfg = EventDriftConfig(entry_lag_days=1, holding_days=5, max_concurrent_positions=5)

    result = run_event_drift_backtest(prices, events, cfg, drift_cfg, start_date=dates[30])

    # 第一筆事件 known_date 在 start_date 之前，不該觸發任何進場
    all_entry_dates = pd.to_datetime(
        list(result.trades["entry_date"]) if not result.trades.empty else []
    ).tolist() + [p.entry_date for p in result.open_positions.values()]
    assert all(d >= dates[30] for d in all_entry_dates)


def test_run_backtest_equity_curve_trimmed_to_requested_window_not_full_history():
    """2026-09-20 發現的真實 bug：equity_curve 原本橫跨完整 prices 的整個
    日期範圍，不管 start_date/end_date 怎麼設，導致 metrics_from_result
    算 CAGR 用的年數分母是整段歷史長度，不是真正的交易窗格長度，CAGR
    被嚴重低估。這裡驗證：只給 start_date 時，equity_curve 不該包含
    start_date 之前的任何日期，長度也不該是完整歷史的長度。
    """
    dates = pd.bdate_range("2024-01-01", periods=500)  # 遠長於窗格本身，模擬「完整歷史遠長於樣本內期間」
    prices = _flat_price_df("S1", dates)
    events = pd.DataFrame([{"stock_id": "S1", "known_date": dates[450], "signal": 0.2}])
    cfg = _zero_cost_cfg()
    drift_cfg = EventDriftConfig(entry_lag_days=1, holding_days=10, max_concurrent_positions=1)

    window_start = dates[400]
    result = run_event_drift_backtest(prices, events, cfg, drift_cfg, start_date=window_start)

    assert result.equity_curve.index.min() == window_start
    assert len(result.equity_curve) <= 100  # 遠小於完整 500 天歷史，不該被完整歷史的長度污染


def test_run_backtest_equity_curve_extends_past_end_date_for_late_triggered_position():
    """窗格結束前才觸發的部位，equity_curve 該延伸到它真正出場那天，
    不能在 end_date 當天硬生生截斷（那樣會漏掉還在跑的部位、也不符合
    「已觸發部位可以持有到窗格結束之後才出場」這個既有承諾）。"""
    dates = pd.bdate_range("2024-01-01", periods=100)
    prices = _flat_price_df("S1", dates)
    events = pd.DataFrame([{"stock_id": "S1", "known_date": dates[50], "signal": 0.2}])
    cfg = _zero_cost_cfg()
    drift_cfg = EventDriftConfig(entry_lag_days=1, holding_days=20, max_concurrent_positions=1)

    end_date = dates[55]  # 窗格在部位還沒出場之前就結束了（出場預計在 dates[51+20]=dates[71]）
    result = run_event_drift_backtest(prices, events, cfg, drift_cfg, start_date=None, end_date=end_date)

    assert result.equity_curve.index.max() >= dates[71]
    assert len(result.trades) == 1


def test_run_backtest_handles_no_events():
    dates = pd.bdate_range("2024-01-01", periods=10)
    prices = _flat_price_df("S1", dates)
    events = pd.DataFrame(columns=["stock_id", "known_date", "signal"])
    cfg = _zero_cost_cfg()
    drift_cfg = EventDriftConfig()

    result = run_event_drift_backtest(prices, events, cfg, drift_cfg)

    assert result.trades.empty
    assert result.open_positions == {}
    assert (result.equity_curve["equity"] == cfg.initial_capital).all()
