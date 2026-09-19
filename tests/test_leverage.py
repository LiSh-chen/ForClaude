"""測試 tw_quant/leverage.py 的槓桿 + 保證金維持率模擬邏輯（純合成數字，
手算驗證，不連資料庫）。"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tw_quant.leverage import LeverageConfig, metrics_from_equity_curve, simulate_leveraged_equity, vol_target_leverage_series


def _dates(n):
    return pd.Series(pd.bdate_range("2020-01-01", periods=n))


def test_leverage_1x_zero_cost_matches_unlevered_exactly():
    n = 10
    rets = pd.Series([0.01, -0.02, 0.015, 0.0, -0.01, 0.02, 0.01, -0.005, 0.0, 0.03])
    cfg = LeverageConfig(annual_borrow_rate=0.0, maintenance_margin_pct=0.0, releverage_freq_days=1)
    result = simulate_leveraged_equity(_dates(n), rets, 1.0, cfg, 10_000.0)

    expected = 10_000.0 * (1 + rets).cumprod()
    for got, exp in zip(result["equity"], expected):
        assert got == pytest.approx(exp, rel=1e-9)


def test_leverage_2x_zero_cost_day1_matches_hand_calc():
    n = 3
    rets = pd.Series([0.01, 0.01, 0.01])
    cfg = LeverageConfig(annual_borrow_rate=0.0, maintenance_margin_pct=0.0, releverage_freq_days=100)
    result = simulate_leveraged_equity(_dates(n), rets, 2.0, cfg, 10_000.0)

    # day0: V0=20000, B0=10000; day1 return 1% -> V=20200, B=10000, equity=10200
    assert result["equity"].iloc[0] == pytest.approx(10_200.0, rel=1e-9)
    # leverage right after day0 should read V/equity = 20200/10200
    assert result["leverage"].iloc[0] == pytest.approx(20200.0 / 10200.0, rel=1e-9)


def test_borrow_cost_drags_equity_when_price_flat():
    n = 5
    rets = pd.Series([0.0] * n)
    cfg = LeverageConfig(annual_borrow_rate=0.08, maintenance_margin_pct=0.0, releverage_freq_days=100)
    result = simulate_leveraged_equity(_dates(n), rets, 2.0, cfg, 10_000.0)

    daily_rate = 0.08 / 252
    # V stays at 20000 (r=0), B grows by daily_rate each day from 10000
    expected_equity = 20_000.0 - 10_000.0 * (1 + daily_rate) ** np.arange(1, n + 1)
    for got, exp in zip(result["equity"], expected_equity):
        assert got == pytest.approx(exp, rel=1e-6)
    # equity should be strictly decreasing (financing cost drag with no price movement)
    assert (result["equity"].diff().dropna() < 0).all()


def test_margin_call_triggers_on_large_drawdown_and_deleverages_to_1x():
    # 2x leverage, one huge drawdown day should breach the 30% maintenance margin
    n = 3
    rets = pd.Series([0.0, -0.40, 0.0])
    cfg = LeverageConfig(annual_borrow_rate=0.0, maintenance_margin_pct=0.30, releverage_freq_days=100)
    result = simulate_leveraged_equity(_dates(n), rets, 2.0, cfg, 10_000.0)

    assert result["margin_call"].iloc[1] == True  # noqa: E712
    assert result["leverage"].iloc[1] == pytest.approx(1.0, rel=1e-9)
    # after forced deleverage to 1x, a flat day should not move equity further
    assert result["equity"].iloc[2] == pytest.approx(result["equity"].iloc[1], rel=1e-9)


def test_no_margin_call_when_ratio_stays_above_maintenance():
    n = 2
    rets = pd.Series([0.0, -0.05])  # small drawdown, 2x leverage shouldn't breach 30% maintenance
    cfg = LeverageConfig(annual_borrow_rate=0.0, maintenance_margin_pct=0.30, releverage_freq_days=100)
    result = simulate_leveraged_equity(_dates(n), rets, 2.0, cfg, 10_000.0)
    assert result["margin_call"].iloc[1] == False  # noqa: E712


def test_equity_never_goes_negative_on_catastrophic_drawdown():
    n = 2
    rets = pd.Series([0.0, -0.99])
    cfg = LeverageConfig(annual_borrow_rate=0.0, maintenance_margin_pct=0.30, releverage_freq_days=100)
    result = simulate_leveraged_equity(_dates(n), rets, 2.0, cfg, 10_000.0)
    assert (result["equity"] >= 0).all()


def test_releverage_only_happens_at_configured_frequency():
    n = 6
    rets = pd.Series([0.0] * n)
    cfg = LeverageConfig(annual_borrow_rate=0.0, maintenance_margin_pct=0.0, releverage_freq_days=3)
    # target leverage changes every day, but should only be picked up at i=0 and i=3
    target = pd.Series([1.0, 3.0, 3.0, 2.0, 3.0, 3.0])
    result = simulate_leveraged_equity(_dates(n), rets, target, cfg, 10_000.0)
    assert result["leverage"].iloc[0] == pytest.approx(1.0)
    assert result["leverage"].iloc[3] == pytest.approx(2.0)


def test_vol_target_leverage_series_shifts_by_one_day():
    rets = pd.Series(np.concatenate([np.zeros(25), [0.5]]))  # a big jump only on the last day
    cfg = LeverageConfig(target_vol=0.20, vol_lookback_days=21, min_leverage=0.5, max_leverage=2.0)
    lev = vol_target_leverage_series(rets, cfg)
    # the jump on the last day must not affect that same day's leverage (anti-lookahead)
    assert lev.iloc[-1] != pytest.approx(cfg.min_leverage)  # low realized vol as of T-1 -> near max leverage
    assert lev.iloc[-1] == pytest.approx(cfg.max_leverage, rel=1e-6)


def test_vol_target_leverage_series_respects_bounds():
    rng = np.random.default_rng(0)
    rets = pd.Series(rng.normal(0, 0.05, 100))  # high vol -> should clip to min_leverage
    cfg = LeverageConfig(target_vol=0.05, vol_lookback_days=21, min_leverage=0.5, max_leverage=2.0)
    lev = vol_target_leverage_series(rets, cfg).dropna()
    assert (lev >= cfg.min_leverage - 1e-9).all()
    assert (lev <= cfg.max_leverage + 1e-9).all()
    assert (lev == cfg.min_leverage).any()  # high vol input should hit the floor somewhere


def test_metrics_from_equity_curve_handles_wipeout():
    values = pd.Series([100.0, 50.0, 0.0, 0.0])
    m = metrics_from_equity_curve(values)
    assert m["total_return"] == pytest.approx(-1.0)
    assert m["cagr"] == -1.0
