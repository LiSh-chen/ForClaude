"""測試 tw_quant/etf_combo.py 的組合模擬邏輯本身（合成資料，不連資料庫、
不連 yfinance）。"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tw_quant.config import CostConfig
from tw_quant.etf_combo import metrics_from_equity_curve, simulate_weighted_portfolio

COST_CFG = CostConfig(tax_rate=0.0000278, fee_rate=0.0, per_share_fee=0.1, exit_slippage_ticks=2)
ZERO_COST_CFG = CostConfig(tax_rate=0.0, fee_rate=0.0, per_share_fee=0.0, exit_slippage_ticks=0)


def _flat_prices(n_days: int, tickers: dict[str, float]) -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=n_days, freq="B")
    return pd.DataFrame({t: [p] * n_days for t, p in tickers.items()}, index=dates)


def test_weights_must_sum_to_one():
    prices = _flat_prices(5, {"A": 100.0, "B": 100.0})
    with pytest.raises(ValueError, match="總和必須是 1.0"):
        simulate_weighted_portfolio(prices, {"A": 0.5, "B": 0.4}, COST_CFG, 10_000.0, None)


def test_no_rebalance_flat_prices_value_stays_at_initial_minus_entry_cost():
    prices = _flat_prices(10, {"A": 100.0, "B": 100.0})
    values, n_rebal, fills = simulate_weighted_portfolio(prices, {"A": 0.5, "B": 0.5}, COST_CFG, 10_000.0, None)

    assert n_rebal == 0
    # 買進 50 股 A + 50 股 B，各花 100*50=5000，手續費 50*0.1=5 元/檔，
    # 總成本 10000 + 5 + 5 = 10010，但只有 10000 資金，取整股後應該略少於 50 股
    assert values.iloc[0] < 10_000.0
    # 價格完全不動，之後市值應該維持不變（沒有再平衡交易）
    assert (values == values.iloc[0]).all()


def test_no_rebalance_with_zero_cost_matches_buyhold_return_exactly():
    dates = pd.date_range("2020-01-01", periods=3, freq="B")
    prices = pd.DataFrame({"A": [100.0, 110.0, 121.0]}, index=dates)  # +10% 兩次
    values, n_rebal, fills = simulate_weighted_portfolio(prices, {"A": 1.0}, ZERO_COST_CFG, 10_000.0, None)

    assert n_rebal == 0
    assert values.iloc[0] == pytest.approx(10_000.0)
    assert values.iloc[-1] / values.iloc[0] == pytest.approx(1.21, rel=1e-6)


def test_rebalance_triggers_on_expected_days_and_pulls_weight_back():
    # A 持續上漲、B 持平：不再平衡的話 A 的權重會越飄越高；
    # 設 rebalance_freq_days=2，應該在 i=2,4,6,8 觸發，共 4 次
    dates = pd.date_range("2020-01-01", periods=9, freq="B")
    prices = pd.DataFrame(
        {"A": [100.0 * (1.05**i) for i in range(9)], "B": [100.0] * 9}, index=dates
    )
    values, n_rebal, fills = simulate_weighted_portfolio(prices, {"A": 0.5, "B": 0.5}, ZERO_COST_CFG, 10_000.0, 2)

    assert n_rebal == 4
    # 期初 2 筆建倉（A、B 各一筆），之後 A 持續上漲、B 持平：每次再平衡都該
    # 賣一點漲多的 A、買一點沒漲的 B，才能拉回 50/50
    initial_fills = [f for f in fills if f["date"] == dates[0]]
    assert {f["ticker"] for f in initial_fills} == {"A", "B"}
    assert all(f["action"] == "buy" for f in initial_fills)
    rebalance_fills = [f for f in fills if f["date"] != dates[0]]
    assert len(rebalance_fills) > 0
    assert any(f["ticker"] == "A" and f["action"] == "sell" for f in rebalance_fills)
    assert any(f["ticker"] == "B" and f["action"] == "buy" for f in rebalance_fills)


def test_rebalance_costs_more_than_no_rebalance_under_pure_drift_no_real_edge():
    # A、B 兩檔完全同步漲跌（相關性 1），再平衡在這種情況下沒有任何
    # 「逢高賣、逢低買」的分散效果，只會白白產生交易成本——用這個極端
    # case 驗證「有成本的再平衡」最終市值不會高於「不再平衡」
    dates = pd.date_range("2020-01-01", periods=100, freq="B")
    rng = np.random.default_rng(0)
    shared_path = 100.0 * np.cumprod(1 + rng.normal(0.0005, 0.01, size=100))
    prices = pd.DataFrame({"A": shared_path, "B": shared_path}, index=dates)

    weights = {"A": 0.5, "B": 0.5}
    values_none, _, _ = simulate_weighted_portfolio(prices, weights, COST_CFG, 10_000.0, None)
    values_rebal, n_rebal, fills_rebal = simulate_weighted_portfolio(prices, weights, COST_CFG, 10_000.0, 5)

    assert n_rebal > 0
    assert values_rebal.iloc[-1] <= values_none.iloc[-1]


def test_metrics_from_equity_curve_matches_manual_calc_for_linear_growth():
    # 252 個交易日剛好等於 1 年，總報酬 21%，CAGR 應該等於總報酬
    values = pd.Series(np.linspace(100.0, 121.0, 252))
    m = metrics_from_equity_curve(values)

    assert m["total_return"] == pytest.approx(0.21, rel=1e-6)
    assert m["cagr"] == pytest.approx(0.21, rel=1e-3)
    assert m["max_dd"] == pytest.approx(0.0, abs=1e-9)  # 單調上升，沒有回檔
    assert m["calmar"] == 0.0  # max_dd=0 時 calmar 定義為 0，避免除以零
