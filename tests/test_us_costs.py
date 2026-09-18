import pytest

from tw_quant.config import CostConfig
from tw_quant.us_config import build_us_config
from tw_quant.us_costs import entry_cost, exit_proceeds, round_to_tick, tick_size


def test_tick_size_is_flat_one_cent_regardless_of_price():
    assert tick_size(0.5) == 0.01
    assert tick_size(5.0) == 0.01
    assert tick_size(500.0) == 0.01
    assert tick_size(5000.0) == 0.01


def test_tick_size_zero_for_non_positive_or_nan():
    assert tick_size(0.0) == 0.0
    assert tick_size(-1.0) == 0.0


def test_round_to_tick_rounds_to_cent():
    assert round_to_tick(10.004) == pytest.approx(10.00)
    assert round_to_tick(10.006) == pytest.approx(10.01)


def test_entry_cost_zero_when_no_fees_configured():
    cfg = CostConfig(tax_rate=0.0, fee_rate=0.0, per_share_fee=0.0, exit_slippage_ticks=2)
    assert entry_cost(100.0, 10, cfg) == 0.0


def test_entry_cost_applies_per_share_fee():
    """複委託手續費是每股固定金額，不是成交金額百分比——股價不影響手續費。"""
    cfg = CostConfig(tax_rate=0.0, fee_rate=0.0, per_share_fee=0.1, exit_slippage_ticks=0)
    assert entry_cost(100.0, 10, cfg) == pytest.approx(1.0)
    assert entry_cost(5.0, 10, cfg) == pytest.approx(1.0)  # 股價不影響每股固定手續費


def test_exit_proceeds_applies_two_cent_slippage_and_no_tax_or_fee():
    cfg = CostConfig(tax_rate=0.0, fee_rate=0.0, per_share_fee=0.0, exit_slippage_ticks=2)
    slipped_price, net = exit_proceeds(100.0, 10, cfg)
    assert slipped_price == pytest.approx(99.98)
    assert net == pytest.approx(999.80)


def test_exit_proceeds_applies_sec_fee_when_tax_rate_set():
    cfg = CostConfig(tax_rate=0.0000278, fee_rate=0.0, per_share_fee=0.0, exit_slippage_ticks=0)
    _, net = exit_proceeds(100.0, 100, cfg)
    assert net == pytest.approx(100.0 * 100 * (1 - 0.0000278))


def test_exit_proceeds_deducts_per_share_fee_on_top_of_tax():
    cfg = CostConfig(tax_rate=0.0000278, fee_rate=0.0, per_share_fee=0.1, exit_slippage_ticks=0)
    _, net = exit_proceeds(100.0, 100, cfg)
    expected_notional = 100.0 * 100
    expected = expected_notional * (1 - 0.0000278) - 100 * 0.1
    assert net == pytest.approx(expected)


def test_build_us_config_sets_lot_size_one_and_us_costs():
    cfg = build_us_config()
    assert cfg.sizing.lot_size == 1
    assert cfg.costs.tax_rate == pytest.approx(0.0000278)
    assert cfg.costs.fee_rate == 0.0
    assert cfg.costs.per_share_fee == pytest.approx(0.1)
    assert cfg.costs.exit_slippage_ticks == 2
