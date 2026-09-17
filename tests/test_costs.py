from tw_quant import costs


def test_tick_size_table_boundaries():
    assert costs.tick_size(9.99) == 0.01
    assert costs.tick_size(10.0) == 0.05
    assert costs.tick_size(49.99) == 0.05
    assert costs.tick_size(50.0) == 0.1
    assert costs.tick_size(999.99) == 1.0
    assert costs.tick_size(1000.0) == 5.0


def test_exit_slippage_moves_price_down_by_n_ticks():
    price = 200.0  # 落在 [100,500) 級距，tick=0.5，兩檔滑價都不會跨到下一個級距
    slipped = costs.apply_exit_slippage(price, ticks=2)
    assert slipped < price
    assert abs(slipped - 199.0) < 1e-9


def test_exit_proceeds_nets_tax_and_fee():
    cfg = costs.CostConfig()
    slipped_price, net = costs.exit_proceeds(100.0, 1000, cfg)
    gross = slipped_price * 1000
    expected_net = gross * (1 - cfg.tax_rate - cfg.fee_rate)
    assert abs(net - expected_net) < 1e-6
    assert net < gross


def test_entry_cost_only_charges_fee_no_tax():
    cfg = costs.CostConfig()
    fee = costs.entry_cost(100.0, 1000, cfg)
    assert abs(fee - 100.0 * 1000 * cfg.fee_rate) < 1e-9
