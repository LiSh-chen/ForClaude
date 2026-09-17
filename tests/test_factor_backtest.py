from tw_quant.backtest import summarize_performance
from tw_quant.config import StrategyConfig
from tw_quant.data_provider import SyntheticUniverseConfig, generate_synthetic_universe
from tw_quant.factor_backtest import FactorConfig, run_factor_backtest


def test_factor_backtest_conserves_value_and_produces_trades():
    """月/季調倉動能因子組合：權益曲線不應無中生有，且在合理設定下應該有換股交易。"""
    data = generate_synthetic_universe(SyntheticUniverseConfig(n_stocks=20, n_days=500, seed=5))
    cfg = StrategyConfig()
    factor_cfg = FactorConfig(momentum_window=60, rebalance_freq_days=21, top_n=8)

    result = run_factor_backtest(data["prices"], cfg, factor_cfg)

    assert (result.equity_curve["equity"] > 0).all()
    assert len(result.trades) > 0
    assert len(result.open_positions) <= factor_cfg.top_n

    metrics = summarize_performance(result, cfg.initial_capital)
    assert metrics["n_trades"] == len(result.trades)


def test_factor_backtest_respects_lot_size():
    data = generate_synthetic_universe(SyntheticUniverseConfig(n_stocks=20, n_days=500, seed=5))
    cfg = StrategyConfig()
    factor_cfg = FactorConfig(momentum_window=60, rebalance_freq_days=21, top_n=8)

    result = run_factor_backtest(data["prices"], cfg, factor_cfg)

    if not result.trades.empty:
        assert (result.trades["shares"] % cfg.sizing.lot_size == 0).all()
    for pos in result.open_positions.values():
        assert pos.shares % cfg.sizing.lot_size == 0
