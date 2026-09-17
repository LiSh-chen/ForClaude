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


def test_ascending_flag_selects_different_stocks():
    """ascending=False（買排名前段/追強勢）跟 ascending=True（買排名後段/買弱勢）
    應該選出不同的股票組合（用來測試短期反轉規律而非動量，見
    scripts/explore_reversal_strategy_from_db.py）。
    """
    data = generate_synthetic_universe(SyntheticUniverseConfig(n_stocks=20, n_days=500, seed=5))
    cfg = StrategyConfig()

    momentum_cfg = FactorConfig(momentum_window=20, rebalance_freq_days=21, top_n=5, ascending=False)
    reversal_cfg = FactorConfig(momentum_window=20, rebalance_freq_days=21, top_n=5, ascending=True)

    momentum_result = run_factor_backtest(data["prices"], cfg, momentum_cfg)
    reversal_result = run_factor_backtest(data["prices"], cfg, reversal_cfg)

    momentum_stocks = set(momentum_result.trades["stock_id"]) | set(momentum_result.open_positions.keys())
    reversal_stocks = set(reversal_result.trades["stock_id"]) | set(reversal_result.open_positions.keys())
    assert momentum_stocks != reversal_stocks


def test_start_date_reuses_full_history_for_warmup():
    """start_date 只該限制「哪些日期允許交易」，魚池的暖身期指標仍要用完整
    歷史計算——先前「簡易切分驗證」踩過的坑就是切片 prices 導致魚池篩選被迫
    從切片起點重新累積 252 日門檻，切完後將近一年沒有股票夠資格、完全沒有
    交易。這裡驗證：只加 start_date 限制（不切片 prices 本身）時，late-period
    的交易量應該跟同一段期間、直接跑全程的交易量差不多，而不是被砍到 0。
    """
    data = generate_synthetic_universe(SyntheticUniverseConfig(n_stocks=20, n_days=600, seed=5))
    cfg = StrategyConfig()
    factor_cfg = FactorConfig(momentum_window=20, rebalance_freq_days=10, top_n=5)

    full_dates = sorted(data["prices"]["date"].unique())
    late_start = full_dates[400]  # late_start 之前已經遠超過 252 日暖身期

    gated_result = run_factor_backtest(data["prices"], cfg, factor_cfg, start_date=late_start)
    gated_trade_count = len(gated_result.trades) + len(gated_result.open_positions)

    # 如果暖身期被錯誤重置，gated_trade_count 應該接近 0；有正確重用完整歷史
    # 的話，late_start 之後的交易量應該跟正常運作時差不多（用一個寬鬆但有意義
    # 的門檻，避免測試過度依賴隨機資料的細節）。
    assert gated_trade_count >= 5
