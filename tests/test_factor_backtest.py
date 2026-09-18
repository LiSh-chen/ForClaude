from tw_quant.backtest import summarize_performance
from tw_quant.config import CostConfig, StrategyConfig
from tw_quant.data_provider import SyntheticUniverseConfig, generate_synthetic_universe
from tw_quant.factor_backtest import FactorConfig, run_factor_backtest
from tw_quant.us_config import build_us_config
from tw_quant import us_costs


def test_cost_module_injection_lets_us_costs_replace_tw_costs():
    """cost_module 參數讓引擎重用在別的市場成本模型上。不直接比較台股版 vs
    美股版的損益高低——複委託每股固定手續費對低價股的影響比例可能反而
    超過台股的比例制成本，所以「美股一定比較便宜」不是穩健的不變量。
    改成驗證：同一個 us_costs 模組，把費率歸零後重跑，損益應該嚴格更高
    （成本確實被扣掉了，且 cost_module 真的被引擎使用，不是被忽略）。
    """
    data = generate_synthetic_universe(SyntheticUniverseConfig(n_stocks=20, n_days=500, seed=5))
    factor_cfg = FactorConfig(momentum_window=60, rebalance_freq_days=21, top_n=8)

    us_result = run_factor_backtest(data["prices"], build_us_config(), factor_cfg, cost_module=us_costs)

    zero_cost_cfg = build_us_config()
    zero_cost_cfg.costs = CostConfig(tax_rate=0.0, fee_rate=0.0, per_share_fee=0.0, exit_slippage_ticks=0)
    zero_cost_result = run_factor_backtest(data["prices"], zero_cost_cfg, factor_cfg, cost_module=us_costs)

    assert len(us_result.trades) > 0
    assert len(zero_cost_result.trades) > 0
    assert zero_cost_result.trades["pnl"].sum() > us_result.trades["pnl"].sum()


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


def test_signal_fn_overrides_default_momentum_ranking():
    """signal_fn 讓呼叫端替換排名依據（例如跳空幅度、當日盤中報酬），不用
    被限制在「落後報酬率」——用來測試 scripts/test_lagged_gap_signal_from_db.py
    這種跟動量無關的排名訊號。這裡驗證：換一個跟預設動量完全無關、純隨機的
    排名依據，選出的股票組合應該跟預設動量排名不同。
    """
    data = generate_synthetic_universe(SyntheticUniverseConfig(n_stocks=20, n_days=500, seed=5))
    cfg = StrategyConfig()
    factor_cfg = FactorConfig(rebalance_freq_days=21, top_n=5)

    def reversed_stock_id_signal(master):
        # 用股票代號的反向排名當訊號，跟真實動量完全無關，純粹驗證 signal_fn
        # 真的被拿去排名用，而不是被忽略、退回預設動量邏輯。
        rank_by_id = master.groupby("stock_id", sort=False).ngroup()
        return (-rank_by_id).astype(float)

    default_result = run_factor_backtest(data["prices"], cfg, factor_cfg)
    custom_result = run_factor_backtest(data["prices"], cfg, factor_cfg, signal_fn=reversed_stock_id_signal)

    default_stocks = set(default_result.trades["stock_id"]) | set(default_result.open_positions.keys())
    custom_stocks = set(custom_result.trades["stock_id"]) | set(custom_result.open_positions.keys())
    assert default_stocks != custom_stocks
