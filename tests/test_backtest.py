from tw_quant.backtest import run_backtest, summarize_performance
from tw_quant.config import StrategyConfig
from tw_quant.data_provider import SyntheticUniverseConfig, generate_synthetic_universe


def test_backtest_runs_and_conserves_value_with_no_trades():
    """迴圈本身不應該無中生有或憑空蒸發資金：若全程無成交，權益應恆等於期初本金。"""
    data = generate_synthetic_universe(SyntheticUniverseConfig(n_stocks=5, n_days=300, seed=1))
    cfg = StrategyConfig()  # 預設參數在這麼小的合成資料上幾乎不會觸發訊號
    result = run_backtest(data["prices"], data["margin_short"], cfg)
    if result.trades.empty:
        assert (result.equity_curve["equity"] == cfg.initial_capital).all()


def test_backtest_executes_trades_and_respects_position_cap_under_relaxed_settings():
    data = generate_synthetic_universe(SyntheticUniverseConfig(n_stocks=30, n_days=700, seed=7))
    cfg = StrategyConfig()
    cfg.regime.breadth_threshold = 0.25
    cfg.regime.volume_ratio_threshold = 0.9
    cfg.squeeze.pr_threshold = 40
    cfg.ignition.volume_multiplier = 1.2
    cfg.global_risk.max_industry_exposure_pct = 0.30

    result = run_backtest(data["prices"], data["margin_short"], cfg)
    assert len(result.trades) > 0

    metrics = summarize_performance(result, cfg.initial_capital)
    assert metrics["n_trades"] == len(result.trades)
    assert metrics["max_dd"] >= 0

    # 每一筆已實現交易的成本基礎不應超過市值上限防禦（20%）太多（含手續費誤差）
    max_allowed = cfg.initial_capital * cfg.sizing.max_position_value_pct_of_capital * 1.01
    assert (result.trades["shares"] * result.trades["entry_price"] <= max_allowed).all()


def test_backtest_forbids_odd_lot_positions():
    data = generate_synthetic_universe(SyntheticUniverseConfig(n_stocks=30, n_days=700, seed=7))
    cfg = StrategyConfig()
    cfg.regime.breadth_threshold = 0.25
    cfg.regime.volume_ratio_threshold = 0.9
    cfg.squeeze.pr_threshold = 40
    cfg.ignition.volume_multiplier = 1.2
    cfg.global_risk.max_industry_exposure_pct = 0.30

    result = run_backtest(data["prices"], data["margin_short"], cfg)
    assert (result.trades["shares"] % cfg.sizing.lot_size == 0).all()
