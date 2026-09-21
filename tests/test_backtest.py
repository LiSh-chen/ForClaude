import numpy as np
import pandas as pd

from tw_quant.backtest import run_backtest, summarize_performance
from tw_quant.config import CostConfig, StrategyConfig
from tw_quant.data_provider import SyntheticUniverseConfig, generate_synthetic_universe
from tw_quant.us_config import build_us_config
from tw_quant import us_costs


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


def test_entry_signal_fn_overrides_default_strategy_signals():
    """entry_signal_fn 讓呼叫端替換進場邏輯，出場/風控/成本引擎維持不變
    （見 scripts/explore_alt_strategies_from_db.py 用來比較多種進場設計）。
    """
    data = generate_synthetic_universe(SyntheticUniverseConfig(n_stocks=10, n_days=400, seed=2))
    cfg = StrategyConfig()

    def always_false(master, prices, margin_short, cfg):
        n = len(master)
        return np.zeros(n, dtype=bool), np.zeros(n, dtype=bool)

    result = run_backtest(data["prices"], data["margin_short"], cfg, entry_signal_fn=always_false)
    assert result.trades.empty

    def always_true(master, prices, margin_short, cfg):
        n = len(master)
        return np.ones(n, dtype=bool), np.zeros(n, dtype=bool)

    cfg.regime.breadth_threshold = 0.25
    cfg.regime.volume_ratio_threshold = 0.9
    cfg.global_risk.max_industry_exposure_pct = 0.30
    result = run_backtest(data["prices"], data["margin_short"], cfg, entry_signal_fn=always_true)
    assert len(result.trades) > 0


def test_membership_param_blocks_entries_outside_membership_window_without_touching_signal_indicators():
    """2026-09-21 架構修正：membership 只 AND 進最終的進場訊號，不能拿來
    事先砍 prices——否則會像 tw_quant/factor_backtest.py 修正前那樣，
    連帶弄壞 entry_signal_fn 依賴的均線/ATR 這些指標。這裡用「一律進場」
    的假訊號驗證：membership 空窗期內完全不該有任何交易，區間內則正常
    進場，且用的是完整未過濾的 prices。
    """
    data = generate_synthetic_universe(SyntheticUniverseConfig(n_stocks=3, n_days=400, seed=2))
    cfg = StrategyConfig()
    cfg.regime.breadth_threshold = 0.0
    cfg.regime.volume_ratio_threshold = 0.0
    cfg.global_risk.max_industry_exposure_pct = 1.0

    def always_true(master, prices, margin_short, cfg):
        n = len(master)
        return np.ones(n, dtype=bool), np.zeros(n, dtype=bool)

    dates = sorted(data["prices"]["date"].unique())
    cutoff = pd.Timestamp(dates[300])  # S1000 只在這天之後才是已知的指數成分股
    membership = pd.DataFrame({"stock_id": ["S1000"], "start_date": [cutoff], "end_date": [pd.NaT]})

    result = run_backtest(
        data["prices"], data["margin_short"], cfg, entry_signal_fn=always_true, membership=membership
    )

    # S1001/S1002 沒有 membership 紀錄，視為一直合格（保守不誤殺），照常交易
    assert result.trades["stock_id"].isin(["S1001", "S1002"]).all() or result.trades.empty is False
    s1000_entries = result.trades.loc[result.trades["stock_id"] == "S1000", "entry_date"]
    assert (s1000_entries >= cutoff).all()  # S1000 空窗期內完全不該有任何進場


def test_cost_module_injection_lets_us_costs_replace_tw_costs():
    """cost_module 參數讓引擎重用在別的市場成本模型上（見 tw_quant/us_costs.py）。
    不直接比較台股版 vs 美股版的損益高低——複委託每股固定手續費對低價股的
    影響比例可能反而超過台股的比例制成本，所以「美股一定比較便宜」不是
    穩健的不變量。改成驗證更直接的東西：同一個 us_costs 模組，把費率歸零
    後重跑，損益應該嚴格更高（成本確實被扣掉了，且 cost_module 真的被
    引擎使用，不是被忽略）。
    """
    data = generate_synthetic_universe(SyntheticUniverseConfig(n_stocks=30, n_days=700, seed=7))

    def always_true(master, prices, margin_short, cfg):
        n = len(master)
        return np.ones(n, dtype=bool), np.zeros(n, dtype=bool)

    us_cfg = build_us_config()
    us_cfg.regime.breadth_threshold = 0.25
    us_cfg.regime.volume_ratio_threshold = 0.9
    us_cfg.global_risk.max_industry_exposure_pct = 0.30
    us_result = run_backtest(
        data["prices"], data["margin_short"], us_cfg, entry_signal_fn=always_true, cost_module=us_costs
    )

    zero_cost_cfg = build_us_config()
    zero_cost_cfg.regime.breadth_threshold = 0.25
    zero_cost_cfg.regime.volume_ratio_threshold = 0.9
    zero_cost_cfg.global_risk.max_industry_exposure_pct = 0.30
    zero_cost_cfg.costs = CostConfig(tax_rate=0.0, fee_rate=0.0, per_share_fee=0.0, exit_slippage_ticks=0)
    zero_cost_result = run_backtest(
        data["prices"], data["margin_short"], zero_cost_cfg, entry_signal_fn=always_true, cost_module=us_costs
    )

    assert len(us_result.trades) > 0
    assert len(zero_cost_result.trades) > 0
    assert zero_cost_result.trades["pnl"].sum() > us_result.trades["pnl"].sum()


def test_pre_holiday_exit_dates_force_liquidates_and_blocks_new_entries():
    """長假風控疊加層：觸發日收盤強制平倉現有部位（隔天開盤成交），且當天
    不產生任何新候選（避免新單隔天一開盤就進場、還是曝險在缺口裡）。
    """
    data = generate_synthetic_universe(SyntheticUniverseConfig(n_stocks=10, n_days=400, seed=2))
    cfg = StrategyConfig()
    cfg.regime.breadth_threshold = 0.25
    cfg.regime.volume_ratio_threshold = 0.9
    cfg.global_risk.max_industry_exposure_pct = 0.30

    def always_true(master, prices, margin_short, cfg):
        n = len(master)
        return np.ones(n, dtype=bool), np.zeros(n, dtype=bool)

    all_dates = sorted(data["prices"]["date"].unique())
    trigger_date = all_dates[300]
    day_after_trigger = all_dates[301]

    result = run_backtest(
        data["prices"], data["margin_short"], cfg, entry_signal_fn=always_true,
        pre_holiday_exit_dates={trigger_date},
    )

    # 觸發日之前有開倉的部位，都應該在「觸發日隔天」被強制平倉
    forced_exits = result.trades[result.trades["exit_date"] == day_after_trigger]
    assert len(forced_exits) > 0

    # 觸發日當天不該產生任何新候選 -> 沒有任何交易的進場日是「觸發日隔天」
    # （因為進場日隔天執行的訊號，是觸發日當天核准的候選，應該是空的）
    entries_on_day_after = result.trades[result.trades["entry_date"] == day_after_trigger]
    assert entries_on_day_after.empty
