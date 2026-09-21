import pandas as pd

from tw_quant.backtest import summarize_performance
from tw_quant.config import CostConfig, StrategyConfig
from tw_quant.data_provider import SyntheticUniverseConfig, generate_synthetic_universe
from tw_quant.factor_backtest import FactorConfig, run_factor_backtest
from tw_quant.us_config import build_us_config
from tw_quant import us_costs


def _two_stock_prices(n_days: int = 400) -> pd.DataFrame:
    """建一組兩檔股票的合成資料，專門用來重現 MRVL 那種「長期股票、但
    membership 只記錄近期一小段區間」的情境：FAST 漲得比 SLOW 快（動量
    永遠比較強），兩檔都有完整 n_days 天的連續歷史（遠超過
    min_history_days=252），收盤價單調上升、站在自己的 60 日均線之上，
    成交量固定夠大，確保兩檔在「有資格交易」這件事上只差 membership，
    不會被均線/均量條件干擾判斷。
    """
    dates = pd.bdate_range("2020-01-02", periods=n_days)
    rows = []
    for stock_id, slope in [("FAST", 3.0), ("SLOW", 1.0)]:
        for i, d in enumerate(dates):
            price = 100.0 + slope * i
            rows.append(
                {
                    "date": d, "stock_id": stock_id, "industry": "IND0",
                    "open": price, "high": price, "low": price, "close": price,
                    "volume": 5_000_000, "turnover_value": price * 5_000_000,
                }
            )
    return pd.DataFrame(rows)


def test_membership_param_excludes_stock_before_its_window_without_touching_its_indicators():
    """2026-09-21 架構修正的端對端回歸測試（重現 MRVL bug 的最小案例）：
    FAST 有完整 400 天連續歷史、動量永遠比 SLOW 強，但 membership 只記錄
    「第 350 天之後」是成分股。在 membership 區間開始「之前」的調倉日，
    FAST 即使動量最強也不該被選中；「之後」的調倉日，FAST 應該正確入選
    ——因為它的均線/均量/歷史長度是用完整 400 天序列算的，不會因為
    membership 區間短就被誤判成歷史不足 252 天（這正是修正前 MRVL 被
    誤傷的那個 bug）。
    """
    prices = _two_stock_prices(n_days=400)
    dates = sorted(prices["date"].unique())
    membership_start = dates[350]

    membership = pd.DataFrame({"stock_id": ["FAST"], "start_date": [membership_start], "end_date": [pd.NaT]})

    cfg = build_us_config()
    factor_cfg = FactorConfig(momentum_window=30, rebalance_freq_days=30, top_n=1, ascending=False)

    # 第一次調倉在 membership 區間開始「之前」：FAST 動量最強，但還不是
    # 成分股，應該選不到，退而求其次選 SLOW。
    early_end = dates[300]
    early_result = run_factor_backtest(prices, cfg, factor_cfg, end_date=early_end, cost_module=us_costs, membership=membership)
    early_positions = set(early_result.open_positions.keys()) | set(early_result.trades["stock_id"]) if not early_result.trades.empty else set(early_result.open_positions.keys())
    assert "FAST" not in early_positions
    assert "SLOW" in early_positions

    # 之後的調倉在 membership 區間開始「之後」：FAST 現在是成分股了，
    # 均線/歷史長度用完整序列算，依然合格，動量最強，應該被選中。
    late_end = dates[390]
    late_result = run_factor_backtest(prices, cfg, factor_cfg, end_date=late_end, cost_module=us_costs, membership=membership)
    assert "FAST" in late_result.open_positions


def test_membership_none_keeps_old_behavior_unchanged():
    """membership=None（預設值）時，行為要跟修正前完全一樣——這個參數是
    選填的，不能影響任何沒有傳它的既有呼叫端（台股回測、舊腳本）。
    """
    prices = _two_stock_prices(n_days=400)
    cfg = build_us_config()
    factor_cfg = FactorConfig(momentum_window=30, rebalance_freq_days=30, top_n=1, ascending=False)

    result = run_factor_backtest(prices, cfg, factor_cfg, cost_module=us_costs)

    # 沒有 membership 限制時，FAST 動量永遠最強，應該從一開始就被選中。
    assert "FAST" in result.open_positions


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


def test_rebalance_dates_anchored_to_full_history_not_slice_start():
    """調倉日曆要錨定在完整歷史的第一天，不能被 start_date 切片起點污染——
    不然同一組 factor_cfg，光是 start_date 給的日期不同，調倉日期本身就會
    跟著平移，對高度集中的 top_n 小組合來說，買到/賣掉的股票可能完全不同，
    報酬率能差到一倍以上（見 2026-09-19 對話紀錄：把同一段歷史切成兩段
    分別回測、再跟一次連續回測接起來對比時，總報酬對不上，根源就是這個）。
    """
    data = generate_synthetic_universe(SyntheticUniverseConfig(n_stocks=20, n_days=600, seed=5))
    cfg = StrategyConfig()
    factor_cfg = FactorConfig(momentum_window=20, rebalance_freq_days=10, top_n=5)

    full_dates = sorted(data["prices"]["date"].unique())
    expected_rebalance_dates = set(full_dates[:: factor_cfg.rebalance_freq_days])

    late_start = full_dates[303]  # 刻意選一個「不是」完整歷史調倉日的日期
    assert late_start not in expected_rebalance_dates

    gated_result = run_factor_backtest(data["prices"], cfg, factor_cfg, start_date=late_start)

    entry_dates = set(gated_result.trades["entry_date"]) | {
        pos.entry_date for pos in gated_result.open_positions.values()
    }
    assert entry_dates, "測試前提：這個窗格內應該至少進場過一次"
    assert entry_dates <= expected_rebalance_dates


def test_top_n_one_still_enters_despite_fee_on_top_of_full_budget():
    """top_n=1 時 per_stock_budget = 全部資金，floor 完股數後剛好用滿預算，
    手續費疊上去會讓 total_cost 略微超過 cash，導致每次進場都被 continue
    跳過、整段回測 0 筆交易——這是 US S&P 500 動量策略持股檔數網格
    （scripts/test_us_momentum_topn_from_db.py）測 top_n=1~2 時，用合成
    資料跑煙霧測試才第一次抓到的既有引擎 bug，其他既有測試/正式回測都
    沒踩到（top_n 通常 >= 3，budget 沒有剛好用滿，手續費還有餘裕空間）。
    """
    data = generate_synthetic_universe(SyntheticUniverseConfig(n_stocks=10, n_days=500, seed=3))
    factor_cfg = FactorConfig(momentum_window=60, rebalance_freq_days=21, top_n=1)

    result = run_factor_backtest(data["prices"], build_us_config(), factor_cfg, cost_module=us_costs)

    assert len(result.trades) > 0 or len(result.open_positions) > 0


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
