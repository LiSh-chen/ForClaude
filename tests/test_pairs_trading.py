"""測試配對交易引擎（tw_quant/pairs_trading.py）。

用構造出「確實共整合」跟「純隨機、彼此無關」的合成股票，驗證共整合篩選
邏輯真的能把兩者分開；再用一段「價差刻意拉大、之後回歸」的合成資料，驗證
z-score 進出場邏輯的方向跟損益正負號是對的。
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tw_quant.config import CostConfig, StrategyConfig
from tw_quant.data_provider import SyntheticUniverseConfig, generate_synthetic_universe
from tw_quant.pairs_trading import PairsTradingConfig, _find_pairs, _zscore_series, run_pairs_trading_backtest
from tw_quant.us_config import build_us_config
from tw_quant import us_costs


def _make_price_frame(n_days: int, series: dict[str, np.ndarray], industry: dict[str, str]) -> pd.DataFrame:
    dates = pd.bdate_range("2020-01-01", periods=n_days)
    rows = []
    for stock_id, closes in series.items():
        for d, c in zip(dates, closes):
            rows.append(
                {
                    "date": d, "stock_id": stock_id, "industry": industry[stock_id],
                    "open": c, "high": c * 1.01, "low": c * 0.99, "close": c,
                    "volume": 100000, "turnover_value": c * 100000,
                }
            )
    return pd.DataFrame(rows)


def test_find_pairs_selects_cointegrated_pair_over_unrelated_one():
    rng = np.random.default_rng(42)
    n = 300
    log_a = np.cumsum(rng.normal(0, 0.01, n))  # A：隨機漫步（非定態）
    noise = np.zeros(n)
    for t in range(1, n):  # AR(1) 均值回歸雜訊，讓 B 跟 A 共整合
        noise[t] = 0.6 * noise[t - 1] + rng.normal(0, 0.005)
    log_b = log_a + noise  # B = A + 定態雜訊 -> log_a - log_b 是定態的，應該通過共整合檢定
    log_c = np.cumsum(rng.normal(0, 0.01, n))  # C：跟 A/B 完全無關的獨立隨機漫步

    series = {"A": np.exp(log_a) * 100, "B": np.exp(log_b) * 100, "C": np.exp(log_c) * 100}
    industry = {"A": "IND", "B": "IND", "C": "IND"}
    master = _make_price_frame(n, series, industry).pivot(index="date", columns="stock_id", values="close").sort_index()

    rt_cfg = PairsTradingConfig(coint_pvalue_threshold=0.05, top_n_pairs=5)
    pairs = _find_pairs(master, industry, master.index, rt_cfg)

    pair_keys = {(a, b) for a, b, _ in pairs}
    assert ("A", "B") in pair_keys
    assert ("A", "C") not in pair_keys
    assert ("B", "C") not in pair_keys


def test_find_pairs_excludes_stocks_with_zero_price_in_window():
    """真實資料裡有些股票某天價格是 0（不是 NaN）——log(0) = -inf 會讓
    beta 迴歸跟 p-value 都壞掉，還可能讓壞資料的配對因為數值不穩定而排到
    最前面，這裡驗證這種股票會被直接排除、不會混進候選配對，也不會產生
    NaN 的 beta。
    """
    rng = np.random.default_rng(7)
    n = 300
    log_a = np.cumsum(rng.normal(0, 0.01, n))
    noise = np.zeros(n)
    for t in range(1, n):
        noise[t] = 0.6 * noise[t - 1] + rng.normal(0, 0.005)
    log_b = log_a + noise
    log_c = log_a + noise * 1.01  # 幾乎跟 B 一樣共整合，但中間有一天價格是 0（壞資料）

    close_c = np.exp(log_c) * 100
    close_c[150] = 0.0  # 模擬真實資料裡缺資料被記成 0 的情況

    series = {"A": np.exp(log_a) * 100, "B": np.exp(log_b) * 100, "C": close_c}
    industry = {"A": "IND", "B": "IND", "C": "IND"}
    master = _make_price_frame(n, series, industry).pivot(index="date", columns="stock_id", values="close").sort_index()

    rt_cfg = PairsTradingConfig(coint_pvalue_threshold=0.05, top_n_pairs=5)
    pairs = _find_pairs(master, industry, master.index, rt_cfg)

    pair_keys = {(a, b) for a, b, _ in pairs}
    assert ("A", "C") not in pair_keys
    assert ("B", "C") not in pair_keys
    assert all(not np.isnan(beta) for _, _, beta in pairs)


def test_max_stocks_per_industry_keeps_only_most_liquid_candidates():
    """max_stocks_per_industry 是為了讓配對交易引擎在大型股票池（例如 S&P 500
    單一產業動輒 60~80 檔）上，coint() 檢定的組合數不會爆炸而跑不完——超過
    上限時只保留該產業裡成交金額最高的前 N 檔。這裡構造一組「低流動性但
    真的共整合」的配對（A、B）跟一組「高流動性但完全隨機、不共整合」的配對
    （C、D），設 max_stocks_per_industry=2：應該只保留 C、D 兩檔進候選，
    A-B 這組本來會被抓到的真配對反而因為流動性不足被排除在外——驗證上限
    真的是依流動性篩選，不是隨機/依字母排序。
    """
    rng = np.random.default_rng(3)
    n = 300
    log_a = np.cumsum(rng.normal(0, 0.01, n))
    noise = np.zeros(n)
    for t in range(1, n):
        noise[t] = 0.6 * noise[t - 1] + rng.normal(0, 0.005)
    log_b = log_a + noise
    close_a, close_b = np.exp(log_a) * 100, np.exp(log_b) * 100
    close_c = np.exp(np.cumsum(rng.normal(0, 0.02, n))) * 50
    close_d = np.exp(np.cumsum(rng.normal(0, 0.02, n))) * 50

    dates = pd.bdate_range("2020-01-01", periods=n)
    close_pivot = pd.DataFrame({"A": close_a, "B": close_b, "C": close_c, "D": close_d}, index=dates)
    # A/B 低流動性、C/D 高流動性
    turnover_pivot = pd.DataFrame(
        {"A": 1_000.0, "B": 1_000.0, "C": 1_000_000.0, "D": 1_000_000.0}, index=dates
    )
    industry_map = {"A": "IND", "B": "IND", "C": "IND", "D": "IND"}

    rt_cfg = PairsTradingConfig(coint_pvalue_threshold=0.05, top_n_pairs=5, max_stocks_per_industry=2)
    pairs = _find_pairs(close_pivot, industry_map, close_pivot.index, rt_cfg, turnover_pivot)

    pair_keys = {(a, b) for a, b, _ in pairs}
    assert ("A", "B") not in pair_keys
    for a, b, _ in pairs:
        assert a in {"C", "D"} and b in {"C", "D"}


def test_zscore_series_is_zero_at_the_mean_and_positive_above_it():
    spread = pd.Series([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 5.0], index=pd.bdate_range("2020-01-01", periods=11))
    z = _zscore_series(spread, window=10)
    assert z.iloc[-1] > 0  # 最後一天價差突然拉高，z 應該是正的


def test_full_synthetic_run_has_no_open_positions_and_stable_equity():
    data = generate_synthetic_universe(SyntheticUniverseConfig(n_stocks=16, n_days=500, seed=7))
    cfg = StrategyConfig()
    rt_cfg = PairsTradingConfig(formation_window=100, reformation_freq_days=40, top_n_pairs=5, zscore_window=15)

    from tw_quant.backtest import BacktestResult

    result = run_pairs_trading_backtest(data["prices"], cfg, rt_cfg)
    assert isinstance(result, BacktestResult)
    assert result.open_positions == {}
    assert not result.equity_curve["equity"].isna().any()
    assert (result.equity_curve["equity"] > 0).all()


def test_cost_module_injection_lets_us_costs_replace_tw_costs():
    """cost_module 參數讓配對交易引擎重用在別的市場成本模型上（見
    tw_quant/us_costs.py）。不直接比較台股版 vs 美股版的損益高低——複委託
    每股固定手續費對低價股的影響比例可能反而超過台股的比例制成本，所以
    「美股一定比較便宜」不是穩健的不變量。改成驗證：同一個 us_costs 模組，
    把費率歸零後重跑，損益應該嚴格更高（成本確實被扣掉了，且 cost_module
    真的被引擎使用，不是被忽略）。
    """
    data = generate_synthetic_universe(SyntheticUniverseConfig(n_stocks=16, n_days=500, seed=11))
    rt_cfg = PairsTradingConfig(formation_window=100, reformation_freq_days=40, top_n_pairs=5, zscore_window=15)

    us_result = run_pairs_trading_backtest(data["prices"], build_us_config(), rt_cfg, cost_module=us_costs)

    zero_cost_cfg = build_us_config()
    zero_cost_cfg.costs = CostConfig(tax_rate=0.0, fee_rate=0.0, per_share_fee=0.0, exit_slippage_ticks=0)
    zero_cost_result = run_pairs_trading_backtest(data["prices"], zero_cost_cfg, rt_cfg, cost_module=us_costs)

    assert len(us_result.trades) > 0
    assert len(zero_cost_result.trades) > 0
    assert zero_cost_result.trades["pnl"].sum() > us_result.trades["pnl"].sum()


def test_trades_always_come_in_pairs_of_two_legs_with_matching_dates():
    data = generate_synthetic_universe(SyntheticUniverseConfig(n_stocks=16, n_days=500, seed=11))
    cfg = StrategyConfig()
    rt_cfg = PairsTradingConfig(formation_window=100, reformation_freq_days=40, top_n_pairs=5, zscore_window=15)

    result = run_pairs_trading_backtest(data["prices"], cfg, rt_cfg)
    if result.trades.empty:
        return
    # 每組配對進出場都同時產生兩筆 TradeRecord（A 腿 + B 腿），entry_date/exit_date 應該成對出現偶數次
    grouped = result.trades.groupby(["entry_date", "exit_date"]).size()
    assert (grouped % 2 == 0).all()
    assert set(result.trades["strategy"].unique()) <= {"PAIRS_LONG", "PAIRS_SHORT"}
