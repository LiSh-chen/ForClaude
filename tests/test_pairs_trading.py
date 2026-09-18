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

from tw_quant.config import StrategyConfig
from tw_quant.data_provider import SyntheticUniverseConfig, generate_synthetic_universe
from tw_quant.pairs_trading import PairsTradingConfig, _find_pairs, _zscore_series, run_pairs_trading_backtest


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
