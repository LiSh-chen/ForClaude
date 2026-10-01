"""測試 tw_quant/robustness.py 的拔靴統計邏輯本身，用合成資料驗證：
已知性質的報酬序列（iid 常態、零超額報酬的對照組）應該得到符合預期的
Sharpe 估計跟信賴區間，不用真的接市場資料。
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tw_quant.robustness import (
    annualized_sharpe,
    bootstrap_sharpe_ci,
    daily_returns_from_equity,
    moving_block_bootstrap_indices,
    paired_bootstrap_sharpe_diff,
)


def test_annualized_sharpe_matches_manual_calculation():
    returns = np.array([0.01, -0.02, 0.015, 0.0, 0.005])
    expected = returns.mean() / returns.std(ddof=1) * np.sqrt(252)
    assert annualized_sharpe(returns) == expected


def test_annualized_sharpe_returns_nan_for_zero_variance():
    returns = np.array([0.01, 0.01, 0.01])
    assert np.isnan(annualized_sharpe(returns))


def test_moving_block_bootstrap_indices_has_correct_length_and_valid_range():
    rng = np.random.default_rng(0)
    idx = moving_block_bootstrap_indices(n=100, block_size=10, rng=rng)
    assert len(idx) == 100
    assert idx.min() >= 0
    assert idx.max() < 100


def test_moving_block_bootstrap_indices_rejects_block_larger_than_n():
    rng = np.random.default_rng(0)
    try:
        moving_block_bootstrap_indices(n=5, block_size=10, rng=rng)
        assert False, "應該要丟出 ValueError"
    except ValueError:
        pass


def test_bootstrap_sharpe_ci_brackets_true_sharpe_for_iid_normal_returns():
    """用已知年化 Sharpe ≈ 1.0 的合成 iid 常態報酬序列（夠長的樣本，n=5000
    讓樣本點估計的抽樣誤差夠小），拔靴信賴區間應該要把點估計值包在裡面，
    點估計本身也應該落在真實值附近（允許合理的抽樣誤差範圍，而不是驗證
    拔靴 CI 本身等於真實 Sharpe，因為拔靴法是在估計樣本的不確定性，不是
    消除它）。
    """
    rng = np.random.default_rng(7)
    daily_mean = 1.0 / np.sqrt(252) * 0.01  # 讓年化 Sharpe 約等於 1.0（std=0.01）
    returns = pd.Series(rng.normal(loc=daily_mean, scale=0.01, size=5000))

    result = bootstrap_sharpe_ci(returns, block_size=21, n_boot=500, seed=1)

    assert result["ci_low"] < result["point_estimate"] < result["ci_high"]
    assert 0.5 < result["point_estimate"] < 1.5


def test_bootstrap_sharpe_ci_is_wide_for_short_series():
    """樣本數很少時（模擬只有 20 筆交易那種情況），信賴區間應該明顯比
    長樣本序列寬——這正是本模組要揭露的重點：少交易次數的 Sharpe 點估計
    不能直接拿來下結論。
    """
    rng = np.random.default_rng(3)
    long_returns = pd.Series(rng.normal(loc=0.0005, scale=0.01, size=1000))
    short_returns = pd.Series(rng.normal(loc=0.0005, scale=0.01, size=40))

    long_result = bootstrap_sharpe_ci(long_returns, block_size=10, n_boot=500, seed=1)
    short_result = bootstrap_sharpe_ci(short_returns, block_size=5, n_boot=500, seed=1)

    long_width = long_result["ci_high"] - long_result["ci_low"]
    short_width = short_result["ci_high"] - short_result["ci_low"]
    assert short_width > long_width


def test_paired_bootstrap_sharpe_diff_is_exactly_zero_when_strategy_equals_benchmark():
    """配對拔靴法每次都對策略跟對照組套用同一組隨機區塊索引——如果兩條
    序列完全相同，每次重抽樣兩邊得到的 Sharpe 必然相等，差值分布應該
    精確地整個塌縮在 0 這一點（不是「大致圍繞 0」），prob_strategy_not_better
    因此精確等於 1.0（因為沒有任何一次差值 > 0），這正是配對設計要的效果：
    完全消除共同雜訊之後，沒有真實差異就是沒有，不會因為抽樣而產生假訊號。
    """
    rng = np.random.default_rng(5)
    returns = pd.Series(rng.normal(loc=0.0003, scale=0.008, size=500))

    result = paired_bootstrap_sharpe_diff(returns, returns, block_size=15, n_boot=500, seed=2)

    assert result["point_diff"] == 0.0
    assert result["ci_low"] == 0.0
    assert result["ci_high"] == 0.0
    assert result["prob_strategy_not_better"] == 1.0


def test_paired_bootstrap_sharpe_diff_detects_clear_outperformance():
    rng = np.random.default_rng(9)
    benchmark = pd.Series(rng.normal(loc=0.0002, scale=0.01, size=1000))
    # 策略報酬 = 對照組報酬 + 穩定的正超額報酬，應該要能被偵測出顯著贏過對照組
    strategy = benchmark + 0.0015

    result = paired_bootstrap_sharpe_diff(strategy, benchmark, block_size=21, n_boot=500, seed=2)

    assert result["point_diff"] > 0
    assert result["ci_low"] > 0  # 95% CI 下界都還是正的，代表優勢是穩定的
    assert result["prob_strategy_not_better"] < 0.05


def test_daily_returns_from_equity_drops_first_nan_row():
    equity = pd.Series([100.0, 110.0, 121.0])
    returns = daily_returns_from_equity(equity)
    assert len(returns) == 2
    assert np.isclose(returns.iloc[0], 0.10)
    assert np.isclose(returns.iloc[1], 0.10)
