"""測試 tw_quant/regime.py 的市場狀態分組邏輯（純合成數字，手算驗證，不連
資料庫）。"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tw_quant.market_regime_breakdown import (
    REGIME_HIGH,
    REGIME_LOW,
    REGIME_MID,
    regime_episode_stats,
    trailing_return_regime_labels,
)


def _dates(n):
    return pd.bdate_range("2020-01-01", periods=n)


def test_labels_shift_by_one_day_anti_lookahead():
    # 前 20 天持平，第 21 天單日暴漲；這根暴漲蠟燭「當天」的標籤不該用到
    # 自己這天的漲幅（否則會被錯誤分類成強勢期）
    n = 25
    close = pd.Series([100.0] * 20 + [100.0, 100.0, 100.0, 100.0, 300.0], index=_dates(n))
    labels = trailing_return_regime_labels(close, lookback_days=5)
    assert labels.iloc[-1] != REGIME_HIGH  # 暴漲當天，T-1為止的落後報酬率還是平的


def test_labels_pick_up_prior_trend_one_day_later():
    # 一段持續上漲之後接平盤；上漲段結束後那一天（T-1 已經反映漲幅）應該
    # 被標記為強勢期
    n = 30
    trend = np.linspace(100, 200, 10)
    close = pd.Series(np.concatenate([[100.0] * 15, trend, [200.0] * 5]), index=_dates(n))
    labels = trailing_return_regime_labels(close, lookback_days=10)
    # 上漲段結束後緊接著那一天，T-1 為止的 10 日報酬率是這段期間最高的
    assert labels.iloc[25] == REGIME_HIGH


def test_three_way_split_uses_full_sample_terciles():
    n = 100
    rng = np.random.default_rng(0)
    # 讓報酬率序列本身分布夠開，三分位應該大致均分成三組
    rets = rng.normal(0, 0.02, n)
    close = pd.Series(100 * (1 + pd.Series(rets)).cumprod().values, index=_dates(n))
    labels = trailing_return_regime_labels(close, lookback_days=5).dropna()
    counts = labels.value_counts()
    for regime in (REGIME_LOW, REGIME_MID, REGIME_HIGH):
        assert regime in counts.index
    # 大致三等分（允許一些誤差，因為三分位邊界剛好卡在資料點上）
    total = counts.sum()
    for regime in (REGIME_LOW, REGIME_MID, REGIME_HIGH):
        assert counts[regime] / total == pytest.approx(1 / 3, abs=0.15)


def test_regime_episode_stats_counts_contiguous_runs():
    labels = pd.Series(
        [REGIME_HIGH, REGIME_HIGH, REGIME_MID, REGIME_HIGH, REGIME_HIGH, REGIME_HIGH, REGIME_LOW, REGIME_HIGH],
        index=_dates(8),
    )
    n_episodes, max_len = regime_episode_stats(labels, REGIME_HIGH)
    assert n_episodes == 3  # [0:2], [3:6], [7]
    assert max_len == 3  # 索引 3,4,5 那一段


def test_regime_episode_stats_handles_label_never_occurring():
    labels = pd.Series([REGIME_MID] * 5, index=_dates(5))
    n_episodes, max_len = regime_episode_stats(labels, REGIME_HIGH)
    assert n_episodes == 0
    assert max_len == 0


def test_labels_none_when_insufficient_history():
    close = pd.Series([100.0, 101.0, 102.0], index=_dates(3))
    labels = trailing_return_regime_labels(close, lookback_days=5)
    assert labels.isna().all()
