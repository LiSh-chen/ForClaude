"""驗證 webapp/engines.py 裡「可調參數版」訊號函式，在跟研究腳本原始
寫死參數相同的數值下，產生完全一樣的訊號——這些函式是刻意逐行對照
scripts/test_*_from_db.py 複製出來的（把模組層級常數換成參數），這裡用
回歸測試確保複製過程沒有打錯任何一行邏輯，而不是每次改網頁介面都要
手動比對兩份程式碼。
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tw_quant.config import StrategyConfig
from tw_quant.data_provider import SyntheticUniverseConfig, generate_synthetic_universe
from webapp.engines import (
    make_bollinger_signal_fn,
    make_breakout_combo_signal_fn,
    make_mss_signal_fn,
    make_rsi_signal_fn,
)

from scripts.test_holiday_overlay_from_db import mss_signal_fn as original_mss_signal_fn
from scripts.test_pead_breakout_combo_from_db import make_combo_signal_fn as original_make_combo_signal_fn
from scripts.test_rsi_bollinger_reversion_from_db import (
    make_bollinger_ranking_signal_fn as original_make_bollinger_ranking_signal_fn,
    rsi_ranking_signal_fn as original_rsi_ranking_signal_fn,
)


@pytest.fixture(scope="module")
def synthetic_data():
    data = generate_synthetic_universe(SyntheticUniverseConfig(n_stocks=8, n_days=400, seed=5))
    return data["prices"], data["margin_short"]


def test_mss_signal_fn_matches_original_at_same_defaults(synthetic_data):
    prices, margin_short = synthetic_data
    cfg = StrategyConfig()
    master = prices.sort_values(["stock_id", "date"]).reset_index(drop=True).copy()

    original_entry_a, original_entry_b = original_mss_signal_fn(master, prices, margin_short, cfg)
    webapp_fn = make_mss_signal_fn(downtrend_window=20, swing_window=15, volume_mult=1.3)
    webapp_entry_a, webapp_entry_b = webapp_fn(master, prices, margin_short, cfg)

    np.testing.assert_array_equal(original_entry_a, webapp_entry_a)
    np.testing.assert_array_equal(original_entry_b, webapp_entry_b)


def test_rsi_signal_fn_matches_original_at_same_window(synthetic_data):
    prices, _ = synthetic_data
    master = prices.sort_values(["stock_id", "date"]).reset_index(drop=True).copy()

    original = original_rsi_ranking_signal_fn(master)
    webapp_result = make_rsi_signal_fn(window=14)(master)

    pd.testing.assert_series_equal(original, webapp_result, check_names=False)


@pytest.mark.parametrize("require_volume_spike", [False, True])
def test_bollinger_signal_fn_matches_original_at_same_window(synthetic_data, require_volume_spike):
    prices, _ = synthetic_data
    master = prices.sort_values(["stock_id", "date"]).reset_index(drop=True).copy()

    original = original_make_bollinger_ranking_signal_fn(require_volume_spike)(master)
    webapp_result = make_bollinger_signal_fn(window=20, require_volume_spike=require_volume_spike, volume_mult=1.5)(master)

    pd.testing.assert_series_equal(original, webapp_result, check_names=False)


def test_breakout_combo_signal_fn_matches_original_at_same_params(synthetic_data):
    prices, margin_short = synthetic_data
    master = prices.sort_values(["stock_id", "date"]).reset_index(drop=True).copy()
    cfg = StrategyConfig()

    rng = np.random.default_rng(1)
    master_with_revenue = master.copy()
    master_with_revenue["revenue_momentum_ok_shifted"] = rng.random(len(master)) > 0.7
    revenue_flag_lookup = master_with_revenue.set_index(["stock_id", "date"])["revenue_momentum_ok_shifted"]

    original_fn = original_make_combo_signal_fn(20, 2.0, master_with_revenue)
    webapp_fn = make_breakout_combo_signal_fn(20, 2.0, revenue_flag_lookup)

    original_entry_a, original_entry_b = original_fn(master, prices, margin_short, cfg)
    webapp_entry_a, webapp_entry_b = webapp_fn(master, prices, margin_short, cfg)

    np.testing.assert_array_equal(original_entry_a, webapp_entry_a)
    np.testing.assert_array_equal(original_entry_b, webapp_entry_b)
