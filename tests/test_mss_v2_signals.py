"""測試 refine_mss_strategy_from_db.py 新加的兩個濾網（突破幅度、量能確認）
邏輯是否正確。基礎的「空頭結構前提 + 突破波段高點」邏輯已經在
tests/test_ict_signals.py 驗證過，這裡只測新加的部分。
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.refine_mss_strategy_from_db import market_structure_shift_v2_mask


def _mini_frame(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df["date"] = pd.bdate_range("2024-01-01", periods=len(df))
    df["stock_id"] = "S1"
    return df


# 沿用 test_ict_signals.py 裡驗證過的「空頭結構 -> 突破波段高點」5 天骨架：
# day1-2 是「更早」的波段低點基準，day3-4 是「近期」波段低點（比更早的更低），
# day5 收盤要突破 day3~4 的波段高點(100)。
_BASE_ROWS = [
    {"low": 100, "high": 105, "close": 103, "volume": 1000},
    {"low": 98, "high": 103, "close": 99, "volume": 1000},
    {"low": 95, "high": 100, "close": 96, "volume": 1000},
    {"low": 90, "high": 97, "close": 91, "volume": 1000},
]


def test_breakout_strength_blocks_marginal_break():
    rows = _BASE_ROWS + [{"low": 99, "high": 115, "close": 101, "volume": 1000}]  # 只高於波段高點(100) 1%
    df = _mini_frame(rows)
    mask = market_structure_shift_v2_mask(df, downtrend_window=2, swing_window=2, breakout_strength=0.02)
    assert bool(mask.iloc[4]) is False


def test_breakout_strength_allows_decisive_break():
    rows = _BASE_ROWS + [{"low": 99, "high": 115, "close": 103, "volume": 1000}]  # 高於波段高點(100) 3%
    df = _mini_frame(rows)
    mask = market_structure_shift_v2_mask(df, downtrend_window=2, swing_window=2, breakout_strength=0.02)
    assert bool(mask.iloc[4]) is True


def test_volume_confirm_blocks_low_volume_breakout():
    # 前面 20 天量能都是 1000（撐起 20 日均量），突破當天量能只有 1.4 倍均量，
    # 門檻設 1.5 倍 -> 不該觸發
    warmup = [{"low": 200, "high": 205, "close": 203, "volume": 1000} for _ in range(20)]
    rows = warmup + _BASE_ROWS + [{"low": 99, "high": 115, "close": 103, "volume": 1400}]
    df = _mini_frame(rows)
    mask = market_structure_shift_v2_mask(
        df, downtrend_window=2, swing_window=2, breakout_strength=0.0, volume_confirm_mult=1.5
    )
    assert bool(mask.iloc[-1]) is False


def test_volume_confirm_allows_high_volume_breakout():
    warmup = [{"low": 200, "high": 205, "close": 203, "volume": 1000} for _ in range(20)]
    rows = warmup + _BASE_ROWS + [{"low": 99, "high": 115, "close": 103, "volume": 1600}]
    df = _mini_frame(rows)
    mask = market_structure_shift_v2_mask(
        df, downtrend_window=2, swing_window=2, breakout_strength=0.0, volume_confirm_mult=1.5
    )
    assert bool(mask.iloc[-1]) is True


def test_volume_confirm_disabled_when_mult_is_one():
    # volume_confirm_mult=1.0（預設）代表不濾，低量能一樣要能觸發（只要其他條件成立）
    rows = _BASE_ROWS + [{"low": 99, "high": 115, "close": 103, "volume": 1}]
    df = _mini_frame(rows)
    mask = market_structure_shift_v2_mask(
        df, downtrend_window=2, swing_window=2, breakout_strength=0.0, volume_confirm_mult=1.0
    )
    assert bool(mask.iloc[4]) is True
