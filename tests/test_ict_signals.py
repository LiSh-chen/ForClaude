"""測試 explore_ict_strategies_from_db.py 裡三種 ICT 風格型態判斷的原始邏輯
（不含魚池篩選，用小範例直接驗證型態判斷本身有沒有算對——這種手刻的
布林邏輯最容易出現差一天、邊界條件算錯的問題）。
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.explore_ict_strategies_from_db import (
    fvg_fill_mask,
    liquidity_sweep_mask,
    market_structure_shift_mask,
)


def _mini_frame(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df["date"] = pd.bdate_range("2024-01-01", periods=len(df))
    df["stock_id"] = "S1"
    return df


def test_liquidity_sweep_fires_on_break_and_reclaim():
    # day1-4: low/close 都撐在 100；day5 跌破過去 3 日低點(100)又收回 100 之上 -> 應觸發
    # day6: 沒有跌破前一段低點 -> 不該觸發
    rows = [
        {"low": 100, "close": 100},
        {"low": 100, "close": 100},
        {"low": 100, "close": 100},
        {"low": 100, "close": 100},
        {"low": 90, "close": 101},  # 跌破且收回 -> True
        {"low": 105, "close": 106},  # 沒跌破 -> False
    ]
    df = _mini_frame(rows)
    mask = liquidity_sweep_mask(df, sweep_lookback=3)
    assert bool(mask.iloc[4]) is True
    assert bool(mask.iloc[5]) is False


def test_liquidity_sweep_does_not_fire_on_break_without_reclaim():
    rows = [
        {"low": 100, "close": 100},
        {"low": 100, "close": 100},
        {"low": 100, "close": 100},
        {"low": 100, "close": 100},
        {"low": 90, "close": 95},  # 跌破但收盤沒拉回 100 之上 -> 不該觸發
    ]
    df = _mini_frame(rows)
    mask = liquidity_sweep_mask(df, sweep_lookback=3)
    assert bool(mask.iloc[4]) is False


def test_fvg_fill_fires_on_unfilled_gap_dip_and_reclaim():
    # gap_age=3: T=day6 (index5), gap 形成於 day3（index2）：
    #   high[day1]=90（缺口下緣）, low[day3]=95 > 90 -> 缺口存在
    #   day4/day5 都沒跌破 90 -> 缺口尚未回補
    #   day6 低點跌進缺口（<=90）但收盤又拉回缺口之上（>90）-> 應觸發
    rows = [
        {"high": 90, "low": 88, "close": 89},  # day1
        {"high": 96, "low": 94, "close": 95},  # day2（不影響這組判斷）
        {"high": 100, "low": 95, "close": 98},  # day3：缺口形成日，low=95>90
        {"high": 98, "low": 95, "close": 96},  # day4：沒跌破 90
        {"high": 97, "low": 93, "close": 94},  # day5：沒跌破 90
        {"high": 95, "low": 85, "close": 92},  # day6：跌進缺口(85<=90)又拉回(92>90) -> True
    ]
    df = _mini_frame(rows)
    mask = fvg_fill_mask(df, gap_age=3)
    assert bool(mask.iloc[5]) is True


def test_fvg_fill_does_not_fire_if_already_filled_earlier():
    # 跟上面一樣的缺口，但 day5 就已經跌破缺口下緣（提前回補），
    # day6 即使符合跌進+拉回的表面條件，也不該再算一次「首次回補」
    rows = [
        {"high": 90, "low": 88, "close": 89},  # day1
        {"high": 96, "low": 94, "close": 95},  # day2
        {"high": 100, "low": 95, "close": 98},  # day3：缺口形成
        {"high": 98, "low": 95, "close": 96},  # day4
        {"high": 97, "low": 85, "close": 88},  # day5：提前跌破缺口下緣 90 -> 已回補
        {"high": 95, "low": 85, "close": 92},  # day6：不該再觸發
    ]
    df = _mini_frame(rows)
    mask = fvg_fill_mask(df, gap_age=3)
    assert bool(mask.iloc[5]) is False


def test_fvg_fill_requires_gap_to_exist():
    # 沒有缺口（low[day3] 沒有高於 high[day1]）-> 不該觸發
    rows = [
        {"high": 100, "low": 88, "close": 89},  # day1：high 拉高，蓋掉缺口空間
        {"high": 96, "low": 94, "close": 95},  # day2
        {"high": 100, "low": 95, "close": 98},  # day3：low=95 不高於 high[day1]=100
        {"high": 98, "low": 95, "close": 96},  # day4
        {"high": 97, "low": 93, "close": 94},  # day5
        {"high": 95, "low": 85, "close": 92},  # day6
    ]
    df = _mini_frame(rows)
    mask = fvg_fill_mask(df, gap_age=3)
    assert bool(mask.iloc[5]) is False


def test_market_structure_shift_fires_on_lower_low_then_break_above_swing_high():
    rows = [
        {"low": 100, "high": 105, "close": 103},  # day1
        {"low": 98, "high": 103, "close": 99},  # day2
        {"low": 95, "high": 100, "close": 96},  # day3
        {"low": 90, "high": 97, "close": 91},  # day4：近期(day3~4)低點 90 < 較早(day1~2)低點 98
        {"low": 99, "high": 115, "close": 101},  # day5：收盤突破 day3~4 的波段高點(100) -> True
    ]
    df = _mini_frame(rows)
    mask = market_structure_shift_mask(df, downtrend_window=2, swing_window=2)
    assert bool(mask.iloc[4]) is True


def test_market_structure_shift_requires_downtrend_context():
    # 近期低點沒有比更早的低點更低（一路盤堅），即使突破波段高點也不該算 MSS
    rows = [
        {"low": 100, "high": 105, "close": 103},  # day1
        {"low": 101, "high": 106, "close": 104},  # day2
        {"low": 102, "high": 107, "close": 105},  # day3
        {"low": 103, "high": 108, "close": 106},  # day4：近期低點(103)沒有比更早的低點(100)低
        {"low": 104, "high": 120, "close": 121},  # day5：雖然突破波段高點，但沒有空頭結構前提
    ]
    df = _mini_frame(rows)
    mask = market_structure_shift_mask(df, downtrend_window=2, swing_window=2)
    assert bool(mask.iloc[4]) is False
