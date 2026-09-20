"""測試 scripts/test_us_pead_sector_momentum_from_db.py 的族群動能計算跟
事件篩選邏輯（純合成數字，手算驗證反未來函數，不連資料庫）。"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.test_us_pead_sector_momentum_from_db import compute_sector_momentum_ranks, filter_events_by_sector_momentum


def _price_row(stock_id: str, date, industry: str, close: float) -> dict:
    return {
        "date": date, "stock_id": stock_id, "industry": industry,
        "open": close, "high": close * 1.01, "low": close * 0.99, "close": close,
        "volume": 100000, "turnover_value": close * 100000,
    }


def _closes_from_returns(base: float, rets: list[float]) -> list[float]:
    closes = [base]
    for r in rets:
        closes.append(closes[-1] * (1 + r))
    return closes


def test_sector_momentum_shifts_by_one_day_anti_lookahead():
    dates = pd.bdate_range("2024-01-01", periods=10)
    # A: 全部平盤，只有第5天(index5)暴漲 50%；B 不存在，這裡只測單一產業
    rets_a = [0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0]
    closes_a = _closes_from_returns(100.0, rets_a)
    prices = pd.DataFrame([_price_row("A1", dates[i], "A", closes_a[i]) for i in range(10)])

    ranks = compute_sector_momentum_ranks(prices, lookback_days=3)
    by_date = ranks.set_index("date")["trailing_ret_shifted"]

    # 暴漲當天（index5）的動能分數還不該反映當天自己的漲幅
    assert by_date[dates[5]] == pytest.approx(0.0)
    # 隔一天（index6）才反映出前一天的暴漲
    assert by_date[dates[6]] == pytest.approx(0.5, rel=1e-6)


def test_sector_momentum_excludes_stocks_without_industry():
    dates = pd.bdate_range("2024-01-01", periods=5)
    rows = [_price_row("A1", d, "A", 100.0) for d in dates]
    rows += [_price_row("GHOST", d, "", 999.0) for d in dates]  # 沒有產業分類，不該污染 A 的平均
    prices = pd.DataFrame(rows)

    ranks = compute_sector_momentum_ranks(prices, lookback_days=2)
    assert "" not in set(ranks["industry"])
    assert set(ranks["industry"]) == {"A"}


def test_filter_events_keeps_only_currently_hot_sector():
    dates = pd.bdate_range("2024-01-01", periods=10)
    # B 全程溫和上漲（穩定的熱門族群基準），A 平盤直到第5天暴漲才變熱門
    rets_a = [0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0]
    rets_b = [0.01] * 9
    closes_a = _closes_from_returns(100.0, rets_a)
    closes_b = _closes_from_returns(100.0, rets_b)
    prices = pd.DataFrame(
        [_price_row("A1", dates[i], "A", closes_a[i]) for i in range(10)]
        + [_price_row("B1", dates[i], "B", closes_b[i]) for i in range(10)]
    )

    events = pd.DataFrame(
        [
            {"stock_id": "B1", "known_date": dates[4], "signal": 0.2},  # B 熱門期間 -> 該保留
            {"stock_id": "A1", "known_date": dates[4], "signal": 0.2},  # A 還沒熱起來 -> 該丟掉
            {"stock_id": "A1", "known_date": dates[6], "signal": 0.2},  # A 暴漲後隔天已經熱起來 -> 該保留
            {"stock_id": "B1", "known_date": dates[6], "signal": 0.2},  # 這時 A 更熱，B 被擠出前1名 -> 該丟掉
        ]
    )

    filtered = filter_events_by_sector_momentum(events, prices, lookback_days=3, top_k_sectors=1)
    kept = set(zip(filtered["stock_id"], filtered["known_date"]))

    assert (("B1", dates[4]) in kept) and (("A1", dates[4]) not in kept)
    assert (("A1", dates[6]) in kept) and (("B1", dates[6]) not in kept)


def test_filter_events_drops_stocks_without_industry():
    dates = pd.bdate_range("2024-01-01", periods=10)
    rets_a = [0.01] * 9
    closes_a = _closes_from_returns(100.0, rets_a)
    prices = pd.DataFrame(
        [_price_row("A1", dates[i], "A", closes_a[i]) for i in range(10)]
        + [_price_row("GHOST", dates[i], "", 100.0) for i in range(10)]
    )
    events = pd.DataFrame([{"stock_id": "GHOST", "known_date": dates[5], "signal": 0.2}])

    filtered = filter_events_by_sector_momentum(events, prices, lookback_days=3, top_k_sectors=5)
    assert filtered.empty


def test_filter_events_drops_events_before_lookback_warms_up():
    dates = pd.bdate_range("2024-01-01", periods=10)
    rets_a = [0.01] * 9
    closes_a = _closes_from_returns(100.0, rets_a)
    prices = pd.DataFrame([_price_row("A1", dates[i], "A", closes_a[i]) for i in range(10)])
    # 太早的事件（index1），lookback_days=5 根本還沒有任何一天算得出動能分數
    events = pd.DataFrame([{"stock_id": "A1", "known_date": dates[1], "signal": 0.2}])

    filtered = filter_events_by_sector_momentum(events, prices, lookback_days=5, top_k_sectors=5)
    assert filtered.empty
