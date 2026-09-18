"""測試股本計算/分組邏輯（scripts/test_market_cap_effect_from_db.py）。"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.test_market_cap_effect_from_db import compute_share_capital_yi, split_universe_by_cap


def test_compute_share_capital_uses_par_value_10_and_earliest_reading():
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-06-01", "2023-01-01", "2023-06-01"]),
            "stock_id": ["A", "A", "B", "B"],
            "shares_issued": [1e9, 2e9, 5e8, 6e8],  # A 的股本後來變大，B 的也是，但都該用「最早一筆」
        }
    )
    result = compute_share_capital_yi(df)
    # A 最早一筆（2024-01-01）是 1e9 股，股本 = 1e9*10/1e8 = 100 億
    assert abs(result["A"] - 100.0) < 1e-9
    # B 最早一筆（2023-01-01）是 5e8 股，股本 = 5e8*10/1e8 = 50 億
    assert abs(result["B"] - 50.0) < 1e-9


def test_split_universe_by_cap_uses_median_and_covers_all_stocks():
    share_capital = pd.Series({"A": 10.0, "B": 20.0, "C": 30.0, "D": 40.0})
    small, large = split_universe_by_cap(share_capital)

    assert set(small) | set(large) == {"A", "B", "C", "D"}
    assert set(small) & set(large) == set()
    # 中位數是 25，A/B（<=25）該在小股本組，C/D（>25）該在大股本組
    assert set(small) == {"A", "B"}
    assert set(large) == {"C", "D"}
