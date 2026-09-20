"""測試 scripts/test_us_pead_sector_filter_sensitivity_from_db.py 的網格
迴圈本身（結構正確性——列數/欄位對不對），不重新驗證底層回測數字（那些
已經由 tests/test_event_drift_backtest.py 跟
tests/test_us_pead_sector_momentum.py 涵蓋），純合成資料、不連資料庫。
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tw_quant.us_config import build_us_config
from scripts.test_us_pead_sector_filter_sensitivity_from_db import (
    SECTOR_LOOKBACK_GRID,
    TOP_K_SECTORS_GRID,
    run_sector_param_grid,
)


def _synthetic_prices_and_events():
    dates = pd.bdate_range("2020-01-01", periods=400)
    industries = ["A", "B", "C"]
    rng = np.random.default_rng(3)
    rows = []
    for i, industry in enumerate(industries):
        for j in range(3):
            stock_id = f"{industry}{j}"
            rets = rng.normal(0.0005, 0.01, len(dates))
            closes = 100 * (1 + pd.Series(rets)).cumprod().values
            for k, d in enumerate(dates):
                rows.append(
                    {
                        "date": d, "stock_id": stock_id, "industry": industry,
                        "open": closes[k], "high": closes[k] * 1.01, "low": closes[k] * 0.99, "close": closes[k],
                        "volume": 100000, "turnover_value": closes[k] * 100000,
                    }
                )
    prices = pd.DataFrame(rows)

    events = pd.DataFrame(
        [
            {"stock_id": stock_id, "known_date": dates[i], "signal": rng.uniform(0.0, 0.3)}
            for stock_id in prices["stock_id"].unique()
            for i in range(100, 350, 30)
        ]
    )
    return prices, events


def test_run_sector_param_grid_has_expected_shape_and_columns():
    prices, events = _synthetic_prices_and_events()
    base_cfg = build_us_config()

    grid = run_sector_param_grid(prices, events, base_cfg)

    expected_rows = len(SECTOR_LOOKBACK_GRID) * len(TOP_K_SECTORS_GRID) * 2  # 2 periods
    assert len(grid) == expected_rows
    assert set(grid["period"]) == {"in_sample", "oos"}
    assert set(grid["lookback"]) == set(SECTOR_LOOKBACK_GRID)
    assert set(grid["top_k"]) == set(TOP_K_SECTORS_GRID)
    for col in ("sharpe", "total_return", "cagr", "max_dd", "n_trades", "n_events"):
        assert col in grid.columns
