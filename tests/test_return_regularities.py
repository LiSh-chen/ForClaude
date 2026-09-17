"""測試 analyze_return_regularities_from_db.py 的 IC 計算邏輯（不是驗證真的有
規律，只驗證數學算法本身正確：已知有動量的合成資料應該量到正的 IC）。
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.analyze_return_regularities_from_db import _ic_summary, _rank_ic_series


def _make_perfectly_momentum_data() -> pd.DataFrame:
    """建構一個訊號跟未來報酬完全依排名對應（正相關）的假資料，用來驗證
    IC 計算流程本身沒有算錯方向或算錯公式。
    """
    rng = np.random.default_rng(0)
    dates = pd.date_range("2020-01-01", periods=40, freq="B")
    rows = []
    for d in dates:
        n = 50
        signal = rng.permutation(n).astype(float)  # 每天訊號的排名是隨機打散的 0..n-1
        fwd = signal + rng.normal(0, 0.01, n)  # 未來報酬幾乎完全跟訊號排名同向
        for i in range(n):
            rows.append({"date": d, "stock_id": f"S{i}", "sig": signal[i], "fwd": fwd[i]})
    return pd.DataFrame(rows)


def test_rank_ic_detects_known_positive_relationship():
    df = _make_perfectly_momentum_data()
    ic = _rank_ic_series(df, "sig", "fwd")
    summary = _ic_summary(ic)
    assert summary["mean_ic"] > 0.9  # 幾乎完美正相關，IC 應該接近 1
    assert summary["n_days"] == 40


def test_rank_ic_near_zero_for_pure_noise():
    rng = np.random.default_rng(1)
    dates = pd.date_range("2020-01-01", periods=60, freq="B")
    rows = []
    for d in dates:
        n = 50
        for i in range(n):
            rows.append(
                {"date": d, "stock_id": f"S{i}", "sig": rng.normal(), "fwd": rng.normal()}
            )
    df = pd.DataFrame(rows)
    ic = _rank_ic_series(df, "sig", "fwd")
    summary = _ic_summary(ic)
    assert abs(summary["mean_ic"]) < 0.15  # 純雜訊，IC 應該接近 0（留寬鬆容差避免隨機性造成偶發失敗）


def test_rank_ic_skips_days_below_min_sample():
    df = pd.DataFrame(
        {
            "date": [pd.Timestamp("2020-01-01")] * 5,
            "stock_id": [f"S{i}" for i in range(5)],
            "sig": [1.0, 2.0, 3.0, 4.0, 5.0],
            "fwd": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )
    ic = _rank_ic_series(df, "sig", "fwd")
    assert ic.empty  # 只有 5 檔股票，低於 MIN_STOCKS_PER_DAY，該天應該被排除
