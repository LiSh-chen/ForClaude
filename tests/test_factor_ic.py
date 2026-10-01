"""測試 tw_quant/factor_ic.py 的 IC 計算邏輯本身（不是驗證真的有規律，
只驗證數學算法本身正確：已知有動量的合成資料應該量到正的 IC），對應
tests/test_return_regularities.py 台股版的同一組驗證，確認抽出來共用的
版本行為完全一致。
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tw_quant.factor_ic import forward_return, ic_summary, rank_ic_series


def _make_perfectly_momentum_data() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    dates = pd.date_range("2020-01-01", periods=40, freq="B")
    rows = []
    for d in dates:
        n = 50
        signal = rng.permutation(n).astype(float)
        fwd = signal + rng.normal(0, 0.01, n)
        for i in range(n):
            rows.append({"date": d, "stock_id": f"S{i}", "sig": signal[i], "fwd": fwd[i]})
    return pd.DataFrame(rows)


def test_rank_ic_detects_known_positive_relationship():
    df = _make_perfectly_momentum_data()
    ic = rank_ic_series(df, "sig", "fwd")
    summary = ic_summary(ic)
    assert summary["mean_ic"] > 0.9
    assert summary["n_days"] == 40


def test_rank_ic_near_zero_for_pure_noise():
    rng = np.random.default_rng(1)
    dates = pd.date_range("2020-01-01", periods=60, freq="B")
    rows = []
    for d in dates:
        n = 50
        for i in range(n):
            rows.append({"date": d, "stock_id": f"S{i}", "sig": rng.normal(), "fwd": rng.normal()})
    df = pd.DataFrame(rows)
    ic = rank_ic_series(df, "sig", "fwd")
    summary = ic_summary(ic)
    assert abs(summary["mean_ic"]) < 0.15


def test_rank_ic_skips_days_below_min_sample():
    df = pd.DataFrame(
        {
            "date": [pd.Timestamp("2020-01-01")] * 5,
            "stock_id": [f"S{i}" for i in range(5)],
            "sig": [1.0, 2.0, 3.0, 4.0, 5.0],
            "fwd": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )
    ic = rank_ic_series(df, "sig", "fwd", min_stocks_per_day=30)
    assert ic.empty


def test_rank_ic_respects_custom_min_stocks_per_day():
    df = pd.DataFrame(
        {
            "date": [pd.Timestamp("2020-01-01")] * 5,
            "stock_id": [f"S{i}" for i in range(5)],
            "sig": [1.0, 2.0, 3.0, 4.0, 5.0],
            "fwd": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )
    ic = rank_ic_series(df, "sig", "fwd", min_stocks_per_day=5)
    assert not ic.empty
    assert np.isclose(ic.iloc[0], 1.0)


def test_forward_return_shifts_within_each_stock_group():
    master = pd.DataFrame(
        {
            "stock_id": ["A", "A", "A", "B", "B", "B"],
            "date": pd.to_datetime(
                ["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-01", "2020-01-02", "2020-01-03"]
            ),
            "close": [100.0, 110.0, 121.0, 50.0, 45.0, 40.5],
        }
    )
    fwd1 = forward_return(master, horizon=1)
    assert np.isclose(fwd1.iloc[0], 0.10)  # A: 100 -> 110
    assert np.isclose(fwd1.iloc[3], -0.10)  # B: 50 -> 45
    assert pd.isna(fwd1.iloc[2])  # A 最後一天沒有未來可看
    assert pd.isna(fwd1.iloc[5])  # B 最後一天同理


def test_ic_summary_returns_nan_for_too_few_days():
    ic = pd.Series([0.1, 0.2, 0.3])
    summary = ic_summary(ic)
    assert np.isnan(summary["mean_ic"])
    assert summary["n_days"] == 0
