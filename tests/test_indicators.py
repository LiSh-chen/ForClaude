import numpy as np
import pandas as pd

from tw_quant import indicators as ind


def _make_df(values, stock_id="A"):
    n = len(values)
    return pd.DataFrame(
        {
            "stock_id": [stock_id] * n,
            "date": pd.bdate_range("2020-01-01", periods=n),
            "close": values,
            "high": [v + 0.5 for v in values],
            "low": [v - 0.5 for v in values],
            "volume": [1000] * n,
        }
    )


def test_sma_basic():
    df = _make_df([1, 2, 3, 4, 5])
    result = ind.sma(df, "close", 3)
    assert np.isnan(result.iloc[0])
    assert np.isnan(result.iloc[1])
    assert result.iloc[2] == 2.0
    assert result.iloc[4] == 4.0


def test_sma_does_not_cross_stock_groups():
    df = pd.concat([_make_df([1, 1, 1], "A"), _make_df([100, 100, 100], "B")], ignore_index=True)
    result = ind.sma(df, "close", 2)
    b_rows = df["stock_id"] == "B"
    assert (result[b_rows].dropna() == 100).all()


def test_rolling_percentile_rank_matches_manual_calc():
    s = pd.Series([5, 1, 2, 3, 4, 10, 1])
    df = pd.DataFrame({"stock_id": ["A"] * len(s), "v": s.values})
    pr = ind.rolling_percentile_rank(df, "v", 5)
    # window [5,1,2,3,4] -> last value 4 is 4th smallest of 5 => 80
    assert pr.iloc[4] == 80.0
    # window [1,2,3,4,10] -> last value 10 is largest => 100
    assert pr.iloc[5] == 100.0


def test_atr_positive_and_widens_with_volatility():
    calm = [10.0] * 30
    df_calm = _make_df(calm)
    atr_calm = ind.atr(df_calm, 14)
    assert (atr_calm.dropna() >= 0).all()

    volatile_high = [10 + (5 if i % 2 == 0 else 0) for i in range(30)]
    df_vol = pd.DataFrame(
        {
            "stock_id": ["A"] * 30,
            "date": pd.bdate_range("2020-01-01", periods=30),
            "close": [10.0] * 30,
            "high": volatile_high,
            "low": [9.5] * 30,
            "volume": [1000] * 30,
        }
    )
    atr_vol = ind.atr(df_vol, 14)
    assert atr_vol.iloc[-1] > atr_calm.iloc[-1]


def test_shift_by_group_respects_group_boundaries():
    df = pd.concat([_make_df([1, 2, 3], "A"), _make_df([9, 8, 7], "B")], ignore_index=True)
    s = pd.Series(df["close"].values, index=df.index)
    shifted = ind.shift_by_group(s, df, periods=1)
    # 每組第一筆位移後應為 NaN，不應該吃到另一組的值
    first_a_idx = df[df["stock_id"] == "A"].index[0]
    first_b_idx = df[df["stock_id"] == "B"].index[0]
    assert np.isnan(shifted.loc[first_a_idx])
    assert np.isnan(shifted.loc[first_b_idx])
