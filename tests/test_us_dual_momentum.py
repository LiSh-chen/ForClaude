"""測試 scripts/test_us_dual_momentum_from_db.py 的濾網訊號函式本身
（不連網、不用真的跑回測引擎）：多頭日該跟預設動量排名一致，空頭日該
全部變成 NaN，而且用的是 T-1 資訊、不會偷看當天還沒收盤的 QQQ 價格。
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from tw_quant import indicators as ind
from test_us_dual_momentum_from_db import make_trend_filtered_momentum_signal_fn  # noqa: E402


def _master(dates, stock_id="AAA", closes=None):
    closes = closes if closes is not None else [100.0 + i for i in range(len(dates))]
    return pd.DataFrame({"date": dates, "stock_id": [stock_id] * len(dates), "close": closes})


def test_bull_days_match_default_momentum_ranking():
    dates = pd.bdate_range("2024-01-01", periods=10)
    master = _master(dates)
    # QQQ 全程上升趨勢 -> 均線以上 -> 全部應該是多頭日
    qqq = pd.DataFrame({"date": dates, "close": [50.0 + i for i in range(len(dates))]})

    signal_fn = make_trend_filtered_momentum_signal_fn(qqq, trend_ma_window=3, momentum_window=2)
    result = signal_fn(master)

    ret = master.groupby("stock_id", sort=False)["close"].pct_change(2)
    expected_default = ind.shift_by_group(ret, master, periods=1)

    # 前幾天均線暖身期還沒好，QQQ 端會是 NaN -> bull=False -> 濾網後也是 NaN，
    # 跟純動量的差異只會出現在暖身期，暖身期過後兩者應該完全一致
    warm = 3  # rolling(3) 暖身
    pd.testing.assert_series_equal(
        result.iloc[warm:].reset_index(drop=True), expected_default.iloc[warm:].reset_index(drop=True)
    )


def test_bear_days_force_nan_regardless_of_momentum():
    dates = pd.bdate_range("2024-01-01", periods=10)
    master = _master(dates)
    # QQQ 全程下跌 -> 遠低於均線 -> 全部應該是空頭日 -> 濾網後全部 NaN
    qqq = pd.DataFrame({"date": dates, "close": [100.0 - i * 5 for i in range(len(dates))]})

    signal_fn = make_trend_filtered_momentum_signal_fn(qqq, trend_ma_window=3, momentum_window=2)
    result = signal_fn(master)

    assert result.isna().all()


def test_regime_flip_uses_t_minus_1_information_not_same_day():
    dates = pd.bdate_range("2024-01-01", periods=6)
    master = _master(dates)
    # QQQ 前 4 天緩步盤整、第 5 天(index 4)收盤大漲站上均線 -> 應該要到「下一天」
    # (index 5) 濾網才會判定為多頭，第 5 天當天本身不該因為自己的漲幅而被視為多頭
    closes = [100.0, 100.5, 99.5, 100.0, 120.0, 121.0]
    qqq = pd.DataFrame({"date": dates, "close": closes})

    signal_fn = make_trend_filtered_momentum_signal_fn(qqq, trend_ma_window=3, momentum_window=2)
    result = signal_fn(master)

    assert pd.isna(result.iloc[4])  # 大漲當天本身依然是空頭日（用的是前一天的資訊）
    assert not pd.isna(result.iloc[5])  # 隔天才反映站上均線


def test_all_bear_means_no_stock_eligible_across_multiple_stocks():
    dates = pd.bdate_range("2024-01-01", periods=8)
    master = pd.concat(
        [_master(dates, "AAA", [10.0 + i for i in range(len(dates))]), _master(dates, "BBB", [20.0 + i * 2 for i in range(len(dates))])],
        ignore_index=True,
    )
    qqq = pd.DataFrame({"date": dates, "close": [100.0 - i * 3 for i in range(len(dates))]})

    signal_fn = make_trend_filtered_momentum_signal_fn(qqq, trend_ma_window=3, momentum_window=2)
    result = signal_fn(master)

    assert result.isna().all()
