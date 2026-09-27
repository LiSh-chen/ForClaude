"""測試 tw_quant/us_market_signal.py：美股跨市場訊號的讀取與日期對齊。"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tw_quant.us_market_signal import align_to_txf_dates, load_us_daily  # noqa: E402


def test_load_us_daily_has_expected_columns_and_no_gaps_in_price():
    d = load_us_daily()
    assert set(["date", "sp500_close", "vix_close", "sp500_ret", "vix_chg"]).issubset(d.columns)
    assert d["sp500_close"].notna().all()
    assert d["vix_close"].notna().all()
    assert d["date"].is_monotonic_increasing
    assert not d["date"].duplicated().any()


def test_load_us_daily_matches_known_historical_values():
    d = load_us_daily().set_index("date")
    # 這幾個日期有公開歷史記錄可以對照（見模組docstring的查證說明）
    assert d.loc["2008-09-15", "sp500_close"] == 1192.70  # 雷曼兄弟倒閉當天
    assert d.loc["2020-03-23", "sp500_close"] == 2237.3999  # COVID崩盤最低點
    assert d.loc["2023-12-29", "sp500_close"] == 4769.8301  # 跟TXF資料同一天


def test_align_to_txf_dates_never_uses_same_or_future_us_date():
    txf_dates = pd.to_datetime(["2001-01-02", "2001-01-03", "2010-06-15", "2023-12-29"])
    aligned = align_to_txf_dates(pd.Series(txf_dates))
    assert (aligned["us_date"] < aligned["txf_date"]).all()


def test_align_to_txf_dates_handles_holiday_gaps_without_nulls():
    # 2001-01-02(週二)是台股開盤但美股1/1剛過元旦假期，兩邊行事曆本來就
    # 不完全對齊，這裡確認backward asof不會因此產生NaN或錯誤配對
    txf_dates = pd.to_datetime(["2001-01-02"])
    aligned = align_to_txf_dates(pd.Series(txf_dates))
    assert aligned["us_date"].notna().all()
    assert aligned.iloc[0]["us_date"] < aligned.iloc[0]["txf_date"]
