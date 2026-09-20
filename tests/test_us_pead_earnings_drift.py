"""測試美股財報驚喜訊號的計算邏輯
（scripts/test_us_pead_earnings_drift_from_db.py）。

用手算過的 EPS 序列驗證驚喜幅度公式對不對，並確認訊號真的用財報公布日
本身當基準日期、shift(1) 反未來函數規則有生效（訊號不會提早出現在財報
公布之前的交易日上，公布當天也還看不到，要等下一個交易日）。
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.test_us_pead_earnings_drift_from_db import _earnings_surprise_frame, attach_earnings_signal


def _make_earnings(rows: list[tuple[str, str, float, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"date": pd.Timestamp(date), "stock_id": stock_id, "eps_estimate": est, "eps_actual": act}
            for stock_id, date, est, act in rows
        ]
    )


def test_surprise_matches_hand_computed_value():
    earnings = _make_earnings([("AAPL", "2024-01-25", 1.0, 1.2)])  # (1.2-1.0)/1.0 = 0.2
    out = _earnings_surprise_frame(earnings)

    assert len(out) == 1
    assert out["stock_id"].iloc[0] == "AAPL"
    assert out["known_date"].iloc[0] == pd.Timestamp("2024-01-25")
    assert abs(out["surprise"].iloc[0] - 0.2) < 1e-9


def test_surprise_handles_negative_estimate_using_absolute_value_denominator():
    # 預期虧損 -1.0，實際只虧 -0.5（比預期好），驚喜幅度該是正的
    earnings = _make_earnings([("S1", "2024-01-25", -1.0, -0.5)])
    out = _earnings_surprise_frame(earnings)
    assert abs(out["surprise"].iloc[0] - 0.5) < 1e-9  # (-0.5 - -1.0) / abs(-1.0) = 0.5


def test_surprise_drops_rows_with_near_zero_estimate_to_avoid_division_blowup():
    earnings = _make_earnings([("S1", "2024-01-25", 0.0, 0.05)])
    out = _earnings_surprise_frame(earnings)
    assert out.empty


def test_surprise_drops_rows_without_actual_eps_yet():
    # 還沒公布的未來排定財報日（eps_actual 是 NaN）不該產生驚喜訊號
    earnings = pd.DataFrame(
        [{"date": pd.Timestamp("2026-11-01"), "stock_id": "AAPL", "eps_estimate": 1.5, "eps_actual": None}]
    )
    out = _earnings_surprise_frame(earnings)
    assert out.empty


def test_earnings_surprise_frame_handles_empty_input():
    out = _earnings_surprise_frame(pd.DataFrame(columns=["date", "stock_id", "eps_estimate", "eps_actual"]))
    assert out.empty
    assert list(out.columns) == ["stock_id", "known_date", "surprise"]


def test_attach_earnings_signal_only_appears_after_report_date_shifted_by_one_day():
    dates = pd.bdate_range("2024-01-01", "2024-03-31")
    prices = pd.DataFrame(
        {
            "date": dates, "stock_id": "S1", "industry": "IND",
            "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0,
            "volume": 100000, "turnover_value": 1e7,
        }
    )
    report_date = pd.Timestamp("2024-01-25")
    earnings = _make_earnings([("S1", "2024-01-25", 1.0, 1.3)])  # surprise = 0.3

    master = attach_earnings_signal(prices, earnings)

    before = master[master["date"] < report_date]
    on_report_day = master[master["date"] == report_date]
    after = master[master["date"] > report_date]

    assert before["eps_surprise_shifted"].isna().all()  # 公布前完全沒有訊號
    assert on_report_day["eps_surprise_shifted"].isna().all()  # 公布當天盤前/盤後未知，還看不到
    assert abs(after["eps_surprise_shifted"].dropna().iloc[0] - 0.3) < 1e-9  # 下一個交易日才看到


def test_attach_earnings_signal_uses_most_recent_known_surprise_per_stock():
    dates = pd.bdate_range("2024-01-01", "2024-06-30")
    prices = pd.DataFrame(
        {
            "date": dates, "stock_id": "S1", "industry": "IND",
            "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0,
            "volume": 100000, "turnover_value": 1e7,
        }
    )
    earnings = _make_earnings([("S1", "2024-01-25", 1.0, 1.1), ("S1", "2024-04-25", 1.0, 0.8)])

    master = attach_earnings_signal(prices, earnings)

    mid = master[(master["date"] > pd.Timestamp("2024-01-26")) & (master["date"] < pd.Timestamp("2024-04-25"))]
    late = master[master["date"] > pd.Timestamp("2024-04-26")]

    assert abs(mid["eps_surprise_shifted"].iloc[0] - 0.1) < 1e-9
    assert abs(late["eps_surprise_shifted"].iloc[-1] - (-0.2)) < 1e-9


def test_attach_earnings_signal_handles_no_matching_earnings_data():
    dates = pd.bdate_range("2024-01-01", periods=5)
    prices = pd.DataFrame(
        {
            "date": dates, "stock_id": "GHOST", "industry": "IND",
            "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0,
            "volume": 100000, "turnover_value": 1e7,
        }
    )
    master = attach_earnings_signal(prices, pd.DataFrame(columns=["date", "stock_id", "eps_estimate", "eps_actual"]))
    assert master["eps_surprise_shifted"].isna().all()
