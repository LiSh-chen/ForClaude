"""測試月營收意外訊號的計算邏輯（scripts/test_pead_revenue_drift_from_db.py）。

用手算過的營收序列驗證年增率/意外程度公式對不對，並確認「次月 15 號才算
公開已知」這個反未來函數規則真的有生效（訊號不會提早出現在還沒公告的
交易日上）。
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.test_pead_revenue_drift_from_db import DISCLOSURE_BUFFER_DAY, _revenue_surprise_frame, attach_revenue_signal


def _make_revenue(stock_id: str, monthly_values: dict[tuple[int, int], float]) -> pd.DataFrame:
    rows = [
        {"date": pd.Timestamp(year=y, month=m, day=1), "stock_id": stock_id, "revenue": v, "revenue_year": y, "revenue_month": m}
        for (y, m), v in monthly_values.items()
    ]
    return pd.DataFrame(rows)


def test_yoy_surprise_matches_hand_computed_value():
    # 連續 19 個月，前 13 個月都是 100，第 19 個月（跟第 7 個月同月份）
    # 突然變成 150 -> 該月年增率 = 150/100-1 = 50%，明顯高於近 6 個月的平均年增率（0%）
    monthly = {(2022, m): 100.0 for m in range(1, 13)}
    monthly.update({(2023, m): 100.0 for m in range(1, 7)})
    monthly[(2023, 7)] = 150.0  # 對應 (2022,7) 的隔年同月

    revenue = _make_revenue("S1", monthly)
    out = _revenue_surprise_frame(revenue)

    row = out[(out["stock_id"] == "S1")]
    row_2023_07 = row[row["known_date"] == pd.Timestamp("2023-08-" + f"{DISCLOSURE_BUFFER_DAY:02d}")]
    assert len(row_2023_07) == 1
    assert abs(row_2023_07["surprise"].iloc[0] - 0.5) < 1e-9  # yoy_growth=0.5, trailing_avg 前面全是 0% -> surprise=0.5


def test_known_date_is_next_month_with_buffer_day():
    # 需要至少 12（算年增率）+ 6（算近半年平均年增率的暖身）= 18 個月連續資料，
    # 才會有第一筆非 NaN 的 surprise，所以這裡給足 26 個月歷史
    monthly = {(2021, m): 100.0 for m in range(1, 13)}
    monthly.update({(2022, m): 100.0 for m in range(1, 13)})
    monthly.update({(2023, m): 110.0 for m in range(1, 3)})
    revenue = _make_revenue("S1", monthly)
    out = _revenue_surprise_frame(revenue)

    assert not out.empty
    # 2023 年 2 月營收 -> 應該要到 2023-03-15（次月 + DISCLOSURE_BUFFER_DAY）才算已知
    row = out[(out["stock_id"] == "S1")].sort_values("known_date")
    assert (row["known_date"].dt.day == DISCLOSURE_BUFFER_DAY).all()
    feb_row = row[row["known_date"] == pd.Timestamp(f"2023-03-{DISCLOSURE_BUFFER_DAY:02d}")]
    assert len(feb_row) == 1


def test_attach_revenue_signal_only_appears_after_disclosure_and_not_before():
    dates = pd.bdate_range("2023-01-01", "2023-06-30")
    prices = pd.DataFrame(
        {
            "date": dates, "stock_id": "S1", "industry": "IND",
            "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0,
            "volume": 100000, "turnover_value": 1e7,
        }
    )
    # 需要至少 18 個月連續歷史才會有第一筆非 NaN 的 surprise
    monthly = {(2021, m): 100.0 for m in range(1, 13)}
    monthly.update({(2022, m): 100.0 for m in range(1, 13)})
    monthly[(2023, 1)] = 120.0  # yoy = 0.2，known_date = 2023-02-15
    revenue = _make_revenue("S1", monthly)

    master = attach_revenue_signal(prices, revenue)
    known_date = pd.Timestamp(f"2023-02-{DISCLOSURE_BUFFER_DAY:02d}")

    # 2023-01 之前每個月都是平盤（yoy=0），所以 known_date 之前的訊號應該還是
    # 舊的基準值 0.0，不會提早看到 2023-01 那筆意外值 0.2——這才是真正要驗證的
    # 反未來函數行為，而不是「完全沒有訊號」（歷史夠長，更早的月份本來就有
    # 合法已知的基準訊號）。
    just_before = master[(master["date"] < known_date) & (master["date"] >= known_date - pd.Timedelta(days=10))]
    just_after = master[master["date"] > known_date]

    assert not just_before.empty and not just_after.empty
    assert (just_before["revenue_surprise_shifted"].dropna() == 0.0).all()
    assert abs(just_after["revenue_surprise_shifted"].dropna().iloc[0] - 0.2) < 1e-9
