"""測試 scripts/probe_us_fundamentals_data.py 的純邏輯（抽樣、模糊欄位比對、
探測摘要），不連網、用假 provider 替換掉真的 yfinance 呼叫。
"""

import sys
from pathlib import Path
from unittest import mock

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from probe_us_fundamentals_data import (  # noqa: E402
    _find_col_label,
    _find_row_label,
    _is_quarter_end_date,
    pick_sample_tickers,
    probe_one,
)


def test_pick_sample_tickers_returns_all_when_fewer_than_n():
    assert pick_sample_tickers(["C", "A", "B"], 10) == ["A", "B", "C"]


def test_pick_sample_tickers_spreads_evenly_across_full_range():
    all_ids = [f"S{i:03d}" for i in range(100)]
    sample = pick_sample_tickers(all_ids, 10)
    assert len(sample) == 10
    # 應該涵蓋清單頭尾，不是只抽到某一段
    assert sample[0] == "S000"
    assert sample[-1] == "S099"


def test_pick_sample_tickers_deduplicates_and_is_deterministic():
    all_ids = [f"S{i:03d}" for i in range(50)]
    sample1 = pick_sample_tickers(all_ids, 20)
    sample2 = pick_sample_tickers(all_ids, 20)
    assert sample1 == sample2
    assert len(sample1) == len(set(sample1))


def test_find_row_label_matches_case_and_space_insensitively():
    index = ["Total Revenue", "Net Income", "Gross Profit"]
    assert _find_row_label(index, ("revenue",)) == "Total Revenue"
    assert _find_row_label(index, ("net", "income")) == "Net Income"
    assert _find_row_label(index, ("nonexistent",)) is None


def test_find_col_label_matches_case_and_space_insensitively():
    columns = ["EPS Estimate", "Reported EPS", "Surprise(%)"]
    assert _find_col_label(columns, ("reported", "eps")) == "Reported EPS"
    assert _find_col_label(columns, ("surprise",)) == "Surprise(%)"
    assert _find_col_label(columns, ("nonexistent",)) is None


def test_is_quarter_end_date():
    assert _is_quarter_end_date(pd.Timestamp("2023-03-31"))
    assert _is_quarter_end_date(pd.Timestamp("2023-12-31"))
    assert not _is_quarter_end_date(pd.Timestamp("2023-01-15"))


def test_probe_one_reports_error_instead_of_raising():
    provider = mock.Mock()
    provider.fetch_quarterly_fundamentals.side_effect = RuntimeError("boom")
    result = probe_one(provider, "BROKEN")
    assert result["stock_id"] == "BROKEN"
    assert "RuntimeError" in result["error"]
    assert "boom" in result["error"]


def test_probe_one_handles_empty_dataframes():
    provider = mock.Mock()
    provider.fetch_quarterly_fundamentals.return_value = {
        "quarterly_income_stmt": pd.DataFrame(),
        "earnings_dates": pd.DataFrame(),
    }
    result = probe_one(provider, "GHOST")
    assert result["error"] is None
    assert result["n_quarters_revenue"] == 0
    assert result["n_earnings_rows"] == 0
    assert result["pct_with_actual_eps"] == 0.0


def test_probe_one_counts_usable_revenue_quarters_and_finds_range():
    cols = pd.to_datetime(["2024-06-30", "2024-03-31", "2023-12-31"])
    income = pd.DataFrame(
        {cols[0]: [100.0, 10.0], cols[1]: [90.0, 9.0], cols[2]: [None, 8.0]},
        index=["Total Revenue", "Net Income"],
    )
    provider = mock.Mock()
    provider.fetch_quarterly_fundamentals.return_value = {
        "quarterly_income_stmt": income,
        "earnings_dates": pd.DataFrame(),
    }
    result = probe_one(provider, "AAPL")
    assert result["has_revenue_row"] is True
    assert result["n_quarters_revenue"] == 2  # 2023-12-31 那欄的 revenue 是 NaN
    assert result["revenue_range"] == (cols[1].date(), cols[0].date())


def test_probe_one_flags_quarter_end_dates_and_missing_actual_eps():
    idx = pd.to_datetime(["2024-03-31", "2024-01-25"])  # 一個卡季末、一個不是
    earnings = pd.DataFrame(
        {"EPS Estimate": [1.0, 1.1], "Reported EPS": [1.05, None]},
        index=idx,
    )
    provider = mock.Mock()
    provider.fetch_quarterly_fundamentals.return_value = {
        "quarterly_income_stmt": pd.DataFrame(),
        "earnings_dates": earnings,
    }
    result = probe_one(provider, "MSFT")
    assert result["n_earnings_rows"] == 2
    assert result["has_actual_eps_col"] is True
    assert result["pct_with_actual_eps"] == 0.5
    assert result["pct_dates_on_quarter_end"] == 0.5
