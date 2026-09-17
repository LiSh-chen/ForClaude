"""FinMindDataProvider 欄位對應的迴歸測試。

用真實 GitHub Actions 執行 scripts/_debug_finmind.py（已刪除的診斷腳本）
實測到的 TaiwanStockPrice 原始 JSON 結構 mock 掉 requests.get，
不觸碰真實網路（這個沙盒連不到 api.finmindtrade.com）。

這個測試存在的原因：fetch_price() 曾經因為欄位選取邏輯重複 append
["date", "stock_id"]，選出帶重複欄名的 DataFrame，導致下游
pd.to_datetime(d["date"]) 把 d["date"]（變成 DataFrame 而非 Series）誤判成
要組裝 year/month/day，拋出 "cannot assemble with duplicate keys"——這個
bug 在單元測試裡完全測不出來，是實際跑 GitHub Actions workflow 才發現的，
所以在這裡把真實回傳結構釘死，防止回歸。
"""

from unittest.mock import MagicMock, patch

from tw_quant.data_provider import FinMindDataProvider

# 2026-09-17 從 GitHub Actions 實測 TaiwanStockPrice 拿到的真實欄位形狀
REAL_FINMIND_PRICE_RECORD = {
    "date": "2024-01-02",
    "stock_id": "2330",
    "Trading_Volume": 27997826,
    "Trading_money": 16549619798,
    "open": 590.0,
    "max": 593.0,
    "min": 589.0,
    "close": 593.0,
    "spread": 0.0,
    "Trading_turnover": 20667,
}


def _mock_response(data):
    resp = MagicMock()
    resp.status_code = 200
    resp.raise_for_status.return_value = None
    resp.json.return_value = {"msg": "success", "status": 200, "data": data}
    return resp


def test_fetch_price_returns_no_duplicate_columns():
    with patch("requests.get", return_value=_mock_response([REAL_FINMIND_PRICE_RECORD])):
        provider = FinMindDataProvider()
        df = provider.fetch_price("2330", "2024-01-01", "2024-01-10")

    assert not df.columns.duplicated().any()
    assert set(df.columns) == {"date", "stock_id", "open", "high", "low", "close", "volume", "turnover_value"}


def test_fetch_price_date_column_is_a_series_not_a_dataframe():
    with patch("requests.get", return_value=_mock_response([REAL_FINMIND_PRICE_RECORD])):
        provider = FinMindDataProvider()
        df = provider.fetch_price("2330", "2024-01-01", "2024-01-10")

    # d["date"] 若因重複欄名變成 DataFrame，這裡的 .dt accessor 會直接拋錯
    assert df["date"].dt.year.iloc[0] == 2024


def test_fetch_price_maps_finmind_field_names_correctly():
    with patch("requests.get", return_value=_mock_response([REAL_FINMIND_PRICE_RECORD])):
        provider = FinMindDataProvider()
        df = provider.fetch_price("2330", "2024-01-01", "2024-01-10")

    row = df.iloc[0]
    assert row["volume"] == 27997826  # Trading_Volume
    assert row["turnover_value"] == 16549619798  # Trading_money
    assert row["high"] == 593.0  # max
    assert row["low"] == 589.0  # min
