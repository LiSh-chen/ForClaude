"""測試 generate_stock_universe.py 的篩選邏輯（mock FinMind，不碰真實網路）。"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import generate_stock_universe


class FakeProvider:
    def __init__(self, token=None):
        self.token = token

    def fetch_stock_info(self):
        return pd.DataFrame(
            {
                "stock_id": ["2330", "1101", "0050", "0056", "2330A", "6488KY", "1216", "2317", "9999"],
                "stock_name": ["台積電", "台泥", "元大台灣50", "元大高股息", "台積電特別股", "環球晶", "統一", "鴻海", "統一實"],
                "industry": ["半導體"] * 9,
                "type": ["twse", "twse", "twse", "twse", "twse", "tpex", "twse", "twse", "tpex"],
            }
        )


def test_filters_out_etf_and_derivative_codes(tmp_path, monkeypatch):
    monkeypatch.setattr(generate_stock_universe, "FinMindDataProvider", FakeProvider)
    output = tmp_path / "stock_universe.txt"

    monkeypatch.setattr(
        sys, "argv", ["generate_stock_universe.py", "--limit", "10", "--output", str(output)]
    )
    generate_stock_universe.main()

    stocks = output.read_text().splitlines()
    # 0050/0056（ETF）、2330A（特別股）、6488KY（5碼衍生代號）都應該被排除
    assert set(stocks) == {"1101", "1216", "2317", "2330", "9999"}


def test_respects_limit(tmp_path, monkeypatch):
    monkeypatch.setattr(generate_stock_universe, "FinMindDataProvider", FakeProvider)
    output = tmp_path / "stock_universe.txt"

    monkeypatch.setattr(
        sys, "argv", ["generate_stock_universe.py", "--limit", "2", "--output", str(output)]
    )
    generate_stock_universe.main()

    stocks = output.read_text().splitlines()
    assert len(stocks) == 2
    assert stocks == sorted(stocks)  # 依代號排序取前 N 檔
