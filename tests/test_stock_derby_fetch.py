import importlib.util
from pathlib import Path

import pandas as pd

spec = importlib.util.spec_from_file_location(
    "fetch_prices", Path(__file__).parent.parent / "stock-derby" / "fetch_prices.py")
fp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fp)


def frame(rows):
    return pd.DataFrame(rows).set_index("date")


def test_merge_overlap_new_wins_and_appends():
    old = frame([{"date": "2026-01-02", "A": 1.0}, {"date": "2026-01-03", "A": 2.0}])
    new = frame([{"date": "2026-01-03", "A": 2.5}, {"date": "2026-01-06", "A": 3.0}])
    out = fp.merge_prices(old, new)
    assert list(out.index) == ["2026-01-02", "2026-01-03", "2026-01-06"]
    assert out.loc["2026-01-03", "A"] == 2.5


def test_merge_empty_old_and_drops_empty_rows():
    new = frame([{"date": "2026-01-02", "A": None}, {"date": "2026-01-03", "A": 1.0}])
    out = fp.merge_prices(pd.DataFrame(), new)
    assert list(out.index) == ["2026-01-03"]
