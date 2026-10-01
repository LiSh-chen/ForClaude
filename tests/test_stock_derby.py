import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

D = Path(__file__).parent.parent / "stock-derby"
sys.path.insert(0, str(D))
import build_derby as bd  # noqa: E402
import factors as F  # noqa: E402
import fetch_extras as fe  # noqa: E402


def test_style_label_boundaries():
    assert bd.style_label(1.0)[0] == "領放"
    assert bd.style_label(4.0)[0] == "跟前"
    assert bd.style_label(6.0)[0] == "居中"
    assert bd.style_label(9.0)[0] == "後上"


def test_stars_range():
    assert bd.stars(0) == 1 and bd.stars(100) == 5 and bd.stars(55) == 3


def test_ranks_best_is_one_and_nan_ignored():
    r = bd.ranks_of(pd.Series({"a": 0.1, "b": 0.3, "c": np.nan}))
    assert r["b"] == 1 and r["a"] == 2 and np.isnan(r["c"])


def test_window_ranks_uses_only_window_and_handles_missing_history():
    close = pd.DataFrame({"a": [1, 2, 3, 4, 5, 6.0], "b": [1, 1, 1, 1, 1, 1.0], "c": [np.nan] * 3 + [1, 1, 3.0]})
    early, fin = bd.window_ranks(close, ["a", "b", "c"], 0, 5)
    assert fin["a"] == 1 and fin["b"] == 2 and np.isnan(fin["c"])
    assert bd.window_ranks(close, ["a"], -1, 3) is None


def test_factors_no_lookahead():
    idx = pd.bdate_range("2020-01-01", periods=400)
    rng = np.random.default_rng(0)
    close = pd.DataFrame(100 * np.exp(rng.normal(0, .01, (400, 3)).cumsum(0)), index=idx, columns=list("abc"))
    p = {"close": close, "high": close * 1.01, "low": close * .99, "volume": close * 0 + 1e6}
    e = pd.DataFrame(columns=["date", "stock_id", "eps_actual", "surprise_pct"])
    base = F.compute_all(p, e)
    p2 = {k: v.copy() for k, v in p.items()}
    p2["close"].iloc[-1] *= 5                                  # tamper with the last day only
    alt = F.compute_all(p2, e)
    for name in ("mom_12_1", "mom_3m", "high_52w", "vol_60", "rsi_14"):
        pd.testing.assert_frame_equal(base[name].iloc[:-1], alt[name].iloc[:-1])


def test_merge_long_new_wins():
    old = pd.DataFrame({"date": ["2026-01-02", "2026-01-03"], "stock_id": "A", "close": [1.0, 2.0]})
    new = pd.DataFrame({"date": ["2026-01-03", "2026-01-06"], "stock_id": "A", "close": [2.5, 3.0]})
    out = fe.merge_long(old, new)
    assert list(out["close"]) == [1.0, 2.5, 3.0]


def test_committed_derby_json_is_consistent():
    import json
    d = json.loads((D / "data" / "derby.json").read_text())
    assert set(d["pools"]) == {"spx", "ndx", "all"}
    for pool in d["pools"].values():
        for dist, race in pool.items():
            assert len(race["horses"]) == 10
            assert sorted(h["result"] for h in race["horses"]) == list(range(1, 11))
            assert all(len(h["path"]) == race["days"] + 1 for h in race["horses"])
            assert len(d["key_factors"]) == 5
